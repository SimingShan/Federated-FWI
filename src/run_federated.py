from datetime import datetime
import os
import glob
import copy
from typing import List, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import torch
import src.utils.pytorch_ssim as pytorch_ssim
import torch.multiprocessing as mp
import flwr as fl
from flwr.common import ndarrays_to_parameters
import torch.nn.functional as F
from flwr.server.strategy import FedAvg, FedAvgM, FedProx
from src.utils.data_trans import (
    s_normalize_none,
    v_denormalize,
)
import src.utils.data_trans as data_trans
from src.diffusion_models.diffusion_model import Unet, GaussianDiffusion
from src.federated_learning.flwr_client import client_fn_factory
from src.federated_learning.flwr_evaluation import get_evaluate_fn
from src.federated_learning.flwr_utils import (
    tensor_to_ndarrays, ndarrays_to_tensor, fit_metrics_fn, get_fit_config_fn
)
from src.pde_solvers.client_pde_solver import FWIForward

def run_full_experiment(
    config_path: str = None,
    target_families: Optional[List[str]] = None,
    target_instances: Optional[List[int]] = None,
    config_override=None,
    batched: bool = False,
):
    """
    Loads configuration, runs FL experiments and saves results.

    When batched=True, all families are loaded into a single batch tensor and
    run through ONE FL simulation (combined dataset). Otherwise, each
    (family, instance) pair gets its own simulation.
    """
    assert config_path is not None or config_override is not None, "config_path or config_override is required"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if config_override is not None:
        config = config_override
    else:
        config = OmegaConf.load(config_path)
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- STARTING FULL EXPERIMENT RUN ---")
    print(f"Strategy: {config.experiment.strategy}, Regularization: {config.experiment.regularization}, Scenario: {config.experiment.scenario_flag}")
    if batched:
        print(f"Batched mode: {len(target_families)} families in one simulation")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build directory name with all sweep-relevant params for uniqueness
    _rl = config.experiment.reg_lambda
    _lr = config.federated.local_lr
    _np = getattr(config.experiment, 'num_patches', 'none')
    _sm = getattr(config.experiment, 'server_momentum', 'none')
    base_dir_name = (
        f"main_{config.experiment.strategy}_{config.experiment.regularization}_"
        f"rl{_rl}_lr{_lr}_np{_np}_sm{_sm}_{config.experiment.scenario_flag}"
    )

    # Find existing directories; prefer those with most instance results
    candidate_patterns = [
        os.path.join(config.path.output_path, f"{base_dir_name}_*"),
        # Older non-lambda naming
        os.path.join(config.path.output_path, f"main_{config.experiment.strategy}_{config.experiment.regularization}_{config.experiment.scenario_flag}_*"),
        # Generic lambda-aware pattern
        os.path.join(config.path.output_path, f"main_{config.experiment.strategy}_{config.experiment.regularization}_*_{config.experiment.scenario_flag}_*"),
    ]
    existing_dirs: List[str] = []
    seen = set()
    for pat in candidate_patterns:
        for p in glob.glob(pat):
            if p not in seen:
                seen.add(p)
                existing_dirs.append(p)
    if existing_dirs:
        def dir_score(d: str) -> Tuple[int, float]:
            count = len(glob.glob(os.path.join(d, "*_result.pkl")))
            return (count, os.path.getctime(d))
        main_output_dir = max(existing_dirs, key=dir_score)
        print(f"Using existing directory: {main_output_dir}")
    else:
        # Create new directory
        main_output_dir = os.path.join(config.path.output_path, f"{base_dir_name}_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        print(f"Created new directory: {main_output_dir}")

    # --- 1. SETUP SHARED COMPONENTS ---
    # These components are the same for all runs.
    # Forward Solver
    ctx = {'n_grid': config.forward.n_grid, 'nt': config.forward.nt, 'dx': config.forward.dx,
           'nbc': config.forward.nbc, 'dt': config.forward.dt, 'f': config.forward.f,
           'sz': config.forward.sz, 'gz': config.forward.gz, 'ng': config.forward.ng,
           'ns': getattr(config.forward, 'ns', 10)}
    fwi_forward = FWIForward(ctx, device, normalize=True, v_denorm_func=v_denormalize, s_norm_func=s_normalize_none)

    # Build per-client forward solvers for split scenarios.
    # Each client gets its own geographic subset of sources and receivers.
    per_client_fwi_forwards = None
    scenario_flag = config.experiment.scenario_flag
    if scenario_flag in ('geo_split', 'snr_split', 'freq_split', 'density_split'):

        n_grid = config.forward.n_grid
        ns = getattr(config.forward, 'ns', 10)
        ng = config.forward.ng
        dx = config.forward.dx
        # All source positions (physical units)
        sx_all = np.linspace(0, n_grid - 1, num=ns) * dx
        # Geographic split: left half / right half
        n_half_src = ns // 2      # 5
        n_half_recv = ng // 2     # 95
        client_sx = [sx_all[:n_half_src], sx_all[n_half_src:]]
        # Default: both clients get dense receivers in their half
        client_gx = [np.arange(n_half_recv), np.arange(n_half_recv, ng)]

        # density_split: client 2 gets sparse receivers (every Nth grid point)
        if scenario_flag == 'density_split':
            strides = list(config.forward.client_receiver_strides)  # e.g. [1, 3]
            client_gx = [np.arange(0, n_half_recv, strides[0]),
                         np.arange(n_half_recv, ng, strides[1])]

        per_client_fwi_forwards = []
        for c in range(config.experiment.num_clients):
            client_ctx = copy.deepcopy(ctx)
            client_ctx['sx'] = client_sx[c]
            client_ctx['gx'] = client_gx[c]
            client_ctx['ns'] = n_half_src
            client_ctx['ng'] = len(client_gx[c])
            if scenario_flag == 'freq_split':
                client_ctx['f'] = config.forward.client_frequencies[c]
            fwi = FWIForward(client_ctx, device, normalize=True,
                             v_denorm_func=v_denormalize, s_norm_func=s_normalize_none)
            per_client_fwi_forwards.append(fwi)
    server_diffusion_model = None
    diffusion_state_dict = None
    diffusion_args = None

    if config.experiment.regularization == "diffusion":
            diffusion_args = {
            'dim': config.diffusion.dim, 'dim_mults': config.diffusion.dim_mults,
            'flash_attn': config.diffusion.flash_attn, 'channels': config.diffusion.channels,
            'image_size': config.diffusion.image_size, 'timesteps': config.diffusion.timesteps,
            'sampling_timesteps': config.diffusion.sampling_timesteps,
            'objective': config.diffusion.objective
            }

            unet_model = Unet(
                dim=diffusion_args.get('dim'),
                dim_mults=diffusion_args.get('dim_mults'),
                flash_attn=diffusion_args.get('flash_attn'),
                channels=diffusion_args.get('channels')
            )
            diffusion = GaussianDiffusion(
                unet_model,
                image_size=diffusion_args.get('image_size'),
                timesteps=diffusion_args.get('timesteps'),
                sampling_timesteps=diffusion_args.get('sampling_timesteps'),
                objective=diffusion_args.get('objective')
            ).to(device)

            # Load the pretrained weights for the diffusion model
            checkpoint = torch.load(config.path.model_path, map_location=device, weights_only=True)
            state_dict = checkpoint.get('model', checkpoint)
            diffusion.load_state_dict(state_dict)
            diffusion.eval()

            # Assign the created model and its state dict to the variables
            server_diffusion_model = diffusion
            diffusion_state_dict = diffusion.state_dict()

    ssim_loss = pytorch_ssim.SSIM(window_size=11)

    fit_config_fn = get_fit_config_fn(config)

    # --- Dispatch to batched or sequential path ---
    if batched:
        all_results = _run_batched(
            config=config, device=device, target_families=target_families,
            main_output_dir=main_output_dir, fwi_forward=fwi_forward,
            per_client_fwi_forwards=per_client_fwi_forwards,
            server_diffusion_model=server_diffusion_model,
            diffusion_state_dict=diffusion_state_dict,
            diffusion_args=diffusion_args, ssim_loss=ssim_loss,
            fit_config_fn=fit_config_fn,
        )
    else:
        all_results = _run_sequential(
            config=config, device=device,
            target_families=target_families, target_instances=target_instances,
            main_output_dir=main_output_dir, fwi_forward=fwi_forward,
            per_client_fwi_forwards=per_client_fwi_forwards,
            server_diffusion_model=server_diffusion_model,
            diffusion_state_dict=diffusion_state_dict,
            diffusion_args=diffusion_args, ssim_loss=ssim_loss,
            fit_config_fn=fit_config_fn,
        )

    # --- FINAL SUMMARY ---
    print(f"\n--- EXPERIMENT COMPLETE ---")
    total_instances = len(all_results['individual_runs'])
    print(f"Completed: {total_instances} instances")
    print(f"Individual results saved in: {main_output_dir}")

    return all_results


# ---------------------------------------------------------------------------
# Sequential path: one FL simulation per (family, instance)
# ---------------------------------------------------------------------------

def _run_sequential(
    *, config, device, target_families, target_instances,
    main_output_dir, fwi_forward, per_client_fwi_forwards,
    server_diffusion_model, diffusion_state_dict, diffusion_args,
    ssim_loss, fit_config_fn,
):
    # Determine families
    if target_families is not None:
        families = list(target_families)
    else:
        vel_path = config.path.velocity_data_path
        inferred = os.path.basename(os.path.dirname(vel_path)) if vel_path else None
        families = [inferred] if inferred else []

    all_results = {'individual_runs': []}
    family_to_vm = {fam: np.load(f"{config.path.velocity_data_path}/{fam}.npy", mmap_mode="r") for fam in families}
    family_to_gt = {fam: np.load(f"{config.path.gt_seismic_data_path}/{fam}.npy", mmap_mode="r") for fam in families}
    family_to_clients = {
        fam: [np.load(f"{config.path.client_seismic_data_path}/client{c+1}/{fam}.npy", mmap_mode="r")
            for c in range(config.experiment.num_clients)]
        for fam in families
    }

    for family in families:
        print(f"\n--- Starting Family: {family} ---")

        if target_instances is not None:
            instance_indices = list(target_instances)
        else:
            instance_indices = [0]
        for i in instance_indices:
            print(f"-- Instance detected: {family}{i}", flush=True)
            result_filename = os.path.join(main_output_dir, f"{family}_{i}.npz")
            print(f"   Run: {family}{i} -> {result_filename}", flush=True)

            vm_np = family_to_vm[family][i:i+1, :]
            gt_np = family_to_gt[family][i:i+1, :]
            client_nps = [arr[i:i+1, :] for arr in family_to_clients[family]]

            vm_data = torch.from_numpy(np.ascontiguousarray(vm_np)).float()
            gt_seismic_data = torch.from_numpy(np.ascontiguousarray(gt_np)).float().pin_memory().to(device, non_blocking=True)
            client_data_list = [torch.from_numpy(np.ascontiguousarray(x)).float().pin_memory().to(device, non_blocking=True)
                                for x in client_nps]

            initial_model = data_trans.prepare_initial_model(vm_data, initial_type='smoothed', sigma=config.forward.initial_sigma)
            initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)

            final_parameters_store = {}

            evaluate_fn = get_evaluate_fn(
                seismic_data=gt_seismic_data, mu_true=vm_data,
                fwi_forward=fwi_forward, ssim_loss=ssim_loss, device=device,
                total_rounds=config.federated.num_rounds,
                final_params_store=final_parameters_store, config=config,
                diffusion_model=server_diffusion_model
            )

            strategy_class = {"FedAvg": FedAvg, "FedAvgM": FedAvgM, "FedProx": FedProx}[config.experiment.strategy]

            strategy_params = {
                "fraction_fit": 1.0,
                "min_fit_clients": config.experiment.num_clients,
                "min_available_clients": config.experiment.num_clients,
                "evaluate_fn": evaluate_fn,
                "fraction_evaluate": 0.0,
                "on_fit_config_fn": fit_config_fn,
                "initial_parameters": ndarrays_to_parameters(tensor_to_ndarrays(initial_model.to(device))),
            }

            if config.experiment.strategy == "FedAvgM":
                strategy_params["server_momentum"] = config.experiment.server_momentum
            elif config.experiment.strategy == "FedProx":
                pass

            strategy_params["fit_metrics_aggregation_fn"] = fit_metrics_fn

            strategy = strategy_class(**strategy_params)

            client_fn_instance = client_fn_factory(
                partitions=client_data_list, device=device,
                fwi_forward=fwi_forward,
                diffusion_args=diffusion_args,
                diffusion_state_dict=diffusion_state_dict,
                per_client_fwi_forwards=per_client_fwi_forwards,
            )
            ray_temp_dir = "/tmp/rfl"
            os.makedirs(ray_temp_dir, exist_ok=True)

            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1)

            history = fl.simulation.start_simulation(
                client_fn=client_fn_instance,
                num_clients=config.experiment.num_clients,
                config=fl.server.ServerConfig(num_rounds=config.federated.num_rounds),
                strategy=strategy,
                client_resources={"num_gpus": config.resources.num_gpus_per_client,
                 "num_cpus": config.resources.num_cpus_per_client} if device.type == "cuda" else {},
                ray_init_args={"include_dashboard": False, "_temp_dir": ray_temp_dir},
            )

            saved_ndarrays = final_parameters_store["final_model"]
            final_model = ndarrays_to_tensor(saved_ndarrays, device)
            final_model_np = final_model.cpu().detach().numpy()
            if final_model_np.ndim == 4 and final_model_np.shape[2] >= 3 and final_model_np.shape[3] >= 3:
                mu_interior = final_model_np[:, :, 1:-1, 1:-1]
            else:
                mu_interior = final_model_np

            metrics_centralized = getattr(history, 'metrics_centralized', {})
            losses_centralized = getattr(history, 'losses_centralized', [])
            save_dict = {
                'mu': np.squeeze(mu_interior),
                'rounds': np.array([r for r, _ in losses_centralized], dtype=np.int32),
                'total_losses': np.array([v for _, v in losses_centralized], dtype=np.float32),
            }
            for key in ['seismic_loss', 'reg_loss', 'mae', 'rmse', 'ssim']:
                if key in metrics_centralized:
                    save_dict[key] = np.array([v for _, v in metrics_centralized[key]], dtype=np.float32)

            run_result = {
                "family": family,
                "instance": i,
                "final_model": final_model_np,
                "history": history,
                "config": config,
            }
            all_results['individual_runs'].append(run_result)

            np.savez(result_filename, **save_dict)
            print(f"Result saved to: {result_filename}")

        print(f"--- Finished Family: {family} ---")

    return all_results


# ---------------------------------------------------------------------------
# Batched path: one FL simulation for ALL families
# ---------------------------------------------------------------------------

def _run_batched(
    *, config, device, target_families,
    main_output_dir, fwi_forward, per_client_fwi_forwards,
    server_diffusion_model, diffusion_state_dict, diffusion_args,
    ssim_loss, fit_config_fn,
):
    families = list(target_families)
    all_results = {'individual_runs': []}

    # Build batch_labels: one entry per sample in the combined batch
    # Combined .npy order matches create_combined_dataset.py FAMILIES list
    batch_labels = [(fam, 0) for fam in families]
    print(f"Batched labels: {batch_labels}")

    # Load the combined .npy files directly (already stacked along axis 0)
    vm_np = np.load(f"{config.path.velocity_data_path}/combined.npy")
    gt_np = np.load(f"{config.path.gt_seismic_data_path}/combined.npy")
    client_nps = [
        np.load(f"{config.path.client_seismic_data_path}/client{c+1}/combined.npy")
        for c in range(config.experiment.num_clients)
    ]
    print(f"Loaded combined data: vm={vm_np.shape}, gt={gt_np.shape}, "
          f"clients=[{', '.join(str(c.shape) for c in client_nps)}]")

    vm_data = torch.from_numpy(np.ascontiguousarray(vm_np)).float()
    gt_seismic_data = torch.from_numpy(np.ascontiguousarray(gt_np)).float().pin_memory().to(device, non_blocking=True)
    client_data_list = [
        torch.from_numpy(np.ascontiguousarray(x)).float().pin_memory().to(device, non_blocking=True)
        for x in client_nps
    ]

    initial_model = data_trans.prepare_initial_model(
        vm_data, initial_type='smoothed', sigma=config.forward.initial_sigma
    )
    initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)
    print(f"Initial model shape: {initial_model.shape}")

    final_parameters_store = {}

    evaluate_fn = get_evaluate_fn(
        seismic_data=gt_seismic_data, mu_true=vm_data,
        fwi_forward=fwi_forward, ssim_loss=ssim_loss, device=device,
        total_rounds=config.federated.num_rounds,
        final_params_store=final_parameters_store, config=config,
        diffusion_model=server_diffusion_model,
        batch_labels=batch_labels,
    )

    strategy_class = {"FedAvg": FedAvg, "FedAvgM": FedAvgM, "FedProx": FedProx}[config.experiment.strategy]

    strategy_params = {
        "fraction_fit": 1.0,
        "min_fit_clients": config.experiment.num_clients,
        "min_available_clients": config.experiment.num_clients,
        "evaluate_fn": evaluate_fn,
        "fraction_evaluate": 0.0,
        "on_fit_config_fn": fit_config_fn,
        "initial_parameters": ndarrays_to_parameters(tensor_to_ndarrays(initial_model.to(device))),
    }

    if config.experiment.strategy == "FedAvgM":
        strategy_params["server_momentum"] = config.experiment.server_momentum

    strategy_params["fit_metrics_aggregation_fn"] = fit_metrics_fn

    strategy = strategy_class(**strategy_params)

    client_fn_instance = client_fn_factory(
        partitions=client_data_list, device=device,
        fwi_forward=fwi_forward,
        diffusion_args=diffusion_args,
        diffusion_state_dict=diffusion_state_dict,
        per_client_fwi_forwards=per_client_fwi_forwards,
    )

    ray_temp_dir = "/tmp/rfl"
    os.makedirs(ray_temp_dir, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # --- RUN ONE SIMULATION ---
    print(f"\n--- Starting batched FL simulation ({len(families)} families) ---")
    history = fl.simulation.start_simulation(
        client_fn=client_fn_instance,
        num_clients=config.experiment.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.federated.num_rounds),
        strategy=strategy,
        client_resources={"num_gpus": config.resources.num_gpus_per_client,
         "num_cpus": config.resources.num_cpus_per_client} if device.type == "cuda" else {},
        ray_init_args={"include_dashboard": False, "_temp_dir": ray_temp_dir},
    )

    # --- SLICE AND SAVE PER-FAMILY RESULTS ---
    saved_ndarrays = final_parameters_store["final_model"]
    final_model = ndarrays_to_tensor(saved_ndarrays, device)
    final_model_np = final_model.cpu().detach().numpy()
    if final_model_np.ndim == 4 and final_model_np.shape[2] >= 3 and final_model_np.shape[3] >= 3:
        mu_interior = final_model_np[:, :, 1:-1, 1:-1]
    else:
        mu_interior = final_model_np

    metrics_centralized = getattr(history, 'metrics_centralized', {})
    losses_centralized = getattr(history, 'losses_centralized', [])

    for idx, (fam, inst) in enumerate(batch_labels):
        result_filename = os.path.join(main_output_dir, f"{fam}_{inst}.npz")
        print(f"   Saving: {fam}_{inst} -> {result_filename}", flush=True)

        # Per-sample model slice
        mu_sample = mu_interior[idx:idx+1]

        save_dict = {
            'mu': np.squeeze(mu_sample),
            'rounds': np.array([r for r, _ in losses_centralized], dtype=np.int32),
            'total_losses': np.array([v for _, v in losses_centralized], dtype=np.float32),
        }

        # Save batch-averaged metrics
        for key in ['seismic_loss', 'reg_loss', 'mae', 'rmse', 'ssim']:
            if key in metrics_centralized:
                save_dict[key] = np.array([v for _, v in metrics_centralized[key]], dtype=np.float32)

        # Save per-model metrics if available
        for key_base in ['mae', 'rmse', 'ssim']:
            per_model_key = f'{key_base}_{fam}_{inst}'
            if per_model_key in metrics_centralized:
                save_dict[key_base] = np.array(
                    [v for _, v in metrics_centralized[per_model_key]], dtype=np.float32
                )

        np.savez(result_filename, **save_dict)
        print(f"   Result saved to: {result_filename}")

        run_result = {
            "family": fam,
            "instance": inst,
            "final_model": final_model_np[idx:idx+1],
            "history": history,
            "config": config,
        }
        all_results['individual_runs'].append(run_result)

    return all_results
