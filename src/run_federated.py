from datetime import datetime
import os
from typing import List, Optional
from omegaconf import OmegaConf
import numpy as np
import torch
import src.utils.pytorch_ssim as pytorch_ssim
import torch.multiprocessing as mp
import flwr as fl
from flwr.common import ndarrays_to_parameters
import torch.nn.functional as F
from flwr.server.strategy import FedAvg, FedAvgM, FedProx
import src.utils.data_trans as data_trans
from src.diffusion_models.loader import load_diffusion_model
from src.federated_learning.flwr_client import client_fn_factory
from src.federated_learning.flwr_evaluation import get_evaluate_fn
from src.federated_learning.flwr_utils import (
    tensor_to_ndarrays, ndarrays_to_tensor, fit_metrics_fn, get_fit_config_fn
)
from src.pde_solvers.client_pde_solver import FWIForward
from src.pde_solvers.scenario_setup import build_fwi_ctx, build_per_client_fwi_forwards

def run_full_experiment(
    config_path: str = None,
    target_families: Optional[List[str]] = None,
    config_override=None,
):
    """Run a federated FL experiment over one or more families in a single simulation.
    All families are stacked into one batch; a single-family run is batch_size=1.
    """
    assert config_path is not None or config_override is not None, "config_path or config_override is required"

    config = config_override if config_override is not None else OmegaConf.load(config_path)
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    families = list(target_families)

    print("--- STARTING FULL EXPERIMENT RUN ---")
    print(f"Strategy: {config.experiment.strategy}, Regularization: {config.experiment.regularization}, Scenario: {config.experiment.scenario_flag}")
    print(f"Families: {families}")
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

    main_output_dir = os.path.join(config.path.output_path, f"{base_dir_name}_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Output directory: {main_output_dir}")

    # --- Shared forward solver ---
    ctx = build_fwi_ctx(config)
    fwi_forward = FWIForward(ctx, device, normalize=True, v_denorm_func=data_trans.v_denormalize, s_norm_func=data_trans.s_normalize_none)
    per_client_fwi_forwards = build_per_client_fwi_forwards(config, ctx, device)

    # --- Diffusion model (optional) ---
    server_diffusion_model, diffusion_args = load_diffusion_model(config, device)
    diffusion_state_dict = server_diffusion_model.state_dict() if server_diffusion_model is not None else None

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    fit_config_fn = get_fit_config_fn(config)

    # --- Data loading ---
    # Use 'combined.npy' for multi-family runs; fall back to '{family}.npy' for single-family
    data_suffix = "combined" if len(families) > 1 else families[0]
    vm_np = np.load(f"{config.path.velocity_data_path}/{data_suffix}.npy")
    gt_np = np.load(f"{config.path.gt_seismic_data_path}/{data_suffix}.npy")
    client_nps = [
        np.load(f"{config.path.client_seismic_data_path}/client{c+1}/{data_suffix}.npy")
        for c in range(config.experiment.num_clients)
    ]
    print(f"Loaded data ({data_suffix}): vm={vm_np.shape}, gt={gt_np.shape}, "
          f"clients=[{', '.join(str(c.shape) for c in client_nps)}]")

    vm_data = torch.from_numpy(vm_np).float()
    gt_seismic_data = torch.from_numpy(gt_np).float().to(device)
    client_data_list = [torch.from_numpy(x).float().to(device) for x in client_nps]

    initial_model = data_trans.prepare_initial_model(vm_data, initial_type='smoothed', sigma=config.forward.initial_sigma)
    initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)
    print(f"Initial model shape: {initial_model.shape}")

    # --- FL setup ---
    final_parameters_store = {}

    evaluate_fn = get_evaluate_fn(
        seismic_data=gt_seismic_data, mu_true=vm_data,
        fwi_forward=fwi_forward, ssim_loss=ssim_loss, device=device,
        total_rounds=config.federated.num_rounds,
        final_params_store=final_parameters_store, config=config,
        diffusion_model=server_diffusion_model,
        batch_labels=families,
    )

    strategy_class = {"FedAvg": FedAvg, "FedAvgM": FedAvgM, "FedProx": FedProx}[config.experiment.strategy]
    strategy_params = {
        "fraction_fit": 1.0,
        "min_fit_clients": config.experiment.num_clients,
        "min_available_clients": config.experiment.num_clients,
        "evaluate_fn": evaluate_fn,
        "fraction_evaluate": 0.0,
        "on_fit_config_fn": fit_config_fn,
        "fit_metrics_aggregation_fn": fit_metrics_fn,
        "initial_parameters": ndarrays_to_parameters(tensor_to_ndarrays(initial_model.to(device))),
    }
    if config.experiment.strategy == "FedAvgM":
        strategy_params["server_momentum"] = config.experiment.server_momentum
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

    # --- Run ---
    print(f"\n--- Starting FL simulation ({len(families)} families) ---")
    history = fl.simulation.start_simulation(
        client_fn=client_fn_instance,
        num_clients=config.experiment.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.federated.num_rounds),
        strategy=strategy,
        client_resources={"num_gpus": config.resources.num_gpus_per_client,
                          "num_cpus": config.resources.num_cpus_per_client} if device.type == "cuda" else {},
        ray_init_args={"include_dashboard": False, "_temp_dir": ray_temp_dir},
    )

    # --- Save per-family results ---
    final_model = ndarrays_to_tensor(final_parameters_store["final_model"], device)
    final_model_np = final_model.cpu().detach().numpy()
    mu_interior = final_model_np[:, :, 1:-1, 1:-1]

    metrics_centralized = getattr(history, 'metrics_centralized', {})
    losses_centralized = getattr(history, 'losses_centralized', [])

    for idx, fam in enumerate(families):
        result_filename = os.path.join(main_output_dir, f"{fam}_0.npz")
        save_dict = {
            'mu': np.squeeze(mu_interior[idx:idx+1]),
            'rounds': np.array([r for r, _ in losses_centralized], dtype=np.int32),
            'total_losses': np.array([v for _, v in losses_centralized], dtype=np.float32),
        }
        for key in ['seismic_loss', 'reg_loss', 'mae', 'rmse', 'ssim']:
            if key in metrics_centralized:
                save_dict[key] = np.array([v for _, v in metrics_centralized[key]], dtype=np.float32)
        for key_base in ['mae', 'rmse', 'ssim']:
            per_model_key = f'{key_base}_{fam}'
            if per_model_key in metrics_centralized:
                save_dict[key_base] = np.array(
                    [v for _, v in metrics_centralized[per_model_key]], dtype=np.float32
                )
        np.savez(result_filename, **save_dict)
        print(f"   Result saved to: {result_filename}")

    print(f"\n--- EXPERIMENT COMPLETE: {len(families)} families saved in {main_output_dir} ---")
