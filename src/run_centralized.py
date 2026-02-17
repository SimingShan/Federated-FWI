import os
from omegaconf import OmegaConf
import numpy as np
from src.federated_learning.centralized_loss import scenario_aware_seismic_loss
import src.utils.data_trans as data_trans
import src.utils.pytorch_ssim as pytorch_ssim
from src.diffusion_models.diffusion_model import Unet, GaussianDiffusion
from src.pde_solvers.client_pde_solver import FWIForward
from src.core.inversion import InversionEngine
from src.regularization.base import RegularizationMethod
from src.core.losses import LossCalculator
from datetime import datetime
import glob
import torch
import torch.nn as nn
def run_centralized(config_path: str = None, family: str = None, batch_size: str = None, config_override=None):
    assert config_path is not None or config_override is not None, "config_path or config_override is required"
    assert family is not None, "family is required"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if config_override is not None:
        config = config_override
    else:
        config = OmegaConf.load(config_path)
    gt_seismic_data_path = config["path"]["gt_seismic_data_path"]
    velocity_data_path = config["path"]["velocity_data_path"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _rl = config.experiment.reg_lambda
    _lr = getattr(config.federated, 'local_lr', 'none') if hasattr(config, 'federated') else 'none'
    _np = getattr(config.experiment, 'num_patches', 'none')
    _sm = getattr(config.experiment, 'server_momentum', 'none')
    base_dir_name = f"centralized_{config.experiment.regularization}_rl{_rl}_lr{_lr}_np{_np}_sm{_sm}_{config.experiment.scenario_flag}"
    # Check for existing directories
    existing_dirs = glob.glob(os.path.join(config.path.output_path, f"{base_dir_name}_*"))
    if existing_dirs:
        # Use the most recent existing directory
        main_output_dir = max(existing_dirs, key=os.path.getctime)
        print(f"Using existing directory: {main_output_dir}")
    else:
        # Create new directory
        main_output_dir = os.path.join(config.path.output_path, f"{base_dir_name}_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        print(f"Created new directory: {main_output_dir}")

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

    model_path = config.path.model_path
    diffusion.load_state_dict(torch.load(model_path, map_location=device)['model'])
    diffusion.eval()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    num_patches = getattr(config.experiment, 'num_patches', None)
    engine = InversionEngine(diffusion, ssim_loss, config.experiment.regularization, num_patches=num_patches)

    families = [family]

    family_to_vm = {fam: np.load(f"{velocity_data_path}/{fam}.npy", mmap_mode="r") for fam in families}
    num_instances = family_to_vm[family].shape[0]
    instance_indices = list(range(num_instances))
    family_to_gt = {fam: np.load(f"{gt_seismic_data_path}/{fam}.npy", mmap_mode="r") for fam in families}

    for family in families:
        print(f"\n--- Starting Family: {family} ---")

        # Batch instances for speed (adjust batch_size per memory)
        # Resolve batch_size: 'max' -> entire partition; int string -> int; default to 2 if None
        if isinstance(batch_size, str):
            if batch_size.lower() == 'max':
                resolved_bs = len(instance_indices)
            else:
                try:
                    resolved_bs = int(batch_size)
                except Exception:
                    resolved_bs = 2
        elif isinstance(batch_size, int):
            resolved_bs = batch_size
        else:
            resolved_bs = 2
        for start in range(0, len(instance_indices), resolved_bs):
            idxs = instance_indices[start:start+resolved_bs]
            print(f"-- Running Instances: {family}{idxs} --")
            test_data = torch.from_numpy(family_to_gt[family][idxs, ...]).float().to(device)
            test_vm = torch.from_numpy(family_to_vm[family][idxs, ...]).float().to(device)
            test_init_vm = data_trans.prepare_initial_model(
                test_vm, initial_type="smoothed", sigma=config.forward.initial_sigma
            )
            test_init_vm = torch.nn.functional.pad(test_init_vm, (1, 1, 1, 1), mode="constant", value=0)
            ctx = {'n_grid': config.forward.n_grid, 'nt': config.forward.nt, 'dx': config.forward.dx,
                    'nbc': config.forward.nbc, 'dt': config.forward.dt, 'f': config.forward.f,
                    'sz': config.forward.sz, 'gz': config.forward.gz, 'ng': config.forward.ng,
                    'ns': config.forward.ns}
            # Honor explicit source locations if provided
            try:
                if hasattr(config.forward, 'sx') and config.forward.sx is not None:
                    ctx['sx'] = list(config.forward.sx)
            except Exception:
                pass

            fwi_forward = FWIForward(
                                    ctx, device, normalize=True,
                                    v_denorm_func=data_trans.v_denormalize,
                                    s_norm_func=data_trans.s_normalize_none
                                    )

            # Debug geometry summary for visibility
            try:
                print(f"Scenario: {config.experiment.scenario_flag}, ns (cfg): {config.forward.ns}, explicit sx: {getattr(config.forward, 'sx', None)}")
            except Exception:
                pass

            mu, final_results = engine.optimize(
                                    mu=test_init_vm,
                                    mu_true=test_vm,
                                    y=test_data,
                                    fwi_forward=fwi_forward,
                                    ts=4500,
                                    lr=0.03,
                                    reg_lambda=config.experiment.reg_lambda,
                                    )

            # Save each item in the batch
            try:
                # Compute per-sample final metrics to avoid batch-averaged summaries
                # engine.optimize() already returns interior (no padding)
                predicted_seismic_full = fwi_forward(mu)
                regularization_method = RegularizationMethod(config.experiment.regularization, diffusion, num_patches=num_patches)
                loss_calc = LossCalculator(regularization_method)
                l1_fn = nn.L1Loss()
                l2_fn = nn.MSELoss()
                per_sample_metrics = []
                for j in range(len(idxs)):
                    y_j = test_data[j:j+1]
                    pred_j = predicted_seismic_full[j:j+1]
                    obs_j = scenario_aware_seismic_loss(y_j, pred_j, config.experiment.scenario_flag)
                    reg_j = loss_calc.regularization_loss(mu[j:j+1])
                    total_j = obs_j + config.experiment.reg_lambda * reg_j.mean()
                    vm_sample_unnorm = mu[j:j+1].detach().to('cpu')
                    vm_data_unnorm = data_trans.v_normalize(test_vm[j:j+1]).detach().to('cpu')
                    mae_j = l1_fn(vm_sample_unnorm, vm_data_unnorm)
                    mse_j = l2_fn(vm_sample_unnorm, vm_data_unnorm)
                    rmse_j = float(np.sqrt(mse_j.item()))
                    ssim_j = ssim_loss((vm_sample_unnorm + 1) / 2, (vm_data_unnorm + 1) / 2)
                    per_sample_metrics.append({
                        'obs_loss': float(obs_j.item()),
                        'reg_loss': float(reg_j.mean().item()),
                        'total_loss': float(total_j.item()),
                        'mae': float(mae_j.item()),
                        'rmse': rmse_j,
                        'ssim': float(ssim_j.item()),
                    })
                # engine.optimize() already returns interior — no further stripping needed
                mu_np = mu.detach().cpu().numpy()
                for j, idx in enumerate(idxs):
                    result_path = os.path.join(main_output_dir, f"{family}_{idx}.npz")
                    history_j = final_results[j]
                    metrics_j = per_sample_metrics[j]
                    np.savez(result_path,
                        # Final velocity model (interior, no padding)
                        mu=mu_np[j],
                        # Per-step optimization history
                        total_losses=np.array(history_j['total_losses'], dtype=np.float32),
                        obs_losses=np.array(history_j['obs_losses'], dtype=np.float32),
                        reg_losses=np.array(history_j['reg_losses'], dtype=np.float32),
                        ssim_history=np.array(history_j['ssim'], dtype=np.float32),
                        mae_history=np.array(history_j['mae'], dtype=np.float32),
                        rmse_history=np.array(history_j['rmse'], dtype=np.float32),
                        # Final evaluation metrics (scalars)
                        final_obs_loss=np.float64(metrics_j['obs_loss']),
                        final_reg_loss=np.float64(metrics_j['reg_loss']),
                        final_total_loss=np.float64(metrics_j['total_loss']),
                        final_mae=np.float64(metrics_j['mae']),
                        final_rmse=np.float64(metrics_j['rmse']),
                        final_ssim=np.float64(metrics_j['ssim']),
                    )
                print(f"Saved batch results for indices: {idxs}")
            except Exception as e:
                print(f"Warning: failed to save batch results for {family} {idxs}: {e}")
        print(f"--- Finished Family: {family} ---")
    print(f"\n--- EXPERIMENT COMPLETE ---")
    total_instances = len(families) * len(instance_indices)
    print(f"Completed: {len(families)} families, {total_instances} instances")
    print(f"Individual results saved in: {main_output_dir}")
