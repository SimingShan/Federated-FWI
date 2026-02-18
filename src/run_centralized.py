import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.diffusion_models.loader import load_diffusion_model
from src.federated_learning.centralized_loss import scenario_aware_seismic_loss
import src.utils.data_trans as data_trans
import src.utils.pytorch_ssim as pytorch_ssim
from src.pde_solvers.client_pde_solver import FWIForward
from src.pde_solvers.scenario_setup import build_fwi_ctx
from src.core.inversion import InversionEngine
from src.regularization.base import RegularizationMethod
from src.core.losses import LossCalculator


def run_centralized(config_path: str = None, family: str = None, batch_size: str = None, config_override=None):
    assert config_path is not None or config_override is not None, "config_path or config_override is required"
    assert family is not None, "family is required"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_override if config_override is not None else OmegaConf.load(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _rl = config.experiment.reg_lambda
    _np = getattr(config.experiment, 'num_patches', 'none')
    base_dir_name = f"centralized_{config.experiment.regularization}_rl{_rl}_np{_np}_{config.experiment.scenario_flag}"

    main_output_dir = os.path.join(config.path.output_path, f"{base_dir_name}_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Output directory: {main_output_dir}")

    # --- Diffusion model (optional) ---
    diffusion, _ = load_diffusion_model(config, device)
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    num_patches = getattr(config.experiment, 'num_patches', None)
    engine = InversionEngine(diffusion, ssim_loss, config.experiment.regularization, num_patches=num_patches)

    # --- Forward solver (once, shared across all batches) ---
    ctx = build_fwi_ctx(config)
    if hasattr(config.forward, 'sx') and config.forward.sx is not None:
        ctx['sx'] = list(config.forward.sx)
    fwi_forward = FWIForward(ctx, device, normalize=True,
                             v_denorm_func=data_trans.v_denormalize,
                             s_norm_func=data_trans.s_normalize_none)
    print(f"Scenario: {config.experiment.scenario_flag}, ns={ctx['ns']}, sx={ctx.get('sx')}")

    # --- Metrics helpers (once, shared across all batches) ---
    regularization_method = RegularizationMethod(config.experiment.regularization, diffusion, num_patches=num_patches)
    loss_calc = LossCalculator(regularization_method)
    l1_fn = nn.L1Loss()
    l2_fn = nn.MSELoss()

    # --- Data ---
    vm_all = np.load(f"{config.path.velocity_data_path}/{family}.npy", mmap_mode="r")
    gt_all = np.load(f"{config.path.gt_seismic_data_path}/{family}.npy", mmap_mode="r")
    num_instances = vm_all.shape[0]

    # Resolve batch_size
    if isinstance(batch_size, str):
        resolved_bs = num_instances if batch_size.lower() == 'max' else int(batch_size)
    elif isinstance(batch_size, int):
        resolved_bs = batch_size
    else:
        resolved_bs = 2

    print(f"\n--- Starting Family: {family} ---")
    for start in range(0, num_instances, resolved_bs):
        idxs = list(range(start, min(start + resolved_bs, num_instances)))
        print(f"-- Running Instances: {family}{idxs} --")

        test_data = torch.from_numpy(gt_all[idxs, ...]).float().to(device)
        test_vm = torch.from_numpy(vm_all[idxs, ...]).float().to(device)
        test_init_vm = data_trans.prepare_initial_model(
            test_vm, initial_type="smoothed", sigma=config.forward.initial_sigma
        )
        test_init_vm = torch.nn.functional.pad(test_init_vm, (1, 1, 1, 1), mode="constant", value=0)

        mu, final_results = engine.optimize(
            mu=test_init_vm, mu_true=test_vm, y=test_data,
            fwi_forward=fwi_forward, ts=4500, lr=0.03,
            reg_lambda=config.experiment.reg_lambda,
        )

        # Per-sample final metrics
        predicted_seismic_full = fwi_forward(mu)
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
            per_sample_metrics.append({
                'obs_loss': float(obs_j.item()),
                'reg_loss': float(reg_j.mean().item()),
                'total_loss': float(total_j.item()),
                'mae': float(mae_j.item()),
                'rmse': float(np.sqrt(mse_j.item())),
                'ssim': float(ssim_loss((vm_sample_unnorm + 1) / 2, (vm_data_unnorm + 1) / 2).item()),
            })

        mu_np = mu.detach().cpu().numpy()
        for j, idx in enumerate(idxs):
            result_path = os.path.join(main_output_dir, f"{family}_{idx}.npz")
            history_j = final_results[j]
            metrics_j = per_sample_metrics[j]
            np.savez(result_path,
                mu=mu_np[j],
                total_losses=np.array(history_j['total_losses'], dtype=np.float32),
                obs_losses=np.array(history_j['obs_losses'], dtype=np.float32),
                reg_losses=np.array(history_j['reg_losses'], dtype=np.float32),
                ssim_history=np.array(history_j['ssim'], dtype=np.float32),
                mae_history=np.array(history_j['mae'], dtype=np.float32),
                rmse_history=np.array(history_j['rmse'], dtype=np.float32),
                final_obs_loss=np.float64(metrics_j['obs_loss']),
                final_reg_loss=np.float64(metrics_j['reg_loss']),
                final_total_loss=np.float64(metrics_j['total_loss']),
                final_mae=np.float64(metrics_j['mae']),
                final_rmse=np.float64(metrics_j['rmse']),
                final_ssim=np.float64(metrics_j['ssim']),
            )
        print(f"Saved batch results for indices: {idxs}")

    print(f"--- Finished Family: {family} ---")
    print(f"\n--- EXPERIMENT COMPLETE: {num_instances} instances saved in {main_output_dir} ---")
