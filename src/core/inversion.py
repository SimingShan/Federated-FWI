import torch
from tqdm.auto import tqdm

from src.regularization.base import RegularizationMethod
from src.utils.data_trans import add_noise_to_seismic, missing_trace
from src.utils.pytorch_ssim import SSIM
from src.core.metrices import MetricsCalculator
from src.core.losses import LossCalculator


class InversionEngine:

    def __init__(self, diffusion_model, ssim_loss: SSIM, regularization: str, num_patches: int = None):
        self.diffusion_model = diffusion_model
        self.ssim_loss = ssim_loss
        self.device = diffusion_model.device
        self.regularization_method = RegularizationMethod(regularization, diffusion_model, num_patches=num_patches)

    def optimize(self, mu: torch.Tensor, mu_true: torch.Tensor, y: torch.Tensor,
                fwi_forward, ts: int = 300, lr: float = 0.03, reg_lambda: float = 0.01,
                noise_std: float = 0.0, missing_number: int = 0):

        if mu.shape[0] != y.shape[0]:
            raise ValueError('Batch size mismatch between velocity and seismic data')

        if fwi_forward is None or not callable(fwi_forward):
            raise ValueError('fwi_forward must be a callable forward modeling function')

        fwi_forward = fwi_forward.to(self.device)

        batch_size = mu.shape[0]
        mu = mu.float().clone().detach().to(self.device).requires_grad_(True)
        mu_true = mu_true.float().to(self.device)

        optimizer = torch.optim.Adam([mu], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ts, eta_min=0.0
        )

        metrics_calc = MetricsCalculator(self.ssim_loss)
        loss_calc = LossCalculator(self.regularization_method)

        metrics_history = {
            'total_losses': [], 'obs_losses': [], 'reg_losses': [],
            'ssim': [], 'mae': [], 'rmse': []
        }

        y = add_noise_to_seismic(y, noise_std)
        y, mask = missing_trace(y, missing_number, return_mask=True)
        y = y.to(self.device)
        mask = mask.to(self.device)

        pbar = tqdm(range(ts), desc='Optimizing', unit='step')
        for step in pbar:

            if self.regularization_method.regularization_type == 'diffusion':
                noise_x0 = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype)
                x0_pred = mu + 0.0001 * noise_x0
            else:
                x0_pred = mu

            predicted_seismic = fwi_forward(x0_pred[:, :, 1:-1, 1:-1])
            loss_obs = loss_calc.observation_loss(predicted_seismic, y, mask=mask)

            reg_loss = loss_calc.regularization_loss(x0_pred)

            total_loss = loss_calc.total_loss(loss_obs, reg_loss, reg_lambda)

            optimizer.zero_grad(set_to_none=True)
            total_loss.sum().backward()
            optimizer.step()

            with torch.no_grad():
                mu.data.clamp_(-1, 1)

            scheduler.step()

            mae, rmse, ssim = metrics_calc.calculate(mu[:, :, 1:-1, 1:-1], mu_true)

            metrics_history['total_losses'].append(total_loss.detach().cpu().numpy())
            metrics_history['obs_losses'].append(loss_obs.detach().cpu().numpy())
            metrics_history['reg_losses'].append(reg_loss.detach().cpu().numpy())
            metrics_history['ssim'].append(ssim.detach().cpu().numpy())
            metrics_history['mae'].append(mae.detach().cpu().numpy())
            metrics_history['rmse'].append(rmse.detach().cpu().numpy())

            pbar.set_postfix({
                'MAE': mae.mean().item(),
                'RMSE': rmse.mean().item(),
                'SSIM': ssim.mean().item(),
            })

        final_results_per_model = []
        num_timesteps = len(metrics_history['total_losses'])
        for i in range(batch_size):
            model_results = {
                'total_losses': [metrics_history['total_losses'][t][i] for t in range(num_timesteps)],
                'obs_losses': [metrics_history['obs_losses'][t][i] for t in range(num_timesteps)],
                'reg_losses': [metrics_history['reg_losses'][t][i] for t in range(num_timesteps)],
                'ssim': [metrics_history['ssim'][t][i] for t in range(num_timesteps)],
                'mae': [metrics_history['mae'][t][i] for t in range(num_timesteps)],
                'rmse': [metrics_history['rmse'][t][i] for t in range(num_timesteps)]
            }
            final_results_per_model.append(model_results)

        mu_result = mu[:, :, 1:-1, 1:-1]
        return mu_result, final_results_per_model
