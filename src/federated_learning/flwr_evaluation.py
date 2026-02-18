import torch
from flwr.common import NDArrays
from src.regularization.base import RegularizationMethod
from src.core.losses import LossCalculator
from src.core.metrics import MetricsCalculator
from src.federated_learning.flwr_utils import ndarrays_to_tensor

def get_evaluate_fn(
    seismic_data: torch.Tensor,
    mu_true: torch.Tensor,
    fwi_forward,
    ssim_loss,
    device: torch.device,
    total_rounds: int,
    final_params_store: dict,
    diffusion_model=None,
    config=None,
    batch_labels=None,
):

    assert config is not None, "Config must be provided for evaluation."
    reg_lambda = config.experiment.reg_lambda
    regularization_method = RegularizationMethod(
        config.experiment.regularization, diffusion_model,
        num_patches=getattr(config.experiment, 'num_patches', None)
    )
    loss_calc = LossCalculator(regularization_method)
    metrics_calc = MetricsCalculator(ssim_loss)

    def evaluate(server_round: int, parameters: NDArrays, _):
        if server_round == total_rounds:
            final_params_store["final_model"] = parameters

        model = ndarrays_to_tensor(parameters, device)

        with torch.no_grad():
            model_input = model[:, :, 1:-1, 1:-1]
            predicted_seismic = fwi_forward(model_input)
            seismic_data_dev = seismic_data.to(device)
            loss_obs = loss_calc.observation_loss(predicted_seismic, seismic_data_dev)
            raw_reg_loss = loss_calc.regularization_loss(model)
            total_loss = loss_calc.total_loss(loss_obs, raw_reg_loss, reg_lambda)
            mae, rmse, ssim_val = metrics_calc.calculate(model_input, mu_true.to(device))

        metrics = {
            'seismic_loss': loss_obs.mean().item(),
            'reg_loss': raw_reg_loss.mean().item(),
            'mae': mae.mean().item(), 'rmse': rmse.mean().item(), 'ssim': ssim_val.mean().item()
        }

        # Per-model metrics when running in batched mode
        if batch_labels is not None:
            for idx, fam in enumerate(batch_labels):
                metrics[f'mae_{fam}'] = mae[idx].item()
                metrics[f'rmse_{fam}'] = rmse[idx].item()
                metrics[f'ssim_{fam}'] = ssim_val[idx].item()

        return total_loss.mean().item(), metrics

    return evaluate
