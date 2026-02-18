import torch
from flwr.common import NDArrays, Metrics
from typing import List, Tuple
import numpy as np

def ndarrays_to_tensor(ndarrays: NDArrays, device: torch.device) -> torch.Tensor:
    """Convert Flower NDArrays (List[np.ndarray]) to a PyTorch Tensor."""
    if not ndarrays:
         raise ValueError("Received empty NDArrays list")
    tensor = torch.tensor(ndarrays[0], dtype=torch.float32).to(device)
    return tensor

def tensor_to_ndarrays(tensor: torch.Tensor) -> NDArrays:
    """Convert a PyTorch Tensor to Flower NDArrays (List[np.ndarray])."""
    ndarrays = [tensor.cpu().detach().numpy()]
    return ndarrays

def get_fit_config_fn(config):
    """Factory function to create the on_fit_config_fn."""
    def fit_config_fn(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": config.federated.local_epochs,
            "local_lr": config.federated.local_lr,
            "total_rounds": config.federated.num_rounds,
            "regularization": config.experiment.regularization,
            "reg_lambda": config.experiment.reg_lambda,
            "num_patches": getattr(config.experiment, 'num_patches', 0),
            "seed": 42,
        }
    return fit_config_fn


def fit_metrics_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics (mean and std across clients)."""
    aggregated = {}
    for _, client_metrics in metrics:
        for key, value in client_metrics.items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(value)

    final_metrics = {}
    for key, values in aggregated.items():
        final_metrics[f"{key}_mean"] = sum(values) / len(values)
        final_metrics[f"{key}_std"] = float(np.std(values))

    return final_metrics
