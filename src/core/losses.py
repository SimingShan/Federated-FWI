import torch
import torch.nn as nn
from typing import Optional

from src.regularization.base import RegularizationMethod

class LossCalculator:
    """Calculate observation and regularization losses for FWI optimization."""

    def __init__(self, regularization_method: RegularizationMethod):
        self.regularization_method = regularization_method

    def observation_loss(self, predicted: torch.Tensor, target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        loss = nn.L1Loss(reduction='none')(target.float(), predicted.float())

        if mask is not None:
            loss = loss * mask
            num_observed = mask.sum(dim=tuple(range(1, len(mask.shape)))).clamp(min=1.0)
            loss = loss.sum(dim=tuple(range(1, len(loss.shape)))) / num_observed
        else:
            loss = loss.mean(dim=tuple(range(1, len(loss.shape))))

        return loss

    def regularization_loss(self, mu: torch.Tensor):
        return self.regularization_method.get_reg_loss(mu)

    def total_loss(self, obs_loss: torch.Tensor, reg_loss: torch.Tensor, reg_lambda: float) -> torch.Tensor:
        return obs_loss + reg_lambda * reg_loss
