import torch
from typing import Optional
from src.regularization.diffusion import RED_DiffEq
from src.regularization.benchmarks import total_variation_loss, tikhonov_loss


class RegularizationMethod:

    def __init__(self, regularization_type: Optional[str],
                 diffusion_model=None, num_patches: int = None):

        self.regularization_type = regularization_type
        self.diffusion_model = diffusion_model
        if regularization_type == 'diffusion':
            self.red_diffeq = RED_DiffEq(diffusion_model, num_patches=num_patches)

    def get_reg_loss(self, mu: torch.Tensor):
        if self.regularization_type == 'diffusion':
            if self.diffusion_model is None:
                raise ValueError("Diffusion model required for 'diffusion' regularization")

            height = mu.shape[2]
            width = mu.shape[3]

            if width > self.red_diffeq.input_size or height > self.red_diffeq.input_size:
                reg_loss, _ = self.red_diffeq.get_reg_loss_patched(mu)
            else:
                reg_loss, _ = self.red_diffeq.get_reg_loss(mu)

            return reg_loss

        elif self.regularization_type == 'tikhonov':
            reg_loss = tikhonov_loss(mu)
            return reg_loss

        elif self.regularization_type == 'total_variation':
            reg_loss = total_variation_loss(mu)
            return reg_loss

        else:
            reg_loss = torch.zeros(mu.shape[0], device=mu.device, dtype=mu.dtype)
            return reg_loss
