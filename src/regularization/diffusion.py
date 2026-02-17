import math
import torch
from typing import List, Tuple
from src.utils.diffusion_utils import diffusion_pad, diffusion_crop

def calculate_patches(width: int, height: int, num_patches: int = None) -> Tuple[List[Tuple[int, int]], List[int]]:
    m = height
    n = width
    k_min = math.ceil(n / m)
    if num_patches is not None:
        if num_patches < k_min:
            raise ValueError(f"num_patches={num_patches} is too small to cover width={n} with patch size={m}. Minimum is {k_min}.")
        k = num_patches
    else:
        k = k_min
    if k == 1:
        return [(0, n)], []
    s = (n - m) / (k - 1)
    positions = []
    for i in range(k):
        if i == k - 1:
            positions.append((n - m, n))
        else:
            start = int(i * s)
            positions.append((start, min(start + m, n)))

    overlaps = [positions[i][1] - positions[i + 1][0] for i in range(k - 1)]
    return positions, overlaps


class RED_DiffEq:

    def __init__(self, diffusion_model, num_patches: int = None):
        self.diffusion_model = diffusion_model
        image_size = getattr(diffusion_model, 'image_size', 72)
        self.input_size = image_size[0] if isinstance(image_size, (tuple, list)) else image_size
        self.num_patches = num_patches

    def get_reg_loss(self, mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = mu.shape[0]
        max_timestep = self.diffusion_model.num_timesteps

        time_tensor = torch.randint(
            0, max_timestep,
            (batch_size,), device=mu.device, dtype=torch.long
        )

        noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype)
        x0_pred = mu
        x_t = self.diffusion_model.q_sample(x0_pred, t=time_tensor, noise=noise)

        predictions = self.diffusion_model.model_predictions(
            x_t, t=time_tensor, x_self_cond=None,
            clip_x_start=True, rederive_pred_noise=True
        )

        pred_noise = predictions.pred_noise
        gradient_field = (pred_noise - noise).detach()
        reg_field = gradient_field * x0_pred

        gradient_per_model = gradient_field.view(batch_size, -1).mean(dim=1)
        reg_per_model = reg_field.view(batch_size, -1).mean(dim=1)

        return reg_per_model, gradient_per_model

    def get_reg_loss_patched(self, mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mu_unpadded = diffusion_crop(mu)
        batch_size = mu_unpadded.shape[0]
        height = mu_unpadded.shape[2]
        width = mu_unpadded.shape[3]

        patch_positions, overlaps = calculate_patches(width, height, self.num_patches)

        max_timestep = self.diffusion_model.num_timesteps

        time_tensor = torch.randint(
            0, max_timestep,
            (batch_size,), device=mu_unpadded.device, dtype=torch.long
        )

        noise = torch.randn(mu_unpadded.shape, device=mu_unpadded.device, dtype=mu_unpadded.dtype)

        x0_pred = mu_unpadded

        gradient_field = torch.zeros_like(mu_unpadded)
        weight_map = torch.zeros_like(mu_unpadded)

        for patch_idx, (start_x, end_x) in enumerate(patch_positions):
            x0_pred_patch = x0_pred[:, :, :, start_x:end_x]
            noise_patch = noise[:, :, :, start_x:end_x]

            x0_pred_patch_padded = diffusion_pad(x0_pred_patch)
            noise_patch_padded = diffusion_pad(noise_patch)

            x_t = self.diffusion_model.q_sample(
                x0_pred_patch_padded, t=time_tensor, noise=noise_patch_padded
            )

            predictions = self.diffusion_model.model_predictions(
                x_t, t=time_tensor, x_self_cond=None,
                clip_x_start=True, rederive_pred_noise=True
            )

            pred_noise_patch = diffusion_crop(predictions.pred_noise)
            noise_patch_cropped = diffusion_crop(noise_patch_padded)

            gradient_patch = (pred_noise_patch - noise_patch_cropped).detach()

            patch_width = end_x - start_x
            weight = torch.ones(patch_width, device=mu_unpadded.device)

            if patch_idx > 0:
                weight[:overlaps[patch_idx - 1]] = 0.5

            if patch_idx < len(patch_positions) - 1:
                weight[-overlaps[patch_idx]:] = 0.5

            weight = weight.view(1, 1, 1, -1)

            gradient_field[:, :, :, start_x:end_x] += gradient_patch * weight
            weight_map[:, :, :, start_x:end_x] += weight

        gradient_field = gradient_field / weight_map.clamp(min=1e-8)

        reg_field = gradient_field * mu_unpadded

        gradient_per_model = gradient_field.view(batch_size, -1).mean(dim=1)
        reg_per_model = reg_field.view(batch_size, -1).mean(dim=1)

        return reg_per_model, gradient_per_model