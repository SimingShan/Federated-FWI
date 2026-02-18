from typing import Optional, Tuple
import torch
from src.diffusion_models.diffusion_model import Unet, GaussianDiffusion


def build_diffusion_model(args: dict, device: torch.device) -> GaussianDiffusion:
    """Construct a GaussianDiffusion model from an args dict (no weights loaded)."""
    unet_model = Unet(
        dim=args.get('dim', 64),
        dim_mults=args.get('dim_mults', (1, 2, 4, 8)),
        flash_attn=args.get('flash_attn', False),
        channels=args.get('channels', 1),
    )
    return GaussianDiffusion(
        unet_model,
        image_size=args.get('image_size', 72),
        timesteps=args.get('timesteps', 1000),
        sampling_timesteps=args.get('sampling_timesteps', 250),
        objective=args.get('objective', 'pred_noise'),
    ).to(device)


def load_diffusion_model(
    config, device: torch.device
) -> Tuple[Optional[GaussianDiffusion], Optional[dict]]:
    """Load a pretrained diffusion model from config.

    Returns (model, diffusion_args) if regularization == 'diffusion',
    otherwise (None, None).
    """
    if config.experiment.regularization != "diffusion":
        return None, None

    diffusion_args = {
        'dim': config.diffusion.dim,
        'dim_mults': config.diffusion.dim_mults,
        'flash_attn': config.diffusion.flash_attn,
        'channels': config.diffusion.channels,
        'image_size': config.diffusion.image_size,
        'timesteps': config.diffusion.timesteps,
        'sampling_timesteps': config.diffusion.sampling_timesteps,
        'objective': config.diffusion.objective,
    }

    diffusion = build_diffusion_model(diffusion_args, device)

    checkpoint = torch.load(config.path.model_path, map_location=device, weights_only=True)
    diffusion.load_state_dict(checkpoint.get('model', checkpoint))
    diffusion.eval()

    return diffusion, diffusion_args
