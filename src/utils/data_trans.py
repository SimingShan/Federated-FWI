import random
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def v_normalize(v):
    """Normalize velocity values to [-1, 1]"""
    return (((v - 1500) / 3000) * 2) - 1

def v_denormalize(v_norm):
    """Denormalize velocity values from [-1, 1] to original range"""
    return ((v_norm + 1) / 2) * 3000 + 1500

def s_normalize_none(s):
    """Keep the data in original scale"""
    return s
    
def add_noise_to_seismic(y, std, noise_type='gaussian', generator=None):
    assert std >= 0, "The standard deviation of the noise must be greater than 0"
    if std == 0:
        return y
    y = y.detach().cpu().numpy()
    if generator is not None:
        noise = generator.normal(0, std, y.shape).astype(np.float32)
    else:
        noise = np.random.normal(0, std, y.shape).astype(np.float32)
    return torch.tensor(y + noise).float()

def prepare_initial_model(v_true, initial_type=None, sigma=None):
    assert initial_type in ['smoothed', 'homogeneous', 'linear'], "please choose from 'smoothed' 'homogeneous', and 'linear'"
    v = v_true.clone()
    v_np = v.cpu().numpy()
    v_np = v_normalize(v_np)
    
    if initial_type == 'smoothed':
        # Apply Gaussian blur per-sample and per-channel only; avoid cross-sample/channel blur
        v_blurred = gaussian_filter(v_np, sigma=(0, 0, sigma, sigma))
    elif initial_type == 'homogeneous':
        # Per-sample: use each sample's own top-row minimum
        B = v_np.shape[0]
        v_blurred = np.empty_like(v_np)
        for b in range(B):
            min_top_row = np.min(v_np[b, 0, 0, :])
            v_blurred[b] = min_top_row
    elif initial_type == 'linear':
        # Per-sample: use each sample's own velocity range
        B = v_np.shape[0]
        height = v_np.shape[2]
        width = v_np.shape[3]
        v_blurred = np.empty_like(v_np)
        for b in range(B):
            v_min = np.min(v_np[b])
            v_max = np.max(v_np[b])
            depth_gradient = np.linspace(v_min, v_max, height).reshape(-1, 1)
            v_blurred[b, 0] = np.tile(depth_gradient, (1, width))
    
    # Move to the same device as input tensor to avoid device mismatch
    v_blurred = torch.tensor(v_blurred).float().to(v_true.device)
    return v_blurred

def missing_trace(y, num_missing, return_mask=False, generator=None):
    assert num_missing >= 0, "The number of missing traces must be greater than 0"
    mask = torch.ones_like(y)
    if num_missing == 0:
        return (y, mask) if return_mask else y
    y_np = y.detach().cpu().numpy()
    batch_size, num_sources, time_samples, num_traces = y.shape
    y_missing = y_np.copy()
    mask_np = np.ones_like(y_np)
    rng = generator if generator is not None else np.random
    for b in range(batch_size):
        for s in range(num_sources):
            missing_indices = rng.choice(num_traces, num_missing, replace=False)
            y_missing[b, s, :, missing_indices] = 0
            mask_np[b, s, :, missing_indices] = 0
    result = torch.tensor(y_missing).float()
    mask = torch.tensor(mask_np).float()
    return (result, mask) if return_mask else result