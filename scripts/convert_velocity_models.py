"""
Convert raw velocity models (Foothill, BP salt) to the standard format
used by the project: shape (1, 1, 70, 190), float32, velocity range [1500, 4500].

Usage:
  python scripts/convert_velocity_models.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F


def resize_2d(arr, target_h, target_w):
    """Resize a 2D array to (target_h, target_w) using bilinear interpolation."""
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    t = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=True)
    return t.squeeze().numpy()


def rescale(arr, new_min=1500.0, new_max=4500.0):
    """Linearly rescale array values to [new_min, new_max]."""
    old_min, old_max = arr.min(), arr.max()
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def convert_and_save(arr_2d, name, output_dir, target_shape=(70, 190)):
    """Resize, rescale, and save a velocity model."""
    resized = resize_2d(arr_2d, *target_shape)
    rescaled = rescale(resized, 1500.0, 4500.0).astype(np.float32)
    out = rescaled.reshape(1, 1, *target_shape)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{name}.npy')
    np.save(out_path, out)
    print(f"Saved {name}: shape={out.shape}, dtype={out.dtype}, "
          f"min={out.min():.2f}, max={out.max():.2f} -> {out_path}")


def main():
    # --- Foothill ---
    # Text file (801, 331) stored as (nx=801, nz=331) rows; transpose to (nz, nx)
    foothill_path = 'Foothill_801_331_25m.dat'
    if os.path.exists(foothill_path):
        raw = np.loadtxt(foothill_path)  # (801, 331)
        foothill = raw.T  # (nz=331, nx=801)
        print(f"Foothill raw: {foothill.shape}, min={foothill.min():.0f}, max={foothill.max():.0f}")
        convert_and_save(foothill, 'foothill', 'dataset/foothill/velocity_model')
    else:
        print(f"Foothill file not found: {foothill_path}")

    # --- BP salt ---
    # Binary float32 stored column-major (nx=5395, nz=1911); reshape and transpose to (nz, nx)
    bp_path = 'vel_z6.25m_x12.5m_exact.bin'
    if os.path.exists(bp_path):
        raw = np.fromfile(bp_path, dtype=np.float32).reshape(5395, 1911)
        bpsalt = raw.T  # (nz=1911, nx=5395)
        print(f"BP salt raw: {bpsalt.shape}, min={bpsalt.min():.0f}, max={bpsalt.max():.0f}")
        convert_and_save(bpsalt, 'bpsalt', 'dataset/bpsalt/velocity_model')
    else:
        print(f"BP salt file not found: {bp_path}")


if __name__ == '__main__':
    main()
