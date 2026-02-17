"""
Generate Marmousi seismic data for 4 federated scenarios.

All scenarios share the same geographic base split:
  Client 1: sources 0-4 (left half), receivers 0-94 (left half)
  Client 2: sources 5-9 (right half), receivers in right half

Scenarios differ by additional heterogeneity:
  geo_split     – pure geographic split (baseline)
  snr_split     – geographic split + Gaussian noise on client 2
  freq_split    – geographic split + different Ricker wavelet frequencies (5 Hz vs 20 Hz)
  density_split – geographic split + client 2 has sparse receivers (every 3rd grid point)

Usage:
  python scripts/generate_marmousi_data.py --velocity_model dataset/marmousi/velocity_model/marmousi.npy
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Allow importing project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pde_solvers.client_pde_solver import FWIForward


def build_forward(ctx, device):
    """Build a FWIForward instance without normalization for raw data generation."""
    return FWIForward(ctx, device, normalize=False)


def run_forward(fwi, v_pad, ctx):
    """Run forward modeling and return numpy array (1, ns, nt, ng)."""
    with torch.no_grad():
        seis = fwi.FWM(v_pad, **ctx)
    return seis.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Generate Marmousi seismic data')
    parser.add_argument('--velocity_model', type=str,
                        default='dataset/marmousi/velocity_model/marmousi.npy',
                        help='Path to velocity model .npy file (1, 1, 70, 190)')
    parser.add_argument('--output_dir', type=str,
                        default='dataset/marmousi/seismic_data',
                        help='Root output directory for seismic data')
    parser.add_argument('--noise_std_fraction', type=float, default=0.05,
                        help='Noise std as fraction of max amplitude for snr_split')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load velocity model ---
    vm = np.load(args.velocity_model)
    assert vm.shape == (1, 1, 70, 190), f"Expected (1, 1, 70, 190), got {vm.shape}"
    v_tensor = torch.from_numpy(vm).float().to(device)
    model_name = os.path.splitext(os.path.basename(args.velocity_model))[0]  # e.g. "marmousi" or "overthrust"
    out_fname = f"{model_name}.npy"
    print(f"Loaded velocity model: {vm.shape}, output filename: {out_fname}")

    # --- Shared physics ---
    n_grid = 190
    nz = 70
    nt = 1000
    dx = 10
    nbc = 120
    dt = 1e-3
    f_default = 15
    sz = 10
    gz = 10
    ng = 190
    ns = 10

    sx_all = np.linspace(0, n_grid - 1, num=ns) * dx  # 10 sources across 190 grid points
    gx_all = np.linspace(0, n_grid - 1, num=ng) * dx   # 190 receivers

    # --- Geographic split definitions ---
    n_half_src = ns // 2        # 5 sources per client
    n_half_recv = ng // 2       # 95 receivers per client
    src_left = slice(0, n_half_src)          # sources 0-4
    src_right = slice(n_half_src, ns)        # sources 5-9
    recv_left = slice(0, n_half_recv)        # receivers 0-94
    recv_right = slice(n_half_recv, ng)      # receivers 95-189

    base_ctx = {
        'n_grid': n_grid, 'nt': nt, 'dx': dx, 'nbc': nbc,
        'dt': dt, 'f': f_default, 'sz': sz, 'gz': gz, 'ng': ng,
        'ns': ns, 'sx': sx_all, 'gx': gx_all,
    }

    # Pad velocity model
    v_pad = F.pad(v_tensor, (nbc,) * 4, mode='replicate')

    # FWM context (keys accepted by FWM)
    fwm_ctx = {
        'nbc': nbc, 'dx': dx, 'nt': nt, 'dt': dt, 'f': f_default,
        'sx': sx_all, 'sz': sz, 'gx': gx_all, 'gz': gz,
    }

    # ======================================================================
    # 1. GT: full forward with all 10 sources, f=15Hz, 190 receivers
    # ======================================================================
    print("\n--- Generating GT seismic data ---")
    fwi_gt = build_forward(base_ctx.copy(), device)
    gt_seis = run_forward(fwi_gt, v_pad, fwm_ctx)
    print(f"GT seismic shape: {gt_seis.shape}")  # (1, 10, nt, 190)

    gt_dir = os.path.join(args.output_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    np.save(os.path.join(gt_dir, out_fname), gt_seis)
    print(f"Saved GT to {gt_dir}/{out_fname}")

    # Helper: slice GT by source and receiver indices
    def slice_client_data(seis, src_slice, recv_slice):
        """Slice seismic data: (B, ns, nt, ng) -> (B, n_src, nt, n_recv)"""
        return seis[:, src_slice, :, :][:, :, :, recv_slice]

    # ======================================================================
    # 2. geo_split: pure geographic split
    #    Client 1: sources 0-4, receivers 0-94
    #    Client 2: sources 5-9, receivers 95-189
    # ======================================================================
    print("\n--- Generating geo_split ---")
    geo_dir = os.path.join(args.output_dir, 'geo_split')
    for cname, s_slc, r_slc in [('client1', src_left, recv_left),
                                  ('client2', src_right, recv_right)]:
        cdir = os.path.join(geo_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        client_data = slice_client_data(gt_seis, s_slc, r_slc)
        np.save(os.path.join(cdir, out_fname), client_data)
        print(f"  {cname}: {client_data.shape}")

    # ======================================================================
    # 3. snr_split: geographic split + Gaussian noise on client 2
    #    Client 1: clean data (sources 0-4, receivers 0-94)
    #    Client 2: noisy data (sources 5-9, receivers 95-189)
    # ======================================================================
    print("\n--- Generating snr_split ---")
    snr_dir = os.path.join(args.output_dir, 'snr_split')
    max_amp = np.abs(gt_seis).max()
    noise_std = args.noise_std_fraction * max_amp
    print(f"  Max amplitude: {max_amp:.6f}, noise_std: {noise_std:.6f}")

    # Client 1: clean, left half
    c1_data = slice_client_data(gt_seis, src_left, recv_left)
    c1_dir = os.path.join(snr_dir, 'client1')
    os.makedirs(c1_dir, exist_ok=True)
    np.save(os.path.join(c1_dir, out_fname), c1_data)
    print(f"  client1 (clean): {c1_data.shape}")

    # Client 2: noisy, right half
    c2_data = slice_client_data(gt_seis, src_right, recv_right)
    rng = np.random.default_rng(42)
    c2_data_noisy = c2_data + rng.normal(0, noise_std, c2_data.shape).astype(np.float32)
    c2_dir = os.path.join(snr_dir, 'client2')
    os.makedirs(c2_dir, exist_ok=True)
    np.save(os.path.join(c2_dir, out_fname), c2_data_noisy)
    print(f"  client2 (noisy): {c2_data_noisy.shape}")

    # ======================================================================
    # 4. freq_split: geographic split + different source frequencies
    #    Client 1: f=5Hz, sources 0-4, receivers 0-94
    #    Client 2: f=20Hz, sources 5-9, receivers 95-189
    # ======================================================================
    print("\n--- Generating freq_split ---")
    freq_dir = os.path.join(args.output_dir, 'freq_split')

    for cname, freq, s_slc, r_slc in [('client1', 5, src_left, recv_left),
                                        ('client2', 20, src_right, recv_right)]:
        print(f"  Running forward with f={freq}Hz...")
        freq_fwm_ctx = {
            'nbc': nbc, 'dx': dx, 'nt': nt, 'dt': dt, 'f': freq,
            'sx': sx_all, 'sz': sz, 'gx': gx_all, 'gz': gz,
        }
        freq_base_ctx = base_ctx.copy()
        freq_base_ctx['f'] = freq
        fwi_freq = build_forward(freq_base_ctx, device)
        freq_seis = run_forward(fwi_freq, v_pad, freq_fwm_ctx)
        # Slice to this client's geographic region
        client_data = slice_client_data(freq_seis, s_slc, r_slc)
        cdir = os.path.join(freq_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        np.save(os.path.join(cdir, out_fname), client_data)
        print(f"  {cname} (f={freq}Hz): {client_data.shape}")

    # ======================================================================
    # 5. density_split: geographic split + sparse receivers for client 2
    #    Client 1: sources 0-4, receivers 0-94 (dense, every grid point)
    #    Client 2: sources 5-9, receivers 95,98,101,... (sparse, every 3rd)
    # ======================================================================
    print("\n--- Generating density_split ---")
    density_dir = os.path.join(args.output_dir, 'density_split')
    sparse_stride = 3
    sparse_recv_indices = np.arange(n_half_recv, ng, sparse_stride)  # [95, 98, ..., 188]

    # Client 1: dense left receivers
    c1_data = slice_client_data(gt_seis, src_left, recv_left)
    c1_dir = os.path.join(density_dir, 'client1')
    os.makedirs(c1_dir, exist_ok=True)
    np.save(os.path.join(c1_dir, out_fname), c1_data)
    print(f"  client1 (dense, stride=1): {c1_data.shape}")

    # Client 2: sparse right receivers
    c2_data = gt_seis[:, src_right, :, :][:, :, :, sparse_recv_indices]
    c2_dir = os.path.join(density_dir, 'client2')
    os.makedirs(c2_dir, exist_ok=True)
    np.save(os.path.join(c2_dir, out_fname), c2_data)
    print(f"  client2 (sparse, stride={sparse_stride}): {c2_data.shape}")

    print("\n--- Data generation complete ---")
    print(f"All data saved under: {args.output_dir}")
    print(f"\nAll scenarios use geographic base split:")
    print(f"  Client 1: {n_half_src} sources (left), {n_half_recv} receivers (left)")
    print(f"  Client 2: {n_half_src} sources (right), receivers vary by scenario")


if __name__ == '__main__':
    main()
