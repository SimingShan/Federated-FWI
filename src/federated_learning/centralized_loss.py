import torch

def scenario_aware_seismic_loss(y, predicted_seismic, scenario):
    """
    Compute L1 loss only on source-receiver blocks that exist in the federated setting.
    The mask is built as a UNION of per-client coverage.
    Shapes: (B, num_sources, T, num_receivers)
    """
    if y.shape != predicted_seismic.shape:
        raise ValueError(f"Shape mismatch: y {y.shape} vs pred {predicted_seismic.shape}")

    device = y.device
    mask = torch.zeros_like(y, dtype=y.dtype, device=device)
    B, ns, T, ng = y.shape

    if scenario in ('geo_split', 'snr_split', 'freq_split'):
        mid_src = ns // 2
        mid_recv = ng // 2
        mask[:, :mid_src,  :, :mid_recv] = 1   # client 1 coverage
        mask[:, mid_src:,  :, mid_recv:] = 1    # client 2 coverage
    elif scenario == 'density_split':
        mid_src = ns // 2
        mid_recv = ng // 2
        mask[:, :mid_src, :, :mid_recv] = 1                        # client 1 (dense)
        sparse_indices = list(range(mid_recv, ng, 3))               # [95, 98, ..., 188]
        mask[:, mid_src:, :, sparse_indices] = 1                    # client 2 (sparse)

    diff = (y - predicted_seismic).abs() * mask
    per = diff.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1.0)
    return per.mean()
