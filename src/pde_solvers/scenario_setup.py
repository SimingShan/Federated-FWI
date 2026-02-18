import copy
from typing import List, Optional

import numpy as np
import torch

from src.pde_solvers.client_pde_solver import FWIForward
from src.utils.data_trans import v_denormalize, s_normalize_none

SPLIT_SCENARIOS = ('geo_split', 'snr_split', 'freq_split', 'density_split')


def build_fwi_ctx(config) -> dict:
    """Build the base FWI context dict from config. Callers may extend it (e.g. add 'sx')."""
    return {
        'n_grid': config.forward.n_grid,
        'nt': config.forward.nt,
        'dx': config.forward.dx,
        'nbc': config.forward.nbc,
        'dt': config.forward.dt,
        'f': config.forward.f,
        'sz': config.forward.sz,
        'gz': config.forward.gz,
        'ng': config.forward.ng,
        'ns': getattr(config.forward, 'ns', 10),
    }


def build_per_client_fwi_forwards(
    config, ctx: dict, device: torch.device
) -> Optional[List[FWIForward]]:
    """Build per-client FWI forward solvers for heterogeneous split scenarios.

    Returns None for the no-split (homogeneous) case, so callers can use a
    single shared solver instead.
    """
    scenario_flag = config.experiment.scenario_flag
    if scenario_flag not in SPLIT_SCENARIOS:
        return None

    n_grid = config.forward.n_grid
    ns = getattr(config.forward, 'ns', 10)
    ng = config.forward.ng
    dx = config.forward.dx

    sx_all = np.linspace(0, n_grid - 1, num=ns) * dx
    n_half_src = ns // 2
    n_half_recv = ng // 2

    client_sx = [sx_all[:n_half_src], sx_all[n_half_src:]]
    client_gx = [np.arange(n_half_recv), np.arange(n_half_recv, ng)]

    if scenario_flag == 'density_split':
        strides = list(config.forward.client_receiver_strides)
        client_gx = [np.arange(0, n_half_recv, strides[0]),
                     np.arange(n_half_recv, ng, strides[1])]

    per_client_fwi_forwards = []
    for c in range(config.experiment.num_clients):
        client_ctx = copy.deepcopy(ctx)
        client_ctx['sx'] = client_sx[c]
        client_ctx['gx'] = client_gx[c]
        client_ctx['ns'] = n_half_src
        client_ctx['ng'] = len(client_gx[c])
        if scenario_flag == 'freq_split':
            client_ctx['f'] = config.forward.client_frequencies[c]
        per_client_fwi_forwards.append(
            FWIForward(client_ctx, device, normalize=True,
                       v_denorm_func=v_denormalize, s_norm_func=s_normalize_none)
        )

    return per_client_fwi_forwards
