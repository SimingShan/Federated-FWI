# Marmousi Data Generation & Scenario Reorganization — Change Summary

## Overview

Replaced the 5-family (CF/CV/FF/FV) 70x70 velocity model system with a single Marmousi model (70x190) and 4 physics-motivated federated scenarios.

## Geographic Base Split (All Scenarios)

**Every scenario** shares the same geographic constraint — each client only has access to sources and receivers in their half of the domain:

| | Client 1 (left) | Client 2 (right) |
|---|---|---|
| **Sources** | 0–4 (5 sources) | 5–9 (5 sources) |
| **Receivers** | 0–94 (95 receivers, dense) | 95–189 (95 dense or 32 sparse) |

Shared physics: 10 total sources, 190 total receivers, n_grid=190, nz=70, dx=10m, dt=1e-3, nt=1000, nbc=120.

## The 4 Scenarios

| Scenario | Client 1 data shape | Client 2 data shape | What differs beyond geography |
|----------|--------------------|--------------------|-------------------------------|
| `geo_split` | `(1, 5, 1000, 95)` | `(1, 5, 1000, 95)` | Nothing (baseline) |
| `snr_split` | `(1, 5, 1000, 95)` | `(1, 5, 1000, 95)` | Client 2 has Gaussian noise (std=0.05×max) |
| `freq_split` | `(1, 5, 1000, 95)` | `(1, 5, 1000, 95)` | Client 1: f=5Hz, Client 2: f=20Hz |
| `density_split` | `(1, 5, 1000, 95)` | `(1, 5, 1000, 32)` | Client 2 has sparse receivers (every 3rd grid point) |

---

## Files

### `scripts/generate_marmousi_data.py`
- Loads `dataset/marmousi/velocity_model/marmousi.npy` (shape `(1,1,70,190)`)
- Runs forward modeling to produce GT seismic `(1,10,1000,190)`
- Generates per-client data for all 4 scenarios using the geographic base split:
  - `geo_split`: slices GT by source AND receiver indices
  - `snr_split`: slices GT + adds Gaussian noise to client 2
  - `freq_split`: separate forward passes at f=5Hz and f=20Hz, then slices per client
  - `density_split`: client 1 gets dense left receivers (95), client 2 gets sparse right receivers (32, every 3rd)
- Output tree: `dataset/marmousi/seismic_data/{gt,geo_split,snr_split,freq_split,density_split}/`

### `configs/marmousi/{diff,tv,tik,none}/config_{geo,snr,freq,density}_split.yml` (16 files)
- `diff`: regularization=diffusion, reg_lambda=0.75
- `tv`: regularization=total_variation, reg_lambda=0.3
- `tik`: regularization=tikhonov, reg_lambda=0.1
- `none`: regularization=none, reg_lambda=0.0
- All use: n_grid=190, ns=10, ng=190, num_clients=2
- `freq_split` configs add `forward.client_frequencies: [5, 20]`
- `density_split` configs add `forward.client_receiver_strides: [1, 3]`

### `src/pde_solvers/client_pde_solver.py` — FWIForward

**`__init__`**:
- `gx` in ctx is a 1D array of grid indices, multiplied by `dx` at init time
- No scenario-specific logic; geometry is fully determined by ctx

**`forward(self, v, scenario=None, client_idx=None, num_clients=None)`**:
- Simplified: no scenario-based source/receiver splitting
- All geometry (sx, gx) is determined at init time via the ctx dict
- `scenario`, `client_idx`, `num_clients` args kept for API compatibility but ignored

### `src/run_federated.py`

- **ALL** scenarios build `per_client_fwi_forwards`
- Each per-client FWIForward has client-specific:
  - `sx`: 5 source positions (physical units)
  - `gx`: receiver grid indices (95 for dense, 32 for sparse)
  - `ns`: 5, `ng`: varies by scenario
  - For `freq_split`: also client-specific `f`
  - For `density_split`: receiver grid indices use config strides `[1, 3]`
- The shared `fwi_forward` (all 10 sources, 190 receivers) is still used for server-side evaluation

### `src/federated_learning/centralized_loss.py`

- `scenario_aware_seismic_loss()` supports all 4 scenarios
- `density_split` mask: client 1 covers sources 0–4 × receivers 0–94 (dense), client 2 covers sources 5–9 × receivers [95, 98, 101, ..., 188] (sparse)

---

## Quick-Start

```bash
# Step 1: Generate data (requires marmousi.npy velocity model)
python scripts/generate_marmousi_data.py \
    --velocity_model dataset/marmousi/velocity_model/marmousi.npy

# Step 2: Run one federated experiment
python main.py \
    --config_path configs/marmousi/tv/config_density_split.yml \
    --family marmousi \
    --mode federated
```
