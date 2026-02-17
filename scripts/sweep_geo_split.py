#!/usr/bin/env python3
"""Generate SLURM array job scripts for geo_split hyperparameter sweep.

Grid (216 total):
  reg_lambda:      [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  — 6 values
  local_lr:        [0.005, 0.01, 0.02]                — 3 values
  num_patches:     [3, 5, 7]                           — 3 values
  server_momentum: [0.6, 0.7, 0.8, 0.9]               — 4 values

Usage:
  python scripts/sweep_geo_split.py
"""

import itertools
import os
import stat

# --- Sweep grid ---
REG_LAMBDAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LOCAL_LRS = [0.005, 0.01, 0.02]
NUM_PATCHES = [3, 5, 7]
SERVER_MOMENTUMS = [0.6, 0.7, 0.8, 0.9]

CONFIG_PATH = "configs/combined/diff/config_geo_split.yml"
FAMILY = "combined"
MODE = "federated"

# --- Cluster-specific SLURM headers ---
CLUSTER_HEADERS = {
    "misha": """\
#SBATCH --partition=gpu
#SBATCH --constraint=h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=140G
#SBATCH --time=0-06:00:00""",
    "bouchet": """\
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=140G
#SBATCH --time=0-06:00:00""",
}

OUTPUT_DIR = "scripts/slurm"


def generate_combinations():
    """Return list of all (reg_lambda, local_lr, num_patches, server_momentum) combos."""
    combos = list(itertools.product(REG_LAMBDAS, LOCAL_LRS, NUM_PATCHES, SERVER_MOMENTUMS))
    assert len(combos) == 216, f"Expected 216 combinations, got {len(combos)}"
    return combos


def make_bash_arrays(combos):
    """Build bash array declaration strings for each hyperparameter."""
    rl_arr = " ".join(str(c[0]) for c in combos)
    lr_arr = " ".join(str(c[1]) for c in combos)
    np_arr = " ".join(str(c[2]) for c in combos)
    sm_arr = " ".join(str(c[3]) for c in combos)
    return rl_arr, lr_arr, np_arr, sm_arr


def generate_slurm_script(cluster, combos):
    rl_arr, lr_arr, np_arr, sm_arr = make_bash_arrays(combos)
    n_jobs = len(combos)
    header = CLUSTER_HEADERS[cluster]

    script = f"""\
#!/bin/bash
#SBATCH --job-name=sweep_geo
#SBATCH --array=0-{n_jobs - 1}%8
#SBATCH --output=logs/sweep_geo_%A_%a.out
#SBATCH --error=logs/sweep_geo_%A_%a.err
{header}

# --- Parameter arrays (indexed by SLURM_ARRAY_TASK_ID) ---
REG_LAMBDAS=({rl_arr})
LOCAL_LRS=({lr_arr})
NUM_PATCHES=({np_arr})
SERVER_MOMENTUMS=({sm_arr})

IDX=$SLURM_ARRAY_TASK_ID
RL=${{REG_LAMBDAS[$IDX]}}
LR=${{LOCAL_LRS[$IDX]}}
NP=${{NUM_PATCHES[$IDX]}}
SM=${{SERVER_MOMENTUMS[$IDX]}}

echo "=== Task $IDX: rl=$RL lr=$LR np=$NP sm=$SM ==="

mkdir -p logs

python main.py \\
    --config_path {CONFIG_PATH} \\
    --family {FAMILY} \\
    --mode {MODE} \\
    experiment.reg_lambda=$RL \\
    federated.local_lr=$LR \\
    experiment.num_patches=$NP \\
    experiment.server_momentum=$SM
"""
    return script


def main():
    combos = generate_combinations()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cluster in CLUSTER_HEADERS:
        script = generate_slurm_script(cluster, combos)
        path = os.path.join(OUTPUT_DIR, f"sweep_geo_split_{cluster}.sh")
        with open(path, "w") as f:
            f.write(script)
        # Make executable
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        print(f"Generated: {path} ({len(combos)} jobs)")

    # Print a sample command for dry-run verification
    c = combos[0]
    print(f"\nSample dry-run command:")
    print(
        f"  python main.py --config_path {CONFIG_PATH} --family {FAMILY} --mode {MODE} "
        f"experiment.reg_lambda={c[0]} federated.local_lr={c[1]} "
        f"experiment.num_patches={c[2]} experiment.server_momentum={c[3]}"
    )


if __name__ == "__main__":
    main()
