#!/bin/bash
#SBATCH --job-name=sweep_geo
#SBATCH --array=0-107
#SBATCH --output=logs/sweep_geo_%A_%a.out
#SBATCH --error=logs/sweep_geo_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --constraint=h100_80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=0-04:00:00
#SBATCH --chdir=/gpfs/radev/home/ss5235/scratch/Federated-FWI

module reset
module load miniconda
conda activate /gpfs/radev/home/ss5235/.conda/envs/fwi

# --- Parameter arrays (indexed by SLURM_ARRAY_TASK_ID) ---
REG_LAMBDAS=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7)
LOCAL_LRS=(0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02)
NUM_PATCHES=(3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7)
SERVER_MOMENTUMS=(0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9)

IDX=$SLURM_ARRAY_TASK_ID
RL=${REG_LAMBDAS[$IDX]}
LR=${LOCAL_LRS[$IDX]}
NP=${NUM_PATCHES[$IDX]}
SM=${SERVER_MOMENTUMS[$IDX]}

echo "=== Task $IDX: rl=$RL lr=$LR np=$NP sm=$SM ==="

mkdir -p logs

python main.py \
    --config_path configs/combined/diff/config_geo_split.yml \
    --family combined \
    --mode federated \
    experiment.reg_lambda=$RL \
    federated.local_lr=$LR \
    experiment.num_patches=$NP \
    experiment.server_momentum=$SM
