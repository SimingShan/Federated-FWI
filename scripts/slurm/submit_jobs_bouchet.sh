#!/bin/bash
# Submits 108 individual jobs for geo_split sweep on bouchet.
# Run on the login node: bash scripts/slurm/submit_jobs_bouchet.sh

REG_LAMBDAS=(0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)
LOCAL_LRS=(0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02)
NUM_PATCHES=(3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7 3 3 3 3 5 5 5 5 7 7 7 7)
SERVER_MOMENTUMS=(0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9 0.6 0.7 0.8 0.9)

mkdir -p /home/ss5235/scratch_pi_ll2247/ss5235/Federated-FWI/logs

for IDX in $(seq 0 107); do
    RL=${REG_LAMBDAS[$IDX]}
    LR=${LOCAL_LRS[$IDX]}
    NP=${NUM_PATCHES[$IDX]}
    SM=${SERVER_MOMENTUMS[$IDX]}

    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=sg_rl${RL}_lr${LR}_np${NP}_sm${SM}
#SBATCH --output=/home/ss5235/scratch_pi_ll2247/ss5235/Federated-FWI/logs/sweep_rl${RL}_lr${LR}_np${NP}_sm${SM}.out
#SBATCH --error=/home/ss5235/scratch_pi_ll2247/ss5235/Federated-FWI/logs/sweep_rl${RL}_lr${LR}_np${NP}_sm${SM}.err
#SBATCH --chdir=/home/ss5235/scratch_pi_ll2247/ss5235/Federated-FWI
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=140G
#SBATCH --time=0-03:00:00

module reset
module load miniconda
conda activate /home/ss5235/project_pi_ll2247/ss5235/conda_envs/fwi

python main.py \\
    --config_path configs/combined/diff/config_geo_split.yml \\
    --family combined \\
    --mode federated \\
    experiment.reg_lambda=${RL} \\
    federated.local_lr=${LR} \\
    experiment.num_patches=${NP} \\
    experiment.server_momentum=${SM}
EOF

    echo "Submitted IDX=$IDX: rl=$RL lr=$LR np=$NP sm=$SM"
done

echo "Done — 108 jobs submitted to bouchet."
