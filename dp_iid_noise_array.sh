#!/bin/bash
#SBATCH --job-name=dp_iid_noise
#SBATCH --output=logs/dp_iid_noise_%A_%a.out
#SBATCH --error=logs/dp_iid_noise_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3

# dp_iid_noise_array.sh
# Runs three IID heterogeneous-DP-noise profiles with no malicious clients.
# Submit: sbatch dp_iid_noise_array.sh

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/speed-scratch/ol_tal/simulation/clinical_dt_reviewer}
MAIN_SCRIPT=${MAIN_SCRIPT:-main8_reviewer_update_timeseries.py}
RESULTS_ROOT=${RESULTS_ROOT:-$PROJECT_DIR/results_dp_iid_noise/physionet2012}
VENV_DIR=${VENV_DIR:-$PROJECT_DIR/.venv}

DATASET=${DATASET:-physionet2012}
PHYSIONET_DIR=${PHYSIONET_DIR:-$PROJECT_DIR/data/physionet2012/set-a}

mkdir -p logs
cd "$PROJECT_DIR"

if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
fi

NOISE_PROFILES=(
  "0.0,0.0005,0.001,0.003,0.005,0.01,0.02"
  "0.0,0.001,0.003,0.005,0.01,0.02,0.03"
  "0.0,0.002,0.005,0.01,0.02,0.03,0.05"
)

PROFILE=${NOISE_PROFILES[$SLURM_ARRAY_TASK_ID]}
PROFILE_NAME=$(echo "$PROFILE" | sed 's/,/_/g' | sed 's/\./p/g')
OUT_DIR="$RESULTS_ROOT/noise_${PROFILE_NAME}/trial_1/method_adaptive"
mkdir -p "$OUT_DIR"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATASET=$DATASET"
echo "NOISE_PROFILE=$PROFILE"
echo "OUT_DIR=$OUT_DIR"

python "$MAIN_SCRIPT" \
  --dataset "$DATASET" \
  --physionet2012_dir "$PHYSIONET_DIR" \
  --physionet2012_set a \
  --physionet2012_target In-hospital_death \
  --ts_max_len 48 \
  --rounds 50 \
  --clients 10 \
  --validators 5 \
  --validator_val_size 500 \
  --knn_sample 500 \
  --knn_k 5 \
  --local_epochs 2 \
  --batch_size 32 \
  --lr 0.003 \
  --dirichlet_alpha 100.0 \
  --base_noise 0.0 \
  --clip_norm 2.0 \
  --alpha 0.1 \
  --beta 0.9 \
  --pbft_acceptance_delta 0.0 \
  --method adaptive \
  --attack_type none \
  --malicious_ratio 0.0 \
  --heterogeneous_dp \
  --dp_noise_multipliers "$PROFILE" \
  --out_dir "$OUT_DIR"
