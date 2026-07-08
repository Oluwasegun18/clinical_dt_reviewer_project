#!/bin/bash -l
#SBATCH --job-name=pomdt_heterodp
#SBATCH --partition=ps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/speed-scratch/ol_tal/simulation/clinical_dt_reviewer}"
SCRIPT="${SCRIPT:-main8_reviewer_update.py}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results_reviewer/heterodp_sweep}"

ROUNDS="${ROUNDS:-50}"
CLIENTS="${CLIENTS:-10}"
VALIDATORS="${VALIDATORS:-5}"
VALIDATOR_VAL_SIZE="${VALIDATOR_VAL_SIZE:-2000}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.005}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.05}"
BASE_NOISE="${BASE_NOISE:-0.0005}"
DP_NOISE_MULTIPLIERS="${DP_NOISE_MULTIPLIERS:-0.0005,0.005,0.02}"
KNN_SAMPLE="${KNN_SAMPLE:-2000}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
SEED_BASE="${SEED_BASE:-2000}"

DATASETS=(pathmnist organamnist)
SCENARIOS=(
  "pathmnist none 0.0"
  "pathmnist label_flip 0.2"
  "pathmnist sign_flip 0.2"
  "organamnist none 0.0"
  "organamnist label_flip 0.2"
  "organamnist sign_flip 0.2"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#SCENARIOS[@]} )); then
  echo "Task $TASK_ID exceeds scenario count ${#SCENARIOS[@]}"
  exit 1
fi
read -r DATASET ATTACK_TYPE MALICIOUS_RATIO <<< "${SCENARIOS[$TASK_ID]}"
SEED=$((SEED_BASE + TASK_ID))

mkdir -p "$PROJECT_DIR/logs" "$RESULTS_ROOT"
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.12.0 || true
fi

if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
fi

export OMP_NUM_THREADS="$N_JOBS"
export MKL_NUM_THREADS="$N_JOBS"
export PYTHONUNBUFFERED=1

OUT_DIR="$RESULTS_ROOT/${DATASET}/attack_${ATTACK_TYPE}/ratio_${MALICIOUS_RATIO}/alpha_${DIRICHLET_ALPHA}/dp_${DP_NOISE_MULTIPLIERS//,/plus}/seed_${SEED}"
mkdir -p "$OUT_DIR"

echo "Heterogeneous-DP scenario $TASK_ID/${#SCENARIOS[@]}: dataset=$DATASET attack=$ATTACK_TYPE ratio=$MALICIOUS_RATIO seed=$SEED"

python -u "$SCRIPT" \
  --dataset "$DATASET" \
  --rounds "$ROUNDS" \
  --clients "$CLIENTS" \
  --validators "$VALIDATORS" \
  --validator_val_size "$VALIDATOR_VAL_SIZE" \
  --local_epochs "$LOCAL_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --dirichlet_alpha "$DIRICHLET_ALPHA" \
  --base_noise "$BASE_NOISE" \
  --heterogeneous_dp \
  --dp_noise_multipliers "$DP_NOISE_MULTIPLIERS" \
  --attack_type "$ATTACK_TYPE" \
  --malicious_ratio "$MALICIOUS_RATIO" \
  --knn_sample "$KNN_SAMPLE" \
  --n_jobs "$N_JOBS" \
  --seed "$SEED" \
  --run_all \
  --out_dir "$OUT_DIR"
