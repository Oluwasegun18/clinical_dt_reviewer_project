#!/bin/bash
#SBATCH --job-name=mal_sens
#SBATCH --output=logs/mal_sens_%A_%a.out
#SBATCH --error=logs/mal_sens_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-53%8

# malicious_sensitivity_array.sh
# 6 ratios x 9 methods = 54 jobs.
# Ratios: 0%, 10%, 20%, 30%, 40%, 50%.
# Submit: sbatch malicious_sensitivity_array.sh

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/speed-scratch/ol_tal/simulation/clinical_dt_reviewer}
MAIN_SCRIPT=${MAIN_SCRIPT:-main8_reviewer_update_timeseries.py}
RESULTS_ROOT=${RESULTS_ROOT:-$PROJECT_DIR/results_malicious_sensitivity/physionet2012_signflip}
VENV_DIR=${VENV_DIR:-$PROJECT_DIR/.venv}

DATASET=${DATASET:-physionet2012}
PHYSIONET_DIR=${PHYSIONET_DIR:-$PROJECT_DIR/data/physionet2012/set-a}
ATTACK_TYPE=${ATTACK_TYPE:-sign_flip}

mkdir -p logs
cd "$PROJECT_DIR"

if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
fi

RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5)
METHODS=(adaptive fedavg fedprox fedsgd krum multikrum median trimmed_mean bulyan)

NUM_METHODS=${#METHODS[@]}
RATIO_IDX=$((SLURM_ARRAY_TASK_ID / NUM_METHODS))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))

MALICIOUS_RATIO=${RATIOS[$RATIO_IDX]}
METHOD=${METHODS[$METHOD_IDX]}

if [ "$MALICIOUS_RATIO" = "0.0" ]; then
  EFFECTIVE_ATTACK="none"
else
  EFFECTIVE_ATTACK="$ATTACK_TYPE"
fi

OUT_DIR="$RESULTS_ROOT/attack_${EFFECTIVE_ATTACK}/ratio_${MALICIOUS_RATIO}/method_${METHOD}"
mkdir -p "$OUT_DIR"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATASET=$DATASET"
echo "METHOD=$METHOD"
echo "ATTACK=$EFFECTIVE_ATTACK"
echo "MALICIOUS_RATIO=$MALICIOUS_RATIO"
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
  --dirichlet_alpha 0.05 \
  --base_noise 0.0005 \
  --clip_norm 2.0 \
  --alpha 0.1 \
  --beta 0.9 \
  --pbft_acceptance_delta 0.0 \
  --method "$METHOD" \
  --attack_type "$EFFECTIVE_ATTACK" \
  --malicious_ratio "$MALICIOUS_RATIO" \
  --out_dir "$OUT_DIR"
