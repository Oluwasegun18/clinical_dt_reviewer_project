#!/bin/bash
set -euo pipefail

# Edit this once, then run: bash submit_reviewer_sweeps.sh
export PROJECT_DIR="${PROJECT_DIR:-/speed-scratch/ol_tal/simulation/clinical_dt_reviewer}"
export SCRIPT="${SCRIPT:-main8_reviewer_update.py}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results_reviewer}"

mkdir -p "$PROJECT_DIR/logs" "$RESULTS_ROOT"

sbatch --export=ALL,RESULTS_ROOT="$RESULTS_ROOT/byzantine_sweep" run_byzantine_sweep_array.sh
sbatch --export=ALL,RESULTS_ROOT="$RESULTS_ROOT/ablation_sweep" run_ablation_sweep_array.sh
sbatch --export=ALL,RESULTS_ROOT="$RESULTS_ROOT/heterodp_sweep" run_heterodp_sweep_array.sh
