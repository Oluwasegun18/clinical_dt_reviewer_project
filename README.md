# Clinical DT Reviewer Simulation Project

This project contains the updated simulation code and SLURM scripts for reviewer-response experiments on Byzantine-aware proof-of-model-quality federated learning.

## Contents

```text
clinical_dt_reviewer_project/
├── main8_reviewer_update.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── results_reviewer/
│   └── .gitkeep
└── scripts/
    ├── run_single_reviewer_job.sh
    ├── run_byzantine_sweep_array.sh
    ├── run_ablation_sweep_array.sh
    ├── run_heterodp_sweep_array.sh
    └── submit_reviewer_sweeps.sh
```

## Create a new Git project

```bash
git init
cp -r clinical_dt_reviewer_project/* clinical_dt_reviewer_project/.[!.]* . 2>/dev/null || true
git add .
git commit -m "Initial reviewer-response simulation project"
```

Or, after unzipping directly into the desired repository folder:

```bash
git init
git add .
git commit -m "Initial reviewer-response simulation project"
```

## Python environment

On the cluster, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If the cluster uses modules, load the appropriate Python/CUDA modules before creating the environment.

## Run one test job

Edit the `PROJECT_DIR` in the scripts if needed, then run:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,DATASET=pathmnist,ATTACK_TYPE=label_flip,MALICIOUS_RATIO=0.2,DIRICHLET_ALPHA=0.05,ROUNDS=5 scripts/run_single_reviewer_job.sh
```

Check job status:

```bash
squeue -u $USER
```

Check logs:

```bash
tail -n 50 logs/*.out
tail -n 50 logs/*.err
```

## Submit full reviewer sweeps

```bash
bash scripts/submit_reviewer_sweeps.sh
```

The sweeps include Byzantine robustness, ablation, and heterogeneous-DP fairness experiments.

## Main reviewer-response experiment categories

1. Genuine Byzantine attack experiments
2. Robust FL baselines
3. Ablation study
4. Fairness metrics
5. Heterogeneous-DP fairness analysis
6. Optional future extension to time-series/predictive clinical tasks

## Notes

Large datasets, logs, and results are ignored by Git by default. Keep reproducible scripts and small configuration files in Git, but store large outputs in scratch storage or external artifact storage.
