# Cluster Quickstart

Recommended cluster path:

```bash
mkdir -p /speed-scratch/ol_tal/simulation/clinical_dt_reviewer
cd /speed-scratch/ol_tal/simulation/clinical_dt_reviewer
```

After uploading/unzipping the project:

```bash
chmod +x scripts/*.sh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run a short sanity check:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,DATASET=pathmnist,ATTACK_TYPE=label_flip,MALICIOUS_RATIO=0.2,DIRICHLET_ALPHA=0.05,ROUNDS=5 scripts/run_single_reviewer_job.sh
```

Then submit all sweeps:

```bash
bash scripts/submit_reviewer_sweeps.sh
```
