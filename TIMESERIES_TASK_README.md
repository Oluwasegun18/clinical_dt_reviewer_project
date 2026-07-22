# Added predictive clinical time-series task

This version adds three time-series dataset modes to `main8_reviewer_update_timeseries.py`:

1. `--dataset physionet2012`
   - Intended final paper task.
   - Uses the PhysioNet/Computing in Cardiology Challenge 2012 raw ICU records.
   - Task: patient-level in-hospital mortality prediction from the first 48 hours of physiological/laboratory measurements.
   - Expected data layout:
     ```text
     data/physionet2012/set-a/
     ├── Outcomes-a.txt
     ├── 132539.txt
     ├── 132540.txt
     └── ...
     ```

2. `--dataset timeseries_csv`
   - Generic long-format clinical time-series CSV support.
   - Expected format: one row per patient-time observation.
   - Required columns by default: `patient_id`, `time`, `target`.
   - Example columns:
     ```text
     patient_id,time,heart_rate,spo2,sbp,dbp,creatinine,target
     ```

3. `--dataset synthetic_physio_ts`
   - Synthetic physiological time-series dataset for smoke testing only.
   - Do not use as final paper evidence.

## Smoke test

```bash
python main8_reviewer_update_timeseries.py \
  --dataset synthetic_physio_ts \
  --synthetic_ts_patients 200 \
  --ts_max_len 24 \
  --synthetic_ts_features 8 \
  --rounds 2 \
  --clients 5 \
  --validators 3 \
  --validator_val_size 30 \
  --knn_sample 30 \
  --knn_k 2 \
  --local_epochs 1 \
  --batch_size 16 \
  --method fedavg adaptive \
  --min_selected 2 \
  --out_dir ./results_ts/smoke
```

## PhysioNet 2012 benchmark run

```bash
python main8_reviewer_update_timeseries.py \
  --dataset physionet2012 \
  --physionet2012_dir ./data/physionet2012/set-a \
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
  --method fedavg fedprox fedsgd krum multikrum median trimmed_mean bulyan \
  --out_dir ./results_ts/physionet2012_benchmarks
```

## PhysioNet 2012 adaptive run

```bash
python main8_reviewer_update_timeseries.py \
  --dataset physionet2012 \
  --physionet2012_dir ./data/physionet2012/set-a \
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
  --alpha 0.3 \
  --beta 0.7 \
  --pbft_acceptance_delta 0.0 \
  --method adaptive \
  --out_dir ./results_ts/physionet2012_adaptive
```

## Generic CSV example

```bash
python main8_reviewer_update_timeseries.py \
  --dataset timeseries_csv \
  --ts_csv ./data/my_clinical_timeseries.csv \
  --ts_patient_col patient_id \
  --ts_time_col hour \
  --ts_target_col mortality \
  --ts_feature_cols heart_rate,spo2,sbp,dbp,resp_rate,creatinine \
  --ts_max_len 48 \
  --rounds 50 \
  --clients 10 \
  --method adaptive \
  --out_dir ./results_ts/my_timeseries_adaptive
```
