#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dp_iid_noise_sensitivity.py

Run an IID heterogeneous-DP-noise experiment.

Goal:
- Make the client data partitions approximately IID by using a large Dirichlet alpha.
- Disable malicious clients.
- Vary DP noise from client to client using --heterogeneous_dp and --dp_noise_multipliers.
- Use the incentive ledger to study:
    1. reward vs DP noise
    2. reward-contribution gap vs DP noise
    3. reward vs validated contribution under IID data

Example:
python run_dp_iid_noise_sensitivity.py \
  --main_script main8_reviewer_update_timeseries.py \
  --dataset physionet2012 \
  --physionet2012_dir ./data/physionet2012/set-a \
  --rounds 50 --clients 10 --validators 5 \
  --validator_val_size 500 --knn_sample 500 --knn_k 5 \
  --local_epochs 2 --batch_size 32 --lr 0.003 \
  --iid_alpha 100.0 \
  --noise_profiles "0.0,0.0005,0.001,0.003,0.005,0.01,0.02" \
  --alpha 0.1 --beta 0.9 \
  --out_root ./results_dp_iid_noise/physionet2012
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def split_profiles(profile_string: str) -> List[str]:
    """
    Allows one or more profiles separated by semicolon.

    Example:
    --noise_profiles "0.0,0.0005,0.001,0.005,0.01;0.0,0.001,0.005,0.02,0.05"
    """
    return [p.strip() for p in profile_string.split(";") if p.strip()]


def profile_name(profile: str) -> str:
    clean = profile.replace(",", "_").replace(".", "p").replace("-", "m")
    return f"noise_{clean}"


def build_common_args(args: argparse.Namespace) -> List[str]:
    common = [
        "--dataset", args.dataset,
        "--rounds", str(args.rounds),
        "--clients", str(args.clients),
        "--validators", str(args.validators),
        "--validator_val_size", str(args.validator_val_size),
        "--knn_sample", str(args.knn_sample),
        "--knn_k", str(args.knn_k),
        "--local_epochs", str(args.local_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),

        # High alpha approximates IID Dirichlet partitioning.
        "--dirichlet_alpha", str(args.iid_alpha),

        # No Byzantine/malicious clients.
        "--attack_type", "none",
        "--malicious_ratio", "0.0",

        # Proposed incentive weighting.
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--pbft_acceptance_delta", str(args.pbft_acceptance_delta),

        # Heterogeneous DP noise across clients.
        "--heterogeneous_dp",
    ]

    # Keep base noise as the minimum/default noise level.
    if args.base_noise is not None:
        common += ["--base_noise", str(args.base_noise)]

    if args.clip_norm is not None:
        common += ["--clip_norm", str(args.clip_norm)]

    if args.physionet2012_dir:
        common += ["--physionet2012_dir", args.physionet2012_dir]
    if args.physionet2012_set:
        common += ["--physionet2012_set", args.physionet2012_set]
    if args.physionet2012_target:
        common += ["--physionet2012_target", args.physionet2012_target]
    if args.ts_max_len is not None:
        common += ["--ts_max_len", str(args.ts_max_len)]

    if args.extra:
        common += args.extra

    return common


def main() -> None:
    parser = argparse.ArgumentParser(description="IID heterogeneous-DP-noise sensitivity experiment.")
    parser.add_argument("--main_script", required=True)
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--physionet2012_dir", default=None)
    parser.add_argument("--physionet2012_set", default="a")
    parser.add_argument("--physionet2012_target", default="In-hospital_death")
    parser.add_argument("--ts_max_len", type=int, default=48)

    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--validators", type=int, default=5)
    parser.add_argument("--validator_val_size", type=int, default=500)
    parser.add_argument("--knn_sample", type=int, default=500)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)

    parser.add_argument("--iid_alpha", type=float, default=100.0,
                        help="Large Dirichlet alpha to approximate IID partitioning.")
    parser.add_argument("--base_noise", type=float, default=0.0)
    parser.add_argument("--clip_norm", type=float, default=2.0)

    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Weight of KS/Shapley contribution in validated contribution/reward.")
    parser.add_argument("--beta", type=float, default=0.9,
                        help="Weight of clinical relevance in validated contribution/reward.")
    parser.add_argument("--pbft_acceptance_delta", type=float, default=0.0)

    parser.add_argument(
        "--noise_profiles",
        default="0.0,0.0005,0.001,0.003,0.005,0.01,0.02",
        help=(
            "One or more comma-separated DP-noise profiles. "
            "Use semicolon to separate multiple profiles."
        ),
    )
    parser.add_argument("--method", default="adaptive",
                        help="Usually adaptive/proposed, because the incentive ledger is generated there.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=None)

    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    common = build_common_args(args)
    profiles = split_profiles(args.noise_profiles)

    commands_path = out_root / "dp_iid_noise_commands.txt"
    with commands_path.open("w", encoding="utf-8") as f:
        for trial in range(1, args.trials + 1):
            for profile in profiles:
                p_name = profile_name(profile)
                out_dir = out_root / p_name / f"trial_{trial}" / f"method_{args.method}"
                cmd = [
                    args.python,
                    args.main_script,
                    *common,
                    "--method", args.method,
                    "--dp_noise_multipliers", profile,
                    "--trial", str(trial),
                    "--out_dir", str(out_dir),
                ]
                f.write(" ".join(cmd) + "\n")

    print(f"Saved command list: {commands_path}")

    for trial in range(1, args.trials + 1):
        for profile in profiles:
            p_name = profile_name(profile)
            out_dir = out_root / p_name / f"trial_{trial}" / f"method_{args.method}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python,
                args.main_script,
                *common,
                "--method", args.method,
                "--dp_noise_multipliers", profile,
                "--trial", str(trial),
                "--out_dir", str(out_dir),
            ]

            print("\n" + "=" * 100)
            print(f"Running IID DP-noise experiment: trial={trial}, profile={profile}")
            print(" ".join(cmd))
            print("=" * 100)

            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                msg = f"FAILED: trial={trial}, profile={profile}"
                if args.continue_on_error:
                    print("WARNING:", msg)
                    continue
                raise RuntimeError(msg)

    print("\nAll IID DP-noise sensitivity runs completed.")


if __name__ == "__main__":
    main()
