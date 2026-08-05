#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_malicious_sensitivity.py

Run a malicious-client sensitivity experiment from 0% to 50% malicious clients.
This wrapper calls your main FL experiment script for each method and malicious ratio.

Example:
python run_malicious_sensitivity.py --main_script main5_timeseries_shapley_dp_latency_efficiency_update.py  --dataset cifar10 --rounds 20 --clients 10 --validators 5  --validator_val_size 1000 --knn_sample 1000 --knn_k 5  --local_epochs 2 --batch_size 64 --lr 0.005  --dirichlet_alpha 1000 --base_noise 0.000 --clip_norm 2.0 --alpha 0.1 --beta 0.9 --pbft_acceptance_delta 0.0001  --attack_type label_flip  --ratios 0.0,0.1,0.2,0.3,0.4,0.5,0.6 --methods adaptive fedavg fedprox fedsgd multikrum median  --out_root ./new_result/malicious_sensitivity/cifar10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_ratios(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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
        "--dirichlet_alpha", str(args.dirichlet_alpha),
        "--base_noise", str(args.base_noise),
    ]

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

    if args.alpha is not None:
        common += ["--alpha", str(args.alpha)]
    if args.beta is not None:
        common += ["--beta", str(args.beta)]
    if args.pbft_acceptance_delta is not None:
        common += ["--pbft_acceptance_delta", str(args.pbft_acceptance_delta)]

    if args.heterogeneous_dp:
        common += ["--heterogeneous_dp"]
    if args.dp_noise_multipliers:
        common += ["--dp_noise_multipliers", args.dp_noise_multipliers]

    if args.extra:
        common += args.extra

    return common


def main() -> None:
    parser = argparse.ArgumentParser(description="Run malicious-client sensitivity experiments.")
    parser.add_argument("--main_script", required=True, help="Path to main FL experiment script.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--physionet2012_dir", default=None)
    parser.add_argument("--physionet2012_set", default="a")
    parser.add_argument("--physionet2012_target", default="In-hospital_death")
    parser.add_argument("--ts_max_len", type=int, default=48)

    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--validators", type=int, default=5)
    parser.add_argument("--validator_val_size", type=int, default=500)
    parser.add_argument("--knn_sample", type=int, default=1000)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--dirichlet_alpha", type=float, default=100)
    parser.add_argument("--base_noise", type=float, default=0.0)
    parser.add_argument("--clip_norm", type=float, default=None)

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--pbft_acceptance_delta", type=float, default=0.0001)

    parser.add_argument("--heterogeneous_dp", action="store_true")
    parser.add_argument("--dp_noise_multipliers", default=None)

    parser.add_argument("--attack_type", default="label_flip",
                        choices=["label_flip", "sign_flip", "scaling", "random_update", "gaussian_model_poisoning"])
    parser.add_argument("--ratios", default="0.0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--methods", nargs="+", default=[
        "adaptive", "fedavg", "fedprox", "fedsgd",
        "krum", "multikrum", "median", "trimmed_mean", "bulyan"
    ])
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=None,
                        help="Extra arguments passed to the main script. Put this last.")

    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    main_script = Path(args.main_script)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    common = build_common_args(args)

    commands_file = out_root / "malicious_sensitivity_commands.txt"
    with commands_file.open("w", encoding="utf-8") as f:
        for ratio in ratios:
            effective_attack = "none" if ratio <= 0 else args.attack_type
            for method in args.methods:
                out_dir = out_root / f"attack_{effective_attack}" / f"ratio_{ratio:.2f}" / f"method_{method}"
                cmd = [
                    args.python,
                    str(main_script),
                    *common,
                    "--method", method,
                    "--attack_type", effective_attack,
                    "--malicious_ratio", str(ratio),
                    "--out_dir", str(out_dir),
                ]
                f.write(" ".join(cmd) + "\n")

    print(f"Saved command list: {commands_file}")

    for ratio in ratios:
        effective_attack = "none" if ratio <= 0 else args.attack_type
        for method in args.methods:
            out_dir = out_root / f"attack_{effective_attack}" / f"ratio_{ratio:.2f}" / f"method_{method}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.python,
                str(main_script),
                *common,
                "--method", method,
                "--attack_type", effective_attack,
                "--malicious_ratio", str(ratio),
                "--out_dir", str(out_dir),
            ]

            print("\n" + "=" * 100)
            print(f"Running method={method}, malicious_ratio={ratio:.2f}, attack={effective_attack}")
            print(" ".join(cmd))
            print("=" * 100)

            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                msg = f"FAILED: method={method}, ratio={ratio:.2f}, attack={effective_attack}"
                if args.continue_on_error:
                    print("WARNING:", msg)
                    continue
                raise RuntimeError(msg)

    print("\nAll requested malicious-sensitivity runs completed.")


if __name__ == "__main__":
    main()
