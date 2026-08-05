#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_homogeneous_dp_sweep_v2.py

Run a homogeneous-DP sweep. In each experiment, every client receives the
same DP noise multiplier through --base_noise. The multiplier is then varied
across experiments.

This version accepts extra arguments for the main experiment script directly,
with or without the separator "--".

Do not pass --heterogeneous_dp. This script removes it if supplied by mistake.
Do not pass --dp_noise_multipliers. This script removes it if supplied by mistake.

The runner controls:
  --dataset, --rounds, --base_noise, --seed, --out_dir
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def parse_float_list(text: str) -> List[float]:
    if text is None or str(text).strip() == "":
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip() != ""]


def parse_int_list(text: str) -> List[int]:
    if text is None or str(text).strip() == "":
        return [42]
    return [int(x.strip()) for x in str(text).split(",") if x.strip() != ""]


def safe_float_tag(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def print_tail(path: Path, n_lines: int = 80) -> None:
    """Print the last n_lines of a text file if it exists."""
    if not path.exists():
        print(f"[no file] {path}")
        return

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-int(n_lines):]
        print(f"\n--- Last {len(tail)} lines of {path} ---")
        for line in tail:
            print(line)
        print("--- end error tail ---\n")
    except Exception as exc:
        print(f"[could not read {path}] {exc}")


def remove_conflicting_args(extra: List[str]) -> List[str]:
    no_value_flags = {"--heterogeneous_dp"}
    value_flags = {
        "--dataset",
        "--rounds",
        "--base_noise",
        "--seed",
        "--out_dir",
        "--dp_noise_multipliers",
    }

    cleaned = []
    i = 0

    while i < len(extra):
        token = extra[i]

        if token == "--":
            i += 1
            continue

        if token in no_value_flags:
            i += 1
            continue

        if token in value_flags:
            i += 2
            continue

        if any(token.startswith(flag + "=") for flag in value_flags):
            i += 1
            continue

        cleaned.append(token)
        i += 1

    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run homogeneous-DP experiments by sweeping --base_noise. "
            "Unknown arguments are passed to the main experiment script."
        ),
        allow_abbrev=False,
    )

    parser.add_argument("--main_script", required=True, help="Path to the main experiment Python file.")
    parser.add_argument("--dataset", required=True, help="Dataset name passed to the main experiment.")
    parser.add_argument("--out_root", required=True, help="Root output directory for all DP-noise runs.")

    parser.add_argument(
        "--dp_noise_values",
        default="0.0,0.0005,0.001,0.003,0.005,0.01,0.015,0.02,0.025,0.03",
        help="Comma-separated homogeneous DP noise multipliers.",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds per DP experiment.")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g., 42 or 42,43,44.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip run if incentive_ledger.csv exists.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--error_tail",
        type=int,
        default=80,
        help="Number of run.err lines to print immediately when a run fails.",
    )
    parser.add_argument(
        "--continue_on_failure",
        action="store_true",
        help="Continue the remaining DP sweep even when one DP run fails.",
    )
    parser.add_argument(
        "--force_utf8",
        action="store_true",
        default=True,
        help="Force UTF-8 stdout/stderr for the child experiment process. This avoids Windows cp1252 UnicodeEncodeError.",
    )

    return parser


def main() -> None:
    parser = build_parser()

    # parse_known_args is the key fix. It lets --clients, --validators, --lr,
    # etc. pass through to the main script without requiring a "--" separator.
    args, extra_args = parser.parse_known_args()

    main_script = Path(args.main_script)
    if not main_script.exists():
        raise FileNotFoundError(f"main_script not found: {main_script}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dp_values = parse_float_list(args.dp_noise_values)
    if not dp_values:
        raise ValueError("--dp_noise_values is empty.")

    seeds = parse_int_list(args.seeds)
    extra_args = remove_conflicting_args(extra_args)

    manifest_path = out_root / "homogeneous_dp_sweep_manifest.csv"
    rows = []
    failures = []

    for seed in seeds:
        for noise in dp_values:
            noise_tag = safe_float_tag(noise)
            run_dir = out_root / f"{args.dataset}_dp_{noise_tag}_seed_{seed}"
            ledger_path = run_dir / "incentive_ledger.csv"

            if args.skip_existing and ledger_path.exists():
                print(f"[skip] {run_dir} already has incentive_ledger.csv")
                status = "skipped_existing"
                rows.append({
                    "dataset": args.dataset,
                    "dp_noise_multiplier": noise,
                    "seed": seed,
                    "rounds": args.rounds,
                    "out_dir": str(run_dir),
                    "status": status,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                continue

            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python,
                str(main_script),
                "--dataset", str(args.dataset),
                "--rounds", str(args.rounds),
                "--base_noise", str(noise),
                "--seed", str(seed),
                "--out_dir", str(run_dir),
            ] + extra_args

            print("\n" + "=" * 100)
            print(f"Homogeneous DP run | dataset={args.dataset} | dp_noise={noise} | seed={seed}")
            print("Command:")
            print(" ".join(shlex.quote(x) for x in cmd))
            print("=" * 100)

            status = "dry_run"

            if not args.dry_run:
                log_path = run_dir / "run.log"
                err_path = run_dir / "run.err"

                child_env = os.environ.copy()
                if args.force_utf8:
                    child_env["PYTHONIOENCODING"] = "utf-8"
                    child_env["PYTHONUTF8"] = "1"

                with open(log_path, "w", encoding="utf-8") as log_f, open(err_path, "w", encoding="utf-8") as err_f:
                    proc = subprocess.run(
                        cmd,
                        stdout=log_f,
                        stderr=err_f,
                        text=True,
                        env=child_env,
                    )

                if proc.returncode == 0:
                    status = "success"
                    print(f"[success] {run_dir}")
                else:
                    status = f"failed_returncode_{proc.returncode}"
                    failures.append((noise, seed, run_dir, proc.returncode))
                    print(f"[failed] return code {proc.returncode}; see {err_path}")
                    print_tail(err_path, args.error_tail)
                    if not args.continue_on_failure:
                        rows.append({
                            "dataset": args.dataset,
                            "dp_noise_multiplier": noise,
                            "seed": seed,
                            "rounds": args.rounds,
                            "out_dir": str(run_dir),
                            "status": status,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        })
                        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(
                                f,
                                fieldnames=[
                                    "dataset",
                                    "dp_noise_multiplier",
                                    "seed",
                                    "rounds",
                                    "out_dir",
                                    "status",
                                    "timestamp",
                                ],
                            )
                            writer.writeheader()
                            writer.writerows(rows)
                        raise SystemExit(1)

            rows.append({
                "dataset": args.dataset,
                "dp_noise_multiplier": noise,
                "seed": seed,
                "rounds": args.rounds,
                "out_dir": str(run_dir),
                "status": status,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "dataset",
                        "dp_noise_multiplier",
                        "seed",
                        "rounds",
                        "out_dir",
                        "status",
                        "timestamp",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

    print("\nSweep complete.")
    print(f"Manifest: {manifest_path}")

    if failures:
        print("\nFailures:")
        for noise, seed, run_dir, code in failures:
            print(f"  noise={noise}, seed={seed}, code={code}, dir={run_dir}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
