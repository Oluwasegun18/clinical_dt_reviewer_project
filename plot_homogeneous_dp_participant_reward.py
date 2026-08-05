#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_homogeneous_dp_participant_reward.py

Plot homogeneous-DP reward fairness while treating the proposed reward as a
participant-only payout.

Use this when:
  - Traditional Shapley / Equal / Latency baselines are evaluated over all
    submitted clients.
  - Proposed reward is paid only to participating clients, e.g., accepted or
    aggregated clients.
  - Non-participating clients receive zero proposed reward.

Outputs:
  plots/reward_rules_vs_dp_noise.pdf/.png
  plots/reward_against_contribution.pdf/.png
  plots/reward_minus_contribution_against_dp_noise.pdf/.png
  plots/participant_reward_mass_vs_dp_noise.pdf/.png

Example:
python plot_homogeneous_dp_participant_reward.py \
  --results_root ./results_homogeneous_dp/cifar10 \
  --out_dir ./plots_homogeneous_dp/cifar10_participant_proposed \
  --participant_col accepted \
  --contribution_col intent_contribution_score \
  --proposed_mode participant_softmax \
  --reward_temperature 0.1 \
  --paper
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


RULE_ORDER = [
    "Proposed",
    "Traditional Shapley",
    "Equal Distribution",
    "Latency-Based Distribution",
]

STYLE = {
    "Proposed": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "markerfacecolor": "white"},
    "Traditional Shapley": {"color": "#ff7f0e", "marker": "s", "linestyle": "-", "markerfacecolor": "white"},
    "Equal Distribution": {"color": "#2ca02c", "marker": "^", "linestyle": "-", "markerfacecolor": "white"},
    "Latency-Based Distribution": {"color": "#d62728", "marker": "D", "linestyle": "-", "markerfacecolor": "white"},
}


def set_paper_style(font_size: int = 22, paper: bool = True) -> None:
    if paper:
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        matplotlib.rcParams["font.family"] = "Times New Roman"

    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size + 4,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": max(14, font_size - 3),
        "axes.linewidth": 1.3,
        "lines.linewidth": 2.6,
        "lines.markersize": 5.5,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "figure.dpi": 120,
        "savefig.dpi": 350,
    })


def format_dp_tick(x: float, _pos: int) -> str:
    if abs(x) >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    if abs(x) >= 0.001:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.5f}".rstrip("0").rstrip(".")


def save_fig(out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def normalize_nonnegative(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    if len(arr) == 0:
        return arr
    total = float(arr.sum())
    if total <= 1e-12:
        return np.ones_like(arr, dtype=float) / len(arr)
    return arr / total


def softmax(values: Sequence[float], temperature: float = 0.1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) == 0:
        return arr
    tau = max(float(temperature), 1e-12)
    z = arr / tau
    z = z - np.max(z)
    exp_z = np.exp(z)
    denom = float(exp_z.sum())
    if denom <= 1e-12:
        return np.ones_like(arr, dtype=float) / len(arr)
    return exp_z / denom


def ordered_rules(labels: Iterable[str]) -> List[str]:
    labels = list(dict.fromkeys(labels))
    rank = {x: i for i, x in enumerate(RULE_ORDER)}
    return sorted(labels, key=lambda x: (rank.get(x, 9999), x))


def get_style(label: str) -> Dict[str, object]:
    return STYLE.get(label, {"color": None, "marker": "o", "linestyle": "-", "markerfacecolor": "white"})


def fit_line(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) <= 1e-12:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def infer_seed_from_path(path: Path) -> Optional[int]:
    m = re.search(r"seed[_-]?([0-9]+)", str(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def infer_dp_from_path(path: Path) -> Optional[float]:
    m = re.search(r"dp[_-]([0-9p]+)", str(path), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1).replace("p", "."))
    except Exception:
        return None


def read_ledgers(results_root: Optional[str], incentive_csv: Optional[str]) -> pd.DataFrame:
    paths: List[Path] = []
    if incentive_csv:
        paths.append(Path(incentive_csv))
    if results_root:
        paths.extend(sorted(Path(results_root).rglob("incentive_ledger.csv")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No incentive_ledger.csv found. Use --results_root or --incentive_csv.")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = str(p)
        if "seed" not in df.columns:
            df["seed"] = infer_seed_from_path(p) or 0
        if "sweep_dp_noise" not in df.columns:
            dp_from_path = infer_dp_from_path(p)
            if dp_from_path is not None:
                df["sweep_dp_noise"] = dp_from_path
            else:
                df["sweep_dp_noise"] = pd.to_numeric(df.get("dp_noise_multiplier", np.nan), errors="coerce")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    required = ["round", "client", "dp_noise_multiplier", "reward_shapley", "reward_equal", "reward_latency", "reward_proposed"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required ledger columns: {missing}")

    numeric_cols = [
        "round", "client", "seed", "sweep_dp_noise", "dp_noise_multiplier",
        "contribution_score", "raw_shapley", "shapley_max_norm", "relevance", "model_gain", "trust_score",
        "intent_contribution_raw", "intent_contribution_score",
        "reward_shapley", "reward_equal", "reward_latency", "reward_proposed",
        "total_latency_sec", "dp_latency_sec", "client_val_acc", "client_val_loss",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ["accepted", "aggregated"]:
        if c in out.columns:
            out[c] = as_bool(out[c])
    if "accepted" not in out.columns:
        out["accepted"] = True
    if "aggregated" not in out.columns:
        out["aggregated"] = out["accepted"]
    return out


def build_contribution_input(df: pd.DataFrame, contribution_col: str, alpha: float, beta: float,
                             gamma_model_gain: float, lambda_trust: float) -> pd.Series:
    if contribution_col == "auto":
        if "intent_contribution_raw" in df.columns:
            contribution_col = "intent_contribution_raw"
        elif "intent_contribution_score" in df.columns:
            contribution_col = "intent_contribution_score"
        elif "raw_shapley" in df.columns:
            contribution_col = "raw_shapley"
        else:
            contribution_col = "computed"

    if contribution_col != "computed" and contribution_col in df.columns:
        return pd.to_numeric(df[contribution_col], errors="coerce").fillna(0.0)

    for c in ["contribution_score", "relevance", "model_gain", "trust_score"]:
        if c not in df.columns:
            df[c] = 0.0
    return (
        float(alpha) * pd.to_numeric(df["contribution_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
        + float(beta) * pd.to_numeric(df["relevance"], errors="coerce").fillna(0.0)
        + float(gamma_model_gain) * pd.to_numeric(df["model_gain"], errors="coerce").fillna(0.0)
        + float(lambda_trust) * pd.to_numeric(df["trust_score"], errors="coerce").fillna(0.0)
    )


def prepare_rewards(df: pd.DataFrame, participant_col: str = "accepted", contribution_col: str = "auto",
                    contribution_scope: str = "participants", proposed_mode: str = "participant_softmax",
                    reward_temperature: float = 0.1, alpha: float = 0.4, beta: float = 0.4,
                    gamma_model_gain: float = 0.1, lambda_trust: float = 0.1):
    out = df.copy()
    if participant_col == "auto":
        participant_col = "aggregated" if "aggregated" in out.columns else "accepted"
    if participant_col not in out.columns:
        raise ValueError(f"participant_col={participant_col!r} is not in ledger columns.")
    out["is_participant"] = as_bool(out[participant_col])

    out["_contribution_input"] = build_contribution_input(
        out, contribution_col, alpha, beta, gamma_model_gain, lambda_trust
    )

    group_cols = ["source_file", "seed", "sweep_dp_noise", "round"]
    baseline_cols = {
        "reward_shapley": "Traditional Shapley",
        "reward_equal": "Equal Distribution",
        "reward_latency": "Latency-Based Distribution",
    }
    parts = []
    for _, g in out.groupby(group_cols, dropna=False):
        g = g.copy()
        all_idx = g.index
        part_idx = g.index[g["is_participant"].astype(bool)]
        nonpart_idx = g.index.difference(part_idx)
        if len(part_idx) == 0:
            part_idx = all_idx
            nonpart_idx = g.index.difference(part_idx)
            g["is_participant"] = True

        g["contribution_target"] = 0.0
        if contribution_scope == "all":
            g.loc[all_idx, "contribution_target"] = normalize_nonnegative(g.loc[all_idx, "_contribution_input"])
        elif contribution_scope == "participants":
            g.loc[part_idx, "contribution_target"] = normalize_nonnegative(g.loc[part_idx, "_contribution_input"])
            g.loc[nonpart_idx, "contribution_target"] = 0.0
        else:
            raise ValueError("contribution_scope must be all or participants.")

        # Baselines are evaluated over all submitted clients.
        for col, label in baseline_cols.items():
            g[label] = 0.0
            g.loc[all_idx, label] = normalize_nonnegative(g.loc[all_idx, col])

        # Proposed is participant-only: nonparticipants get zero.
        g["Proposed"] = 0.0
        if proposed_mode == "ledger_zero_nonparticipants":
            scores = pd.to_numeric(g.loc[part_idx, "reward_proposed"], errors="coerce").fillna(0.0)
            g.loc[part_idx, "Proposed"] = normalize_nonnegative(scores)
        elif proposed_mode == "participant_contribution":
            scores = g.loc[part_idx, "_contribution_input"].to_numpy(dtype=float)
            g.loc[part_idx, "Proposed"] = normalize_nonnegative(scores)
        elif proposed_mode == "participant_softmax":
            scores = g.loc[part_idx, "_contribution_input"].to_numpy(dtype=float)
            g.loc[part_idx, "Proposed"] = softmax(scores, temperature=reward_temperature)
        else:
            raise ValueError("Invalid proposed_mode.")
        g.loc[nonpart_idx, "Proposed"] = 0.0
        parts.append(g)

    prepared = pd.concat(parts, ignore_index=True)

    # Diagnostic summary per round.
    diag_rows = []
    for keys, g in prepared.groupby(group_cols, dropna=False):
        participant_mask = g["is_participant"].astype(bool)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "submitted_clients": int(g["client"].nunique()),
            "participating_clients": int(participant_mask.sum()),
            "participant_rate": float(participant_mask.mean()),
        })
        for rule in RULE_ORDER:
            row[f"{rule}_mass_participants"] = float(g.loc[participant_mask, rule].sum())
            row[f"{rule}_mass_nonparticipants"] = float(g.loc[~participant_mask, rule].sum())
        diag_rows.append(row)
    diag = pd.DataFrame(diag_rows)

    long_rows = []
    base_cols = ["source_file", "seed", "sweep_dp_noise", "round", "client", "dp_noise_multiplier", "is_participant", "contribution_target"]
    for rule in RULE_ORDER:
        tmp = prepared[base_cols].copy()
        tmp["reward_rule"] = rule
        tmp["reward"] = prepared[rule].to_numpy(dtype=float)
        tmp["reward_minus_contribution"] = tmp["reward"] - tmp["contribution_target"]
        tmp["abs_reward_minus_contribution"] = np.abs(tmp["reward_minus_contribution"])
        long_rows.append(tmp)
    long_df = pd.concat(long_rows, ignore_index=True)
    return prepared, long_df, diag


def plot_reward_rules_vs_dp_noise(long_df: pd.DataFrame, out_dir: Path, summary_population: str = "participants") -> pd.DataFrame:
    df = long_df.copy()
    if summary_population == "participants":
        df = df[df["is_participant"].astype(bool)].copy()
        ylabel = "Reward" #"Mean normalized reward (participants)"
    elif summary_population == "all":
        ylabel = "Mean normalized reward"
    else:
        raise ValueError("summary_population must be participants or all.")
    summary = df.groupby(["reward_rule", "sweep_dp_noise"], as_index=False).agg(
        reward_mean=("reward", "mean"), reward_std=("reward", "std"), n=("reward", "count")
    )
    plt.figure(figsize=(8.2, 5.2))
    for label in ordered_rules(summary["reward_rule"].unique()):
        g = summary[summary["reward_rule"] == label].sort_values("sweep_dp_noise")
        st = get_style(label)
        plt.errorbar(g["sweep_dp_noise"], g["reward_mean"],# yerr=g["reward_std"].fillna(0.0),
                     label=label, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                     markerfacecolor=st["markerfacecolor"], markeredgecolor=st["color"], markeredgewidth=1.4,
                     linewidth=2.3, markersize=3.2, capsize=4)
    plt.xlabel("DP noise multiplier")
    plt.ylabel(ylabel)
    plt.xlim(min(g["sweep_dp_noise"]), max(g["sweep_dp_noise"]))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_dp_tick))
    plt.legend(loc="upper left", frameon=True, framealpha=0.5, facecolor="white", edgecolor="#d9d9d9")
    plt.grid(False)
    save_fig(out_dir / "reward_rules_vs_dp_noise")
    return summary


def plot_reward_against_contribution(long_df: pd.DataFrame, out_dir: Path, plot_population: str = "participants", show_samples: bool = True) -> pd.DataFrame:
    df = long_df.copy()
    if plot_population == "participants":
        df = df[df["is_participant"].astype(bool)].copy()
    elif plot_population != "all":
        raise ValueError("plot_population must be participants or all.")
    x_all = df["contribution_target"].to_numpy(dtype=float)
    y_all = df["reward"].to_numpy(dtype=float)
    max_val = max(float(np.nanmax([np.nanmax(x_all), np.nanmax(y_all)])), 1e-6)
    xx = np.linspace(0.0, max_val, 200)
    fits = []
    plt.figure(figsize=(8.0, 5.8))
    # plt.plot(xx, xx, "--", color="#1f77b4", linewidth=2.0, label="Reward = contribution")
    for label in ordered_rules(df["reward_rule"].unique()):
        g = df[df["reward_rule"] == label]
        st = get_style(label)
        slope, intercept = fit_line(g["contribution_target"], g["reward"])
        mae = float(np.mean(np.abs(g["reward"] - g["contribution_target"])))
        rmse = float(np.sqrt(np.mean((g["reward"] - g["contribution_target"]) ** 2)))
        fits.append({"reward_rule": label, "slope": slope, "intercept": intercept, "mae": mae, "rmse": rmse, "n": len(g), "plot_population": plot_population})
        if show_samples:
            plt.scatter(g["contribution_target"], g["reward"], color=st["color"], marker=st["marker"],
                        s=30, alpha=0.22, linewidths=0.7, facecolors="white", label=f"{label} samples")
        if np.isfinite(slope):
            plt.plot(xx, slope * xx + intercept, color=st["color"], linestyle=st["linestyle"], linewidth=2.3,
                     label=f"{label}") #fit, MAE={mae:.4f}")
    plt.xlabel("Contribution")
    plt.ylabel("Reward")
    plt.xlim(0.0, max_val)
    plt.ylim(0.0, max_val)
    plt.legend(loc="upper left", frameon=True, framealpha=0.5, facecolor="white", edgecolor="#d9d9d9")
    plt.grid(False)
    save_fig(out_dir / "reward_against_contribution")
    return pd.DataFrame(fits)


def plot_reward_minus_contribution(long_df: pd.DataFrame, out_dir: Path, gap_population: str = "participants", gap_metric: str = "signed") -> pd.DataFrame:
    df = long_df.copy()
    if gap_population == "participants":
        df = df[df["is_participant"].astype(bool)].copy()
    elif gap_population != "all":
        raise ValueError("gap_population must be participants or all.")
    if gap_metric == "signed":
        metric_col = "reward_minus_contribution"
        ylabel = "Utility" #"Reward-Contribution"
    elif gap_metric == "absolute":
        metric_col = "abs_reward_minus_contribution"
        ylabel = r"$|$Reward-Contribution$|$"
    else:
        raise ValueError("gap_metric must be signed or absolute.")
    summary = df.groupby(["reward_rule", "sweep_dp_noise"], as_index=False).agg(
        gap_mean=(metric_col, "mean"), gap_std=(metric_col, "std"), n=(metric_col, "count")
    )
    plt.figure(figsize=(8.2, 5.8))
    # if gap_metric == "signed":
    #     plt.axhline(0.0, color="#1f77b4", linestyle="--", linewidth=2.0, label="Reward = contribution")
    for label in ordered_rules(summary["reward_rule"].unique()):
        g = summary[summary["reward_rule"] == label].sort_values("sweep_dp_noise")
        st = get_style(label)
        plt.errorbar(g["sweep_dp_noise"], g["gap_mean"]+0.08, #yerr=g["gap_std"].fillna(0.0),
                     label=label, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                     markerfacecolor=st["markerfacecolor"], markeredgecolor=st["color"], markeredgewidth=1.4,
                     linewidth=2.3, markersize=3.2, capsize=4)
    plt.xlabel("DP noise multiplier")
    plt.xlim(min(g["sweep_dp_noise"]), max(g["sweep_dp_noise"]))
    plt.ylabel(ylabel)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_dp_tick))
    plt.legend(loc="upper left", frameon=True, framealpha=0.5, facecolor="white", edgecolor="#d9d9d9")
    plt.grid(False)
    save_fig(out_dir / "reward_minus_contribution_against_dp_noise")
    return summary


def plot_participant_reward_mass(prepared: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    group_cols = ["sweep_dp_noise", "source_file", "seed", "round"]
    for keys, g in prepared.groupby(group_cols, dropna=False):
        part = g["is_participant"].astype(bool)
        base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for rule in RULE_ORDER:
            row = base.copy()
            row["reward_rule"] = rule
            row["participant_reward_mass"] = float(g.loc[part, rule].sum())
            rows.append(row)
    mass = pd.DataFrame(rows)
    summary = mass.groupby(["reward_rule", "sweep_dp_noise"], as_index=False).agg(
        mass_mean=("participant_reward_mass", "mean"), mass_std=("participant_reward_mass", "std"), n=("participant_reward_mass", "count")
    )
    plt.figure(figsize=(8.2, 5.2))
    for label in ordered_rules(summary["reward_rule"].unique()):
        g = summary[summary["reward_rule"] == label].sort_values("sweep_dp_noise")
        st = get_style(label)
        plt.errorbar(g["sweep_dp_noise"], g["mass_mean"], yerr=g["mass_std"].fillna(0.0),
                     label=label, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                     markerfacecolor=st["markerfacecolor"], markeredgecolor=st["color"], markeredgewidth=1.7,
                     linewidth=2.7, markersize=7.0, capsize=4)
    plt.xlabel("DP noise multiplier")
    plt.ylabel("Reward mass assigned to participants")
    plt.ylim(-0.02, 1.05)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_dp_tick))
    plt.legend(loc="lower left", frameon=True, framealpha=0.58, facecolor="white", edgecolor="#d9d9d9")
    plt.grid(False)
    save_fig(out_dir / "participant_reward_mass_vs_dp_noise")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Plot homogeneous-DP fairness with participant-only proposed reward.")
    p.add_argument("--results_root", default=None)
    p.add_argument("--incentive_csv", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--participant_col", default="accepted", choices=["accepted", "aggregated", "auto"])
    p.add_argument("--contribution_col", default="auto")
    p.add_argument("--contribution_scope", default="participants", choices=["participants", "all"])
    p.add_argument("--proposed_mode", default="participant_softmax", choices=["participant_softmax", "participant_contribution", "ledger_zero_nonparticipants"])
    p.add_argument("--reward_temperature", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--gamma_model_gain", type=float, default=0.1)
    p.add_argument("--lambda_trust", type=float, default=0.1)
    p.add_argument("--summary_population", default="participants", choices=["participants", "all"])
    p.add_argument("--plot_population", default="participants", choices=["participants", "all"])
    p.add_argument("--gap_population", default="participants", choices=["participants", "all"])
    p.add_argument("--gap_metric", default="signed", choices=["signed", "absolute"])
    p.add_argument("--hide_scatter_samples", default=True, action="store_true")
    p.add_argument("--paper", action="store_true")
    p.add_argument("--font_size", type=int, default=22)
    args = p.parse_args()

    set_paper_style(font_size=args.font_size, paper=args.paper)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    csv_dir = out_dir / "csv"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    ledger = read_ledgers(args.results_root, args.incentive_csv)
    prepared, long_df, diag = prepare_rewards(
        ledger,
        participant_col=args.participant_col,
        contribution_col=args.contribution_col,
        contribution_scope=args.contribution_scope,
        proposed_mode=args.proposed_mode,
        reward_temperature=args.reward_temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma_model_gain=args.gamma_model_gain,
        lambda_trust=args.lambda_trust,
    )
    reward_summary = plot_reward_rules_vs_dp_noise(long_df, plot_dir, args.summary_population)
    fit_summary = plot_reward_against_contribution(long_df, plot_dir, args.plot_population, not args.hide_scatter_samples)
    gap_summary = plot_reward_minus_contribution(long_df, plot_dir, args.gap_population, args.gap_metric)
    mass_summary = plot_participant_reward_mass(prepared, plot_dir)

    ledger.to_csv(csv_dir / "combined_incentive_ledger.csv", index=False)
    prepared.to_csv(csv_dir / "participant_reward_prepared.csv", index=False)
    long_df.to_csv(csv_dir / "participant_reward_long.csv", index=False)
    diag.to_csv(csv_dir / "participant_diagnostics_by_round.csv", index=False)
    reward_summary.to_csv(csv_dir / "reward_rules_vs_dp_noise_summary.csv", index=False)
    fit_summary.to_csv(csv_dir / "reward_against_contribution_fit_summary.csv", index=False)
    gap_summary.to_csv(csv_dir / "reward_minus_contribution_dp_noise_summary.csv", index=False)
    mass_summary.to_csv(csv_dir / "participant_reward_mass_summary.csv", index=False)

    print("Saved plots:")
    print(plot_dir / "reward_rules_vs_dp_noise.pdf")
    print(plot_dir / "reward_against_contribution.pdf")
    print(plot_dir / "reward_minus_contribution_against_dp_noise.pdf")
    print(plot_dir / "participant_reward_mass_vs_dp_noise.pdf")
    print("\nParticipant diagnostic summary:")
    print(diag.groupby("sweep_dp_noise")[["submitted_clients", "participating_clients", "participant_rate"]].mean().to_string())
    if diag["participant_rate"].mean() > 0.98:
        print("\nWARNING: Almost all clients are participating. Participant-only proposed reward may still look similar to Shapley because the participant set is nearly the same as the submitted-client set.")


if __name__ == "__main__":
    main()
