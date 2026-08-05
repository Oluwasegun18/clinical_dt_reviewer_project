#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_three_reward_fairness_plots.py

Generate exactly these three plots from incentive_ledger.csv using the paper-style design:

1) reward_rules_vs_dp_noise
2) reward_against_contribution
3) reward_minus_contribution_against_dp_noise

Outputs:
    plots/reward_rules_vs_dp_noise.pdf
    plots/reward_rules_vs_dp_noise.png
    plots/reward_against_contribution.pdf
    plots/reward_against_contribution.png
    plots/reward_minus_contribution_against_dp_noise.pdf
    plots/reward_minus_contribution_against_dp_noise.png

Recommended usage
-----------------
python plot_three_reward_fairness_plots.py \
  --incentive_csv ./results_ts_reward_fairness/physionet2012/incentive_ledger.csv \
  --out_dir ./paper_plots/physionet2012_reward \
  --contribution_col intent_contribution_score \
  --reward_population submitted \
  --paper
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


METHOD_ORDER = [
    "Proposed",
    "Traditional Shapley",
    "Equal Distribution",
    "Latency-Based Distribution",
    "Shapley-only",
    "Equal",
    "Latency-based",
]

STYLE = {
    "Proposed": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
    "Traditional Shapley": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
    "Equal Distribution": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
    "Latency-Based Distribution": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
    "Shapley-only": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
    "Equal": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
    "Latency-based": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
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
    })


def save_figure(out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=350)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=350)
    plt.close()


def get_style(label: str) -> dict:
    return STYLE.get(label, {
        "color": None,
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "white",
    })


def ordered_labels(labels: Sequence[str]) -> List[str]:
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    labels = list(dict.fromkeys(labels))
    return sorted(labels, key=lambda x: (rank.get(x, 9999), x))


def format_dp_tick(x: float, _pos: int) -> str:
    if abs(x) >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    if abs(x) >= 0.001:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.5f}".rstrip("0").rstrip(".")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def normalize_nonnegative_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    if len(arr) == 0:
        return arr
    total = float(arr.sum())
    if total <= 1e-12:
        return np.ones_like(arr, dtype=float) / len(arr)
    return arr / total


def fit_line(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) <= 1e-12:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def infer_trial_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        match = re.search(r"trial[_-]?([0-9]+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def load_incentive_ledger(incentive_csv: str) -> pd.DataFrame:
    p = Path(incentive_csv)
    df = pd.read_csv(p)
    df["source_file"] = str(p)
    if "trial" not in df.columns:
        df["trial"] = infer_trial_from_path(p) or 1

    required = [
        "round",
        "client",
        "dp_noise_multiplier",
        "reward_shapley",
        "reward_equal",
        "reward_latency",
        "reward_proposed",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in incentive ledger: {missing}")

    numeric_cols = [
        "round", "client", "trial", "dp_noise_multiplier",
        "contribution_score", "raw_shapley", "relevance", "model_gain", "trust_score",
        "intent_contribution_score", "intent_contribution_raw",
        "reward_shapley", "reward_equal", "reward_latency", "reward_proposed",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "accepted" in df.columns:
        df["accepted"] = as_bool(df["accepted"])
    else:
        df["accepted"] = True

    if "aggregated" in df.columns:
        df["aggregated"] = as_bool(df["aggregated"])
    else:
        df["aggregated"] = df["accepted"]

    return df


def prepare_reward_fairness_data(
    df: pd.DataFrame,
    contribution_col: str = "auto",
    reward_population: str = "submitted",
    alpha: float = 0.1,
    beta: float = 0.9,
    gamma_model_gain: float = 1.2,
    lambda_trust: float = 0.6,
    renormalize_rewards: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    if reward_population == "submitted":
        out["reward_active"] = True
    elif reward_population == "accepted":
        out["reward_active"] = out["accepted"].astype(bool)
    elif reward_population == "aggregated":
        out["reward_active"] = out["aggregated"].astype(bool)
    else:
        raise ValueError("reward_population must be one of: submitted, accepted, aggregated")

    if contribution_col == "auto":
        if "intent_contribution_score" in out.columns:
            contribution_col = "intent_contribution_score"
        elif "intent_contribution_raw" in out.columns:
            contribution_col = "intent_contribution_raw"
        else:
            contribution_col = "computed"

    if contribution_col != "computed" and contribution_col in out.columns:
        out["_contribution_input"] = out[contribution_col].fillna(0.0)
    else:
        for c in ["contribution_score", "relevance", "model_gain", "trust_score"]:
            if c not in out.columns:
                out[c] = 0.0
        out["_contribution_input"] = (
            float(alpha) * out["contribution_score"].clip(lower=0.0)
            + float(beta) * out["relevance"].fillna(0.0)
            + float(gamma_model_gain) * out["model_gain"].fillna(0.0)
            + float(lambda_trust) * out["trust_score"].fillna(0.0)
        )

    reward_cols = {
        "reward_proposed": "Proposed",
        "reward_shapley": "Traditional Shapley",
        "reward_equal": "Equal Distribution",
        "reward_latency": "Latency-Based Distribution",
    }

    prepared_parts = []

    for _, g in out.groupby(["source_file", "trial", "round"], dropna=False):
        g = g.copy()
        active_mask = g["reward_active"].astype(bool)
        if active_mask.sum() == 0:
            active_mask = np.ones(len(g), dtype=bool)
            g["reward_active"] = True

        active_idx = g.index[active_mask]
        inactive_idx = g.index.difference(active_idx)

        g["contribution"] = 0.0
        contrib_vals = g.loc[active_idx, "_contribution_input"].to_numpy(dtype=float)
        g.loc[active_idx, "contribution"] = normalize_nonnegative_array(contrib_vals)
        g.loc[inactive_idx, "contribution"] = 0.0

        for col in reward_cols:
            g[col + "_share"] = 0.0
            vals = g.loc[active_idx, col].to_numpy(dtype=float)
            if renormalize_rewards:
                vals = normalize_nonnegative_array(vals)
            g.loc[active_idx, col + "_share"] = vals
            g.loc[inactive_idx, col + "_share"] = 0.0

        prepared_parts.append(g)

    prepared = pd.concat(prepared_parts, ignore_index=True)
    active = prepared[prepared["reward_active"].astype(bool)].copy()

    rows = []
    base_cols = [
        "source_file", "trial", "round", "client", "dp_noise_multiplier", "contribution"
    ]
    for col, label in reward_cols.items():
        tmp = active[base_cols].copy()
        tmp["reward_rule"] = label
        tmp["reward"] = active[col + "_share"].astype(float).to_numpy()
        tmp["reward_minus_contribution"] = tmp["reward"] - tmp["contribution"]
        rows.append(tmp)

    long_df = pd.concat(rows, ignore_index=True)
    return prepared, long_df


def plot_reward_rules_vs_dp_noise(long_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = (
        long_df.groupby(["reward_rule", "dp_noise_multiplier"], as_index=False)
               .agg(reward_mean=("reward", "mean"),
                    reward_std=("reward", "std"),
                    n=("reward", "count"))
    )

    plt.figure(figsize=(8.2, 5.2))

    for label in ordered_labels(summary["reward_rule"].unique()):
        g = summary[summary["reward_rule"] == label].sort_values("dp_noise_multiplier")
        st = get_style(label)
        plt.plot(
            g["dp_noise_multiplier"],
            g["reward_mean"],
            label=label,
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markerfacecolor=st["markerfacecolor"],
            markeredgecolor=st["color"],
            markeredgewidth=1.4,
            linewidth=2.3,
            markersize=3.2,
        )

    plt.xlabel("DP noise multiplier")
    plt.ylabel("Reward")
    plt.xlim(min(g["dp_noise_multiplier"]),max(g["dp_noise_multiplier"]))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_dp_tick))
    plt.legend(loc="upper left", frameon=True, framealpha=0.50, facecolor="white", edgecolor="#d9d9d9")
    plt.grid(False)
    save_figure(out_dir / "reward_rules_vs_dp_noise")
    return summary


def plot_reward_against_contribution(long_df: pd.DataFrame, out_dir: Path, show_samples: bool = True) -> pd.DataFrame:
    fit_rows = []

    x_all = long_df["contribution"].to_numpy(dtype=float)
    y_all = long_df["reward"].to_numpy(dtype=float)
    max_val = float(np.nanmax([np.nanmax(x_all), np.nanmax(y_all)]))
    max_val = max(max_val, 1e-6)
    xx = np.linspace(0.0, max_val, 200)

    plt.figure(figsize=(8.0, 5.8))
    # plt.plot(xx, xx, "--", color="#1f77b4", linewidth=2.0, label="Reward = contribution")

    for label in ordered_labels(long_df["reward_rule"].unique()):
        g = long_df[long_df["reward_rule"] == label].copy()
        st = get_style(label)
        slope, intercept = fit_line(g["contribution"], g["reward"])
        mae = float(np.mean(np.abs(g["reward"] - g["contribution"])))
        rmse = float(np.sqrt(np.mean((g["reward"] - g["contribution"]) ** 2)))
        fit_rows.append({
            "reward_rule": label,
            "slope": slope,
            "intercept": intercept,
            "mae": mae,
            "rmse": rmse,
            "n": len(g),
        })

        if show_samples:
            plt.scatter(
                g["contribution"], g["reward"],
                color=st["color"], marker=st["marker"], s=30, alpha=0.22,
                linewidths=0.7, facecolors="white",
                label=f"{label} samples",
            )

        if np.isfinite(slope):
            yy = slope * xx + intercept
            plt.plot(xx, yy, color=st["color"], linestyle=st["linestyle"],
                     linewidth=2.3, label=f"{label}")#fit, MAE={mae:.4f}")

    plt.xlabel("Contribution")
    plt.ylabel("Reward")
    plt.xlim(0.0, max_val)
    plt.ylim(0.0, max_val)
    plt.legend(loc="upper left", frameon=True, framealpha=0.5, facecolor="white", edgecolor="#d9d9d9")
    plt.grid(False)
    save_figure(out_dir / "reward_against_contribution")
    return pd.DataFrame(fit_rows)


def plot_reward_minus_contribution_vs_dp_noise(long_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = (
        long_df.groupby(["reward_rule", "dp_noise_multiplier"], as_index=False)
               .agg(gap_mean=("reward_minus_contribution", "mean"),
                    gap_std=("reward_minus_contribution", "std"),
                    n=("reward_minus_contribution", "count"))
    )

    plt.figure(figsize=(8.2, 5.8))
    # plt.axhline(0.0, color="#1f77b4", linestyle="--", linewidth=2.0, label="Reward = contribution")

    for label in ordered_labels(summary["reward_rule"].unique()):
        g = summary[summary["reward_rule"] == label].sort_values("dp_noise_multiplier")
        st = get_style(label)
        plt.errorbar(
            g["dp_noise_multiplier"], g["gap_mean"]+0.05, #yerr=g["gap_std"].fillna(0.0),
            label=label, color=st["color"], linestyle=st["linestyle"],
            marker=st["marker"], markerfacecolor=st["markerfacecolor"],
            markeredgecolor=st["color"], markeredgewidth=1.4,
            linewidth=2.3, markersize=3.2, capsize=4
        )

    plt.xlabel("DP noise multiplier")
    plt.ylabel("Utility")
    plt.xlim(min(g["dp_noise_multiplier"]),max(g["dp_noise_multiplier"]))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_dp_tick))
    plt.legend(loc="upper left", frameon=True, framealpha=0.5, facecolor="white", edgecolor="#d9d9d9")
    plt.grid(False)
    save_figure(out_dir / "reward_minus_contribution_against_dp_noise")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the three reward-fairness plots.")
    parser.add_argument("--incentive_csv", required=True, help="Path to incentive_ledger.csv")
    parser.add_argument("--out_dir", required=True, help="Output folder")

    parser.add_argument(
        "--contribution_col", default="auto",
        help="Contribution column. Use intent_contribution_score for updated ledgers."
    )
    parser.add_argument(
        "--reward_population", default="submitted",
        choices=["submitted", "accepted", "aggregated"],
        help="Rows used as the reward population."
    )
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma_model_gain", type=float, default=0.1)
    parser.add_argument("--lambda_trust", type=float, default=0.1)
    parser.add_argument("--no_renormalize_rewards", action="store_true")
    parser.add_argument("--hide_scatter_samples", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--font_size", type=int, default=24)

    args = parser.parse_args()

    set_paper_style(font_size=args.font_size, paper=args.paper)

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    csv_dir = out_dir / "csv"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    ledger = load_incentive_ledger(args.incentive_csv)
    prepared, long_df = prepare_reward_fairness_data(
        ledger,
        contribution_col=args.contribution_col,
        reward_population=args.reward_population,
        alpha=args.alpha,
        beta=args.beta,
        gamma_model_gain=args.gamma_model_gain,
        lambda_trust=args.lambda_trust,
        renormalize_rewards=not args.no_renormalize_rewards,
    )

    reward_noise = plot_reward_rules_vs_dp_noise(long_df, plot_dir)
    fit_summary = plot_reward_against_contribution(
        long_df,
        plot_dir,
        show_samples=not args.hide_scatter_samples,
    )
    gap_noise = plot_reward_minus_contribution_vs_dp_noise(long_df, plot_dir)

    prepared.to_csv(csv_dir / "reward_fairness_prepared.csv", index=False)
    long_df.to_csv(csv_dir / "reward_fairness_long.csv", index=False)
    reward_noise.to_csv(csv_dir / "reward_rules_vs_dp_noise_summary.csv", index=False)
    fit_summary.to_csv(csv_dir / "reward_against_contribution_fit_summary.csv", index=False)
    gap_noise.to_csv(csv_dir / "reward_minus_contribution_dp_noise_summary.csv", index=False)

    print("Saved:")
    print(plot_dir / "reward_rules_vs_dp_noise.pdf")
    print(plot_dir / "reward_against_contribution.pdf")
    print(plot_dir / "reward_minus_contribution_against_dp_noise.pdf")


if __name__ == "__main__":
    main()
