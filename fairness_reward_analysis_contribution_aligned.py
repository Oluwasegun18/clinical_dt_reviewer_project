#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fairness_reward_analysis_contribution_aligned.py

Compute and plot reward/fairness metrics from incentive_ledger.csv.

Key update:
- The main fairness curve is no longer a standard reward-equality Lorenz curve.
- It is now a contribution-aligned reward fairness curve:

      x-axis = cumulative share of validated contribution
      y-axis = cumulative share of reward

  The diagonal y=x means perfect contribution-proportional reward allocation.

Expected columns:
round,method,ablation,client,accepted,is_malicious,attack_type,malicious_ratio,
dp_noise_multiplier,contribution_score,raw_shapley,relevance,model_gain,
reward_shapley,reward_equal,reward_latency,reward_proposed,
compute_time_sec,comm_latency_sec,total_latency_sec,trial

Examples:

# Use raw normalized KS/Shapley contribution_score as the contribution target
python fairness_reward_analysis_contribution_aligned.py \
  --incentive_csv ./results_ts/physionet2012_test/incentive_ledger.csv \
  --out_dir ./fairness_results/physionet2012_test \
  --contribution_col contribution_score \
  --paper --window 5

# Recommended for your current setting alpha=0.1, beta=0.9:
# Use validated contribution = alpha * KS contribution + beta * clinical relevance
python fairness_reward_analysis_contribution_aligned.py \
  --incentive_csv ./results_ts/physionet2012_test/incentive_ledger.csv \
  --out_dir ./fairness_results/physionet2012_test \
  --use_validated_contribution \
  --alpha 0.1 --beta 0.9 \
  --paper --window 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


REWARD_RULES = {
    "reward_shapley": "Traditional Shapley",
    "reward_equal": "Equal Distribution",
    "reward_latency": "Latency-Based Distribution",
    "reward_proposed": "Proposed",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
LINESTYLES = ["-", "-", "-", "-", (0, (3, 1, 1, 1)), (0, (5, 2))]

NUMERIC_COLS = [
    "round", "client", "malicious_ratio", "dp_noise_multiplier",
    "contribution_score", "raw_shapley", "relevance", "model_gain",
    "reward_shapley", "reward_equal", "reward_latency", "reward_proposed",
    "compute_time_sec", "comm_latency_sec", "total_latency_sec", "trial",
]


# ---------------------------------------------------------------------
# Style and utilities
# ---------------------------------------------------------------------
def set_plot_style(paper: bool = False, font_size: int = 15) -> None:
    if paper:
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": max(10, font_size - 2),
        "ytick.labelsize": max(10, font_size - 2),
        "legend.fontsize": max(9, font_size - 2),
        "lines.linewidth": 2.6,
        "patch.force_edgecolor": True,
        "patch.facecolor": "none",
    })


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def normalize_nonnegative(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0:
        return np.ones_like(arr, dtype=float) / len(arr) if len(arr) > 0 else arr
    return arr / total


def gini(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    arr = np.maximum(arr, 0.0)
    if arr.sum() <= 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * arr) / (n * arr.sum())) - ((n + 1) / n))


def normalized_entropy(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    arr = np.maximum(arr, 0.0)
    total = arr.sum()
    if total <= 0:
        return np.nan
    p = arr / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(arr)))


def safe_corr(x: Sequence[float], y: Sequence[float], kind: str = "pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    if SCIPY_AVAILABLE:
        if kind == "spearman":
            return float(spearmanr(x, y).correlation)
        return float(pearsonr(x, y)[0])
    if kind == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy()
        y = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(x, y)[0, 1])


def smooth_series(
    y: pd.Series,
    mode: str = "rolling",
    window: int = 5,
    ema_alpha: float = 0.25,
) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    if mode == "none":
        return y
    if mode == "ema":
        return y.ewm(alpha=ema_alpha, adjust=False, min_periods=1).mean()
    if mode == "rolling_center":
        return y.rolling(window=window, min_periods=1, center=True).mean()
    return y.rolling(window=window, min_periods=1).mean()


def savefig(path_base: Path, dpi: int = 350) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight", dpi=dpi)
    plt.savefig(path_base.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close()


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
def load_ledger(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "round", "method", "client", "accepted", "is_malicious", "contribution_score",
        "reward_shapley", "reward_equal", "reward_latency", "reward_proposed"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in incentive ledger: {missing}")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["accepted"] = as_bool(df["accepted"])
    df["is_malicious"] = as_bool(df["is_malicious"])

    if "trial" not in df.columns:
        df["trial"] = 1
    if "ablation" not in df.columns:
        df["ablation"] = "none"
    if "attack_type" not in df.columns:
        df["attack_type"] = "none"
    if "malicious_ratio" not in df.columns:
        df["malicious_ratio"] = 0.0
    if "dp_noise_multiplier" not in df.columns:
        df["dp_noise_multiplier"] = np.nan

    return df


def add_validated_contribution_score(
    df: pd.DataFrame,
    alpha: float = 0.1,
    beta: float = 0.9,
    contribution_col: str = "contribution_score",
    relevance_col: str = "relevance",
) -> pd.DataFrame:
    """
    Adds a clinical-quality-weighted contribution score:

        validated_contribution_score = alpha * normalized(KS/Shapley contribution)
                                     + beta  * normalized(clinical relevance)

    The final value is normalized again inside each round/trial/method setting.

    This is the recommended contribution target when reward allocation is tilted
    toward clinical relevance, e.g., alpha=0.1 and beta=0.9.
    """
    if contribution_col not in df.columns:
        raise ValueError(f"{contribution_col!r} not found in ledger.")
    if relevance_col not in df.columns:
        raise ValueError(
            f"{relevance_col!r} not found in ledger. "
            "Cannot compute validated contribution without clinical relevance."
        )

    out = df.copy()
    group_cols = ["trial", "method", "ablation", "attack_type", "malicious_ratio", "round"]
    group_cols = [c for c in group_cols if c in out.columns]

    out["contribution_score_norm"] = np.nan
    out["relevance_norm"] = np.nan
    out["validated_contribution_score"] = np.nan

    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        phi = normalize_nonnegative(out.loc[idx, contribution_col].to_numpy())
        rel = normalize_nonnegative(out.loc[idx, relevance_col].to_numpy())

        validated = alpha * phi + beta * rel
        validated = normalize_nonnegative(validated)

        out.loc[idx, "contribution_score_norm"] = phi
        out.loc[idx, "relevance_norm"] = rel
        out.loc[idx, "validated_contribution_score"] = validated

    return out


def final_round_df(df: pd.DataFrame, final_round: Optional[int]) -> pd.DataFrame:
    r = int(df["round"].max()) if final_round is None else int(final_round)
    out = df[df["round"] == r].copy().sort_values("client")
    if len(out) == 0:
        raise ValueError(f"No rows found for final_round={r}")
    return out


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_reward_metrics(df: pd.DataFrame, contribution_col: str) -> pd.DataFrame:
    keys = ["trial", "method", "ablation", "attack_type", "malicious_ratio", "round"]
    keys = [k for k in keys if k in df.columns]
    rows = []

    for key_vals, g in df.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        base = dict(zip(keys, key_vals))

        contrib = normalize_nonnegative(g[contribution_col])
        malicious = g["is_malicious"].to_numpy(dtype=bool)
        accepted = g["accepted"].to_numpy(dtype=bool)
        dp = g["dp_noise_multiplier"].to_numpy(dtype=float)

        for reward_col, reward_name in REWARD_RULES.items():
            reward = normalize_nonnegative(g[reward_col])
            row = dict(base)
            row.update({
                "reward_rule": reward_name,
                "reward_col": reward_col,
                "contribution_target": contribution_col,
                "pearson": safe_corr(contrib, reward, "pearson"),
                "spearman": safe_corr(contrib, reward, "spearman"),
                "l1_gap": float(np.mean(np.abs(reward - contrib))),
                "l2_gap": float(np.sqrt(np.mean((reward - contrib) ** 2))),
                "max_gap": float(np.max(np.abs(reward - contrib))),
                "reward_gini": gini(reward),
                "reward_entropy": normalized_entropy(reward),
                "contribution_gini": gini(contrib),
                "acceptance_rate": float(np.mean(accepted)),
                "malicious_reward_share": float(reward[malicious].sum()) if malicious.any() else 0.0,
                "honest_reward_share": float(reward[~malicious].sum()) if (~malicious).any() else np.nan,
                "malicious_contribution_share": float(contrib[malicious].sum()) if malicious.any() else 0.0,
                "honest_contribution_share": float(contrib[~malicious].sum()) if (~malicious).any() else np.nan,
                "malicious_acceptance_rate": float(np.mean(accepted[malicious])) if malicious.any() else np.nan,
                "malicious_rejection_rate": float(1.0 - np.mean(accepted[malicious])) if malicious.any() else np.nan,
                "honest_acceptance_rate": float(np.mean(accepted[~malicious])) if (~malicious).any() else np.nan,
                "dp_noise_reward_pearson": safe_corr(dp, reward, "pearson"),
                "dp_noise_contribution_pearson": safe_corr(dp, contrib, "pearson"),
            })

            finite_dp = dp[np.isfinite(dp)]
            if len(np.unique(finite_dp)) >= 2:
                low, high = np.nanmin(dp), np.nanmax(dp)
                low_mask, high_mask = np.isclose(dp, low), np.isclose(dp, high)
                row["dp_reward_gap_high_minus_low"] = float(
                    np.nanmean(reward[high_mask]) - np.nanmean(reward[low_mask])
                )
                row["dp_contribution_gap_high_minus_low"] = float(
                    np.nanmean(contrib[high_mask]) - np.nanmean(contrib[low_mask])
                )
                row["dp_acceptance_gap_high_minus_low"] = float(
                    np.nanmean(accepted[high_mask].astype(float))
                    - np.nanmean(accepted[low_mask].astype(float))
                )
            else:
                row["dp_reward_gap_high_minus_low"] = np.nan
                row["dp_contribution_gap_high_minus_low"] = np.nan
                row["dp_acceptance_gap_high_minus_low"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def compute_client_summary(df: pd.DataFrame, contribution_col: str) -> pd.DataFrame:
    keys = ["method", "ablation", "attack_type", "malicious_ratio", "trial", "client"]
    keys = [k for k in keys if k in df.columns]

    agg_spec = {
        "rounds": ("round", "nunique"),
        "acceptance_rate": ("accepted", "mean"),
        "is_malicious": ("is_malicious", "max"),
        "mean_contribution": (contribution_col, "mean"),
        "mean_reward_proposed": ("reward_proposed", "mean"),
        "total_reward_proposed": ("reward_proposed", "sum"),
        "mean_reward_shapley": ("reward_shapley", "mean"),
        "mean_reward_equal": ("reward_equal", "mean"),
        "mean_reward_latency": ("reward_latency", "mean"),
        "mean_dp_noise": ("dp_noise_multiplier", "mean"),
    }
    if "relevance" in df.columns:
        agg_spec["mean_relevance"] = ("relevance", "mean")
    if "model_gain" in df.columns:
        agg_spec["mean_model_gain"] = ("model_gain", "mean")
    if "validated_contribution_score" in df.columns:
        agg_spec["mean_validated_contribution"] = ("validated_contribution_score", "mean")

    out = df.groupby(keys, dropna=False).agg(**agg_spec).reset_index()
    return out


def summarize_final(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["trial", "method", "ablation", "attack_type", "malicious_ratio", "reward_rule"]
    keys = [k for k in keys if k in metrics.columns]

    final_parts = []
    for _, g in metrics.groupby(keys, dropna=False):
        final_parts.append(g[g["round"] == g["round"].max()])
    final = pd.concat(final_parts, ignore_index=True)

    avg_keys = ["method", "ablation", "attack_type", "malicious_ratio", "reward_rule", "contribution_target"]
    avg_keys = [k for k in avg_keys if k in final.columns]
    num_cols = final.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["round", "trial"]]

    summary = final.groupby(avg_keys, dropna=False)[num_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(x) for x in c if str(x) != ""]).rstrip("_")
        if isinstance(c, tuple) else c
        for c in summary.columns
    ]
    return summary


# ---------------------------------------------------------------------
# Fairness curves
# ---------------------------------------------------------------------
def lorenz_points(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard Lorenz curve for inequality only.

    x-axis: cumulative share of clients
    y-axis: cumulative share of values

    This should be interpreted as reward inequality, not contribution fairness.
    """
    y = normalize_nonnegative(values)
    y = np.sort(y)
    cum = np.cumsum(y)
    x = np.insert(np.arange(1, len(y) + 1) / len(y), 0, 0.0)
    y = np.insert(cum, 0, 0.0)
    return x, y


def contribution_reward_parity_points(
    reward_values: Sequence[float],
    contribution_values: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Contribution-aligned reward fairness curve.

    x-axis: cumulative share of contribution
    y-axis: cumulative share of reward

    The diagonal y=x means perfect contribution-proportional reward.
    """
    reward = normalize_nonnegative(reward_values)
    contribution = normalize_nonnegative(contribution_values)

    # Sort clients from low to high contribution, and accumulate contribution/reward.
    order = np.argsort(contribution)
    contribution_sorted = contribution[order]
    reward_sorted = reward[order]

    x = np.cumsum(contribution_sorted)
    y = np.cumsum(reward_sorted)

    x = np.insert(x, 0, 0.0)
    y = np.insert(y, 0, 0.0)

    return x, y


def plot_contribution_aligned_fairness(
    final_df: pd.DataFrame,
    contribution_col: str,
    out_dir: Path,
) -> None:
    """
    Main fairness plot for the paper.

    This replaces the old standard Lorenz interpretation.
    """
    plt.figure(figsize=(7.4, 5.7))

    plt.plot(
        [0, 1], [0, 1], "--",
        linewidth=2.2,
        label="Perfect contribution-proportional reward",
    )

    for i, (reward_col, label) in enumerate(REWARD_RULES.items()):
        x, y = contribution_reward_parity_points(
            final_df[reward_col],
            final_df[contribution_col],
        )
        plt.plot(
            x, y,
            label=label,
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.4,
            linewidth=2.5,
            markevery=max(1, len(x) // 6),
        )

    plt.xlabel("Cumulative share of validated contribution")
    plt.ylabel("Cumulative share of reward")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "contribution_aligned_reward_fairness")


def plot_reward_inequality_lorenz(final_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Optional standard Lorenz curve.

    This measures reward inequality only, not contribution-proportional fairness.
    """
    plt.figure(figsize=(7.4, 5.7))
    plt.plot([0, 1], [0, 1], "--", linewidth=2.2, label="Equal reward distribution")

    for i, (col, label) in enumerate(REWARD_RULES.items()):
        x, y = lorenz_points(final_df[col])
        plt.plot(
            x, y,
            label=label,
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.4,
            linewidth=2.5,
            markevery=max(1, len(x) // 6),
        )

    plt.xlabel("Cumulative share of clients")
    plt.ylabel("Cumulative share of reward")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "reward_inequality_lorenz")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_client_balance(
    final_df: pd.DataFrame,
    contribution_col: str,
    out_dir: Path,
    contribution_label: str = "Contribution",
) -> None:
    clients = final_df["client"].astype(int).to_numpy()
    x = np.arange(len(clients))
    width = 0.15
    bars = [
        ("reward_shapley", "Traditional Shapley"),
        ("reward_equal", "Equal Distribution"),
        ("reward_latency", "Latency-Based Distribution"),
        ("reward_proposed", "Proposed"),
        (contribution_col, contribution_label),
    ]

    plt.figure(figsize=(11.0, 5.4))
    offsets = np.linspace(-2, 2, len(bars)) * width

    for i, (col, label) in enumerate(bars):
        values = normalize_nonnegative(final_df[col])
        plt.bar(
            x + offsets[i], values,
            width=width,
            label=label,
            edgecolor="black",
            linewidth=0.8,
        )

    plt.xlabel("Client")
    plt.ylabel("Reward/Contribution")
    plt.xticks(x, [str(c) for c in clients])
    plt.legend(frameon=True, framealpha=0.5, ncol=2, fontsize=22)
    savefig(out_dir / "client_reward_contribution_balance")


def plot_metric_over_rounds(
    metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: Path,
    window: int,
    smooth_mode: str,
    ema_alpha: float,
) -> None:
    if metric not in metrics.columns or not metrics[metric].notna().any():
        return

    plt.figure(figsize=(8.2, 5.2))
    last_rounds = []

    for i, (rule, g) in enumerate(metrics.groupby("reward_rule", sort=False)):
        gg = g.groupby("round", as_index=False)[metric].mean().sort_values("round")
        if gg.empty:
            continue
        last_rounds.extend(gg["round"].tolist())
        yy = smooth_series(gg[metric], mode=smooth_mode, window=window, ema_alpha=ema_alpha)
        plt.plot(
            gg["round"], yy,
            label=rule,
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.5,
            linewidth=2.7,
            markevery=max(1, len(gg) // 8),
        )

    plt.xlabel("Rounds")
    plt.ylabel(ylabel)
    plt.legend(frameon=True, framealpha=0.5)
    if last_rounds:
        plt.xlim(min(last_rounds), max(last_rounds))
    savefig(out_dir / f"{metric}_over_rounds_by_reward_rule")


def plot_final_bar(summary: pd.DataFrame, metric_col: str, ylabel: str, out_dir: Path) -> None:
    col = metric_col + "_mean"
    err = metric_col + "_std"
    if col not in summary.columns or not summary[col].notna().any():
        return

    df = summary.sort_values(col, ascending=True)
    plt.figure(figsize=(7.4, 4.8))
    y = np.arange(len(df))
    xerr = df[err].to_numpy() if err in df.columns else None
    plt.barh(y, df[col].to_numpy(), xerr=xerr, capsize=4 if xerr is not None else 0)
    plt.yticks(y, df["reward_rule"].astype(str).tolist())
    plt.xlabel(ylabel)
    savefig(out_dir / f"final_{metric_col}_by_reward_rule")


def plot_client_diagnostics(
    client_df: pd.DataFrame,
    out_dir: Path,
    contribution_label: str = "Client contribution",
) -> None:
    g = client_df.groupby("client", as_index=False)[
        ["acceptance_rate", "mean_reward_proposed", "mean_contribution"]
    ].mean().sort_values("client")

    if g.empty:
        return

    plt.figure(figsize=(8.4, 5.2))
    plt.plot(
        g["client"], g["acceptance_rate"],
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
    )
    plt.xlabel("Client")
    plt.ylabel("Acceptance rate")
    plt.ylim(-0.02, 1.02)
    plt.xlim(min(g["client"]), max(g["client"]))
    savefig(out_dir / "client_acceptance_rate")

    plt.figure(figsize=(8.4, 5.2))
    plt.plot(
        g["client"], g["mean_reward_proposed"],
        label="Proposed reward",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
    )
    plt.plot(
        g["client"], g["mean_contribution"],
        label=contribution_label,
        marker="s",
        linestyle="--",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
    )
    plt.xlabel("Client")
    plt.ylabel("Mean value")
    plt.legend(frameon=True, framealpha=0.5)
    plt.xlim(min(g["client"]), max(g["client"]))
    savefig(out_dir / "client_reward_vs_contribution")


def plot_malicious(
    metrics: pd.DataFrame,
    out_dir: Path,
    window: int,
    smooth_mode: str,
    ema_alpha: float,
) -> None:
    df = metrics[metrics["reward_col"] == "reward_proposed"].copy()
    if df.empty:
        return
    if df["malicious_reward_share"].max() <= 0 and df["malicious_acceptance_rate"].isna().all():
        return

    cols = [
        ("malicious_reward_share", "Malicious reward share"),
        ("malicious_contribution_share", "Malicious contribution share"),
        ("malicious_acceptance_rate", "Malicious acceptance rate"),
        ("malicious_rejection_rate", "Malicious rejection rate"),
    ]

    plt.figure(figsize=(8.2, 5.2))
    last_rounds = []

    for i, (col, label) in enumerate(cols):
        if col not in df.columns or not df[col].notna().any():
            continue
        gg = df.groupby("round", as_index=False)[col].mean().sort_values("round")
        if gg.empty:
            continue
        last_rounds.extend(gg["round"].tolist())
        yy = smooth_series(gg[col], mode=smooth_mode, window=window, ema_alpha=ema_alpha)
        plt.plot(
            gg["round"], yy,
            label=label,
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.5,
            linewidth=2.7,
            markevery=max(1, len(gg) // 8),
        )

    plt.xlabel("Rounds")
    plt.ylabel("Rate/share")
    plt.ylim(-0.02, 1.02)
    if last_rounds:
        plt.xlim(min(last_rounds), max(last_rounds))
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "malicious_reward_and_acceptance_over_rounds")


def plot_dp(
    df: pd.DataFrame,
    out_dir: Path,
    contribution_col: str,
    contribution_label: str = "Contribution",
) -> None:
    if "dp_noise_multiplier" not in df.columns or df["dp_noise_multiplier"].nunique(dropna=True) < 2:
        return

    final = final_round_df(df, None).copy()
    final["reward_norm"] = normalize_nonnegative(final["reward_proposed"])
    final["contribution_norm"] = normalize_nonnegative(final[contribution_col])

    plt.figure(figsize=(7.5, 5.2))
    plt.scatter(
        final["dp_noise_multiplier"], final["reward_norm"],
        label="Proposed reward",
        marker="o",
        s=90,
        facecolors="white",
        linewidths=1.8,
    )
    plt.scatter(
        final["dp_noise_multiplier"], final["contribution_norm"],
        label=contribution_label,
        marker="s",
        s=90,
        facecolors="white",
        linewidths=1.8,
    )
    plt.xlabel("DP noise multiplier")
    plt.ylabel("Normalized final-round value")
    plt.legend(frameon=True, framealpha=0.5)
    plt.xlim(min(final["dp_noise_multiplier"]), max(final["dp_noise_multiplier"]))
    savefig(out_dir / "dp_noise_vs_reward_contribution")

    grouped = final.groupby("dp_noise_multiplier", as_index=False)[["reward_norm", "contribution_norm"]].mean()
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(
        grouped["dp_noise_multiplier"], grouped["reward_norm"],
        label="Mean proposed reward",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
    )
    plt.plot(
        grouped["dp_noise_multiplier"], grouped["contribution_norm"],
        label=f"Mean {contribution_label.lower()}",
        marker="s",
        linestyle="--",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
    )
    plt.xlabel("DP noise multiplier")
    plt.ylabel("Mean normalized final-round value")
    plt.xlim(min(final["dp_noise_multiplier"]), max(final["dp_noise_multiplier"]))
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "dp_group_reward_contribution")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fairness and reward analysis for incentive_ledger.csv")
    parser.add_argument("--incentive_csv", required=True, help="Path to incentive_ledger.csv")
    parser.add_argument("--out_dir", required=True, help="Output folder")

    parser.add_argument(
        "--contribution_col",
        default="contribution_score",
        help="Base contribution column. Usually contribution_score or raw_shapley.",
    )
    parser.add_argument(
        "--use_validated_contribution",
        action="store_true",
        help="Use alpha*contribution + beta*clinical relevance as fairness target.",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight on KS/Shapley contribution")
    parser.add_argument("--beta", type=float, default=0.9, help="Weight on clinical relevance")
    parser.add_argument("--relevance_col", default="relevance", help="Clinical relevance column")

    parser.add_argument("--final_round", type=int, default=None, help="Round for bar/fairness plots. Default: max round")
    parser.add_argument("--smooth", default="rolling", choices=["rolling", "rolling_center", "ema", "none"])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--ema_alpha", type=float, default=0.25)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--font_size", type=int, default=24)
    parser.add_argument("--no_bars", action="store_true")
    parser.add_argument(
        "--also_plot_standard_lorenz",
        action="store_true",
        help="Also save the standard reward-inequality Lorenz curve.",
    )
    args = parser.parse_args()

    set_plot_style(paper=args.paper, font_size=args.font_size)

    out_dir = Path(args.out_dir)
    csv_dir = out_dir / "csv"
    plot_dir = out_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_ledger(args.incentive_csv)

    if args.contribution_col not in df.columns:
        raise ValueError(f"{args.contribution_col!r} not found. Available columns: {list(df.columns)}")

    contribution_col = args.contribution_col
    contribution_label = "Client contribution"

    if args.use_validated_contribution:
        df = add_validated_contribution_score(
            df,
            alpha=args.alpha,
            beta=args.beta,
            contribution_col=args.contribution_col,
            relevance_col=args.relevance_col,
        )
        contribution_col = "validated_contribution_score"
        contribution_label = "Contribution"

    # Save prepared ledger so you can inspect exactly what was plotted.
    df.to_csv(csv_dir / "prepared_incentive_ledger_with_fairness_target.csv", index=False)

    final = final_round_df(df, args.final_round)
    final["contribution_norm"] = normalize_nonnegative(final[contribution_col])
    for reward_col in REWARD_RULES:
        final[f"{reward_col}_norm"] = normalize_nonnegative(final[reward_col])
    final.to_csv(csv_dir / "final_round_client_rewards_and_contributions.csv", index=False)

    metrics = compute_reward_metrics(df, contribution_col)
    client_summary = compute_client_summary(df, contribution_col)
    final_summary = summarize_final(metrics)

    metrics.to_csv(csv_dir / "fairness_metrics_by_round_and_reward_rule.csv", index=False)
    client_summary.to_csv(csv_dir / "client_fairness_summary.csv", index=False)
    final_summary.to_csv(csv_dir / "final_fairness_summary_by_reward_rule.csv", index=False)

    plot_client_balance(final, contribution_col, plot_dir, contribution_label=contribution_label)

    # Main fairness plot: contribution-proportional reward alignment.
    plot_contribution_aligned_fairness(final, contribution_col, plot_dir)

    # Optional: standard Lorenz curve for reward inequality only.
    if args.also_plot_standard_lorenz:
        plot_reward_inequality_lorenz(final, plot_dir)

    plot_client_diagnostics(client_summary, plot_dir, contribution_label=contribution_label)
    plot_malicious(metrics, plot_dir, args.window, args.smooth, args.ema_alpha)
    plot_dp(df, plot_dir, contribution_col, contribution_label=contribution_label)

    round_specs = [
        ("pearson", "Reward--contribution Pearson correlation"),
        ("spearman", "Reward--contribution Spearman correlation"),
        ("l1_gap", "Reward--contribution L1 gap"),
        ("l2_gap", "Reward--contribution L2 gap"),
        ("max_gap", "Reward--contribution maximum gap"),
        ("reward_gini", "Reward Gini coefficient"),
        ("reward_entropy", "Reward entropy"),
        ("dp_noise_reward_pearson", "DP-noise/reward Pearson correlation"),
        ("dp_noise_contribution_pearson", "DP-noise/contribution Pearson correlation"),
        ("dp_reward_gap_high_minus_low", "DP reward gap: high noise minus low noise"),
        ("dp_contribution_gap_high_minus_low", "DP contribution gap: high noise minus low noise"),
    ]
    for metric, ylabel in round_specs:
        plot_metric_over_rounds(metrics, metric, ylabel, plot_dir, args.window, args.smooth, args.ema_alpha)

    if not args.no_bars:
        bar_specs = [
            ("pearson", "Final reward--contribution Pearson correlation"),
            ("spearman", "Final reward--contribution Spearman correlation"),
            ("l1_gap", "Final reward--contribution L1 gap"),
            ("l2_gap", "Final reward--contribution L2 gap"),
            ("max_gap", "Final reward--contribution maximum gap"),
            ("reward_gini", "Final reward Gini coefficient"),
            ("malicious_reward_share", "Final malicious reward share"),
            ("malicious_rejection_rate", "Final malicious rejection rate"),
            ("dp_noise_reward_pearson", "Final DP-noise/reward correlation"),
        ]
        for metric, ylabel in bar_specs:
            plot_final_bar(final_summary, metric, ylabel, plot_dir)

    print("\nSaved fairness/reward analysis")
    print(f"Fairness target: {contribution_col}")
    if args.use_validated_contribution:
        print(f"Validated contribution: alpha={args.alpha}, beta={args.beta}, relevance_col={args.relevance_col}")
    print(f"CSV outputs:  {csv_dir}")
    print(f"Plot outputs: {plot_dir}")

    preview_cols = [
        "reward_rule", "contribution_target", "pearson_mean", "spearman_mean",
        "l1_gap_mean", "l2_gap_mean", "max_gap_mean", "reward_gini_mean",
        "reward_entropy_mean", "malicious_reward_share_mean",
        "malicious_rejection_rate_mean", "dp_noise_reward_pearson_mean",
    ]
    preview_cols = [c for c in preview_cols if c in final_summary.columns]
    print("\nFinal fairness summary:")
    print(final_summary[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
