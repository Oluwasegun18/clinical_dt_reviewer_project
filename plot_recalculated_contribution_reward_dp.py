#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_recalculated_contribution_reward_dp.py

Fresh plotting script from incentive_ledger.csv.

This script recalculates the reward from the ledger using the same aggregation-weight
logic used in your adaptive method, but it renames the intent-aware score as
"contribution" for plotting and paper reporting.

Contribution used in this script:
    contribution_i = alpha * max(contribution_score_i, 0)
                   + beta  * relevance_i
                   + model_gain_i

Proposed reward is recalculated as:
    reward_proposed_calc_i = softmax(contribution_i / agg_temperature)

The softmax is applied per round over accepted clients only. If no client is
accepted in a round, the script falls back to the submitted clients in that round,
matching the fallback idea in intent_aware_aggregate(...).

Benchmark rewards are also recalculated:
    reward_shapley_calc  = normalized contribution_score
    reward_equal_calc    = uniform over active clients
    reward_latency_calc  = normalized inverse total latency
    reward_proposed_calc = softmax(contribution / temperature)

Main plots:
1. reward_against_contribution_recalculated.pdf
   Reward against recalculated contribution.

2. normalized_reward_minus_contribution_against_dp_noise_recalculated.pdf
   Normalized reward minus normalized contribution against DP noise.

Expected ledger columns:
round,method,ablation,client,accepted,is_malicious,attack_type,malicious_ratio,
dp_noise_multiplier,contribution_score,raw_shapley,relevance,model_gain,
reward_shapley,reward_equal,reward_latency,reward_proposed,
compute_time_sec,comm_latency_sec,total_latency_sec,trial

Example:
python plot_recalculated_contribution_reward_dp.py \
  --incentive_csv ./incentive_ledger.csv \
  --out_dir ./plots_recalculated_reward \
  --alpha 0.1 \
  --beta 0.9 \
  --agg_temperature 20.0 \
  --round_mode all \
  --client_agg mean \
  --num_clients 10 \
  --paper

For a folder containing many incentive_ledger.csv files:
python plot_recalculated_contribution_reward_dp.py \
  --results_root ./results_dp_iid_noise/physionet2012 \
  --out_dir ./plots_recalculated_reward \
  --alpha 0.1 \
  --beta 0.9 \
  --agg_temperature 20.0 \
  --round_mode all \
  --client_agg mean \
  --num_clients 10 \
  --paper
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


REWARD_RULES: Dict[str, str] = {
    "reward_shapley_calc": "Shapley-only",
    "reward_equal_calc": "Equal",
    "reward_latency_calc": "Latency-based",
    "reward_proposed_calc": "Proposed",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


# ---------------------------------------------------------------------
# Style and helpers
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
        "legend.fontsize": max(9, font_size - 3),
        "lines.linewidth": 2.6,
        "patch.force_edgecolor": True,
    })


def savefig(path_base: Path, dpi: int = 350) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight", dpi=dpi)
    plt.savefig(path_base.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close()


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def normalize_nonnegative(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)

    if len(arr) == 0:
        return arr

    total = float(arr.sum())
    if total <= 0:
        return np.ones_like(arr, dtype=float) / len(arr)

    return arr / total


def softmax(values: Sequence[float], temperature: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr

    tau = max(1e-12, float(temperature))
    logits = arr / tau
    logits = logits - np.nanmax(logits)
    exps = np.exp(logits)
    denom = np.clip(np.nansum(exps), 1e-12, None)
    return exps / denom


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


def linear_fit(x: Sequence[float], y: Sequence[float]) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    out = {
        "n": int(len(x)),
        "slope": np.nan,
        "intercept": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
        "r2": np.nan,
        "mae_to_diagonal": np.nan,
        "rmse_to_diagonal": np.nan,
    }

    if len(x) < 3 or np.std(x) <= 1e-12:
        return out

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    out.update({
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson": safe_corr(x, y, "pearson"),
        "spearman": safe_corr(x, y, "spearman"),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan,
        "mae_to_diagonal": float(np.mean(np.abs(y - x))),
        "rmse_to_diagonal": float(np.sqrt(np.mean((y - x) ** 2))),
    })

    return out


def dp_tick_formatter(x: float, _pos: int) -> str:
    if abs(x) >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    if abs(x) >= 0.001:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.5f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------
# Loading ledgers
# ---------------------------------------------------------------------
def infer_trial_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        m = re.match(r"trial_([0-9]+)", part)
        if m:
            return int(m.group(1))
    return None


def load_one_ledger(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = str(path)

    if "trial" not in df.columns or df["trial"].isna().all():
        inferred = infer_trial_from_path(path)
        df["trial"] = 1 if inferred is None else inferred

    return df


def load_ledgers(incentive_csv: Optional[str], results_root: Optional[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    if incentive_csv:
        frames.append(load_one_ledger(Path(incentive_csv)))

    if results_root:
        root = Path(results_root)
        files = sorted(root.rglob("incentive_ledger.csv"))
        if not files:
            raise FileNotFoundError(f"No incentive_ledger.csv found under {root}")
        for path in files:
            frames.append(load_one_ledger(path))

    if not frames:
        raise ValueError("Provide either --incentive_csv or --results_root.")

    df = pd.concat(frames, ignore_index=True)

    required = [
        "round",
        "client",
        "accepted",
        "dp_noise_multiplier",
        "contribution_score",
        "relevance",
        "model_gain",
        "total_latency_sec",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ledger columns: {missing}")

    numeric_cols = [
        "round",
        "client",
        "malicious_ratio",
        "dp_noise_multiplier",
        "contribution_score",
        "raw_shapley",
        "relevance",
        "model_gain",
        "reward_shapley",
        "reward_equal",
        "reward_latency",
        "reward_proposed",
        "compute_time_sec",
        "comm_latency_sec",
        "total_latency_sec",
        "trial",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["accepted"] = as_bool(df["accepted"])

    if "is_malicious" in df.columns:
        df["is_malicious"] = as_bool(df["is_malicious"])
    else:
        df["is_malicious"] = False

    if "method" not in df.columns:
        df["method"] = "adaptive"
    if "ablation" not in df.columns:
        df["ablation"] = "none"
    if "attack_type" not in df.columns:
        df["attack_type"] = "none"
    if "malicious_ratio" not in df.columns:
        df["malicious_ratio"] = 0.0
    if "trial" not in df.columns:
        df["trial"] = 1

    df["method"] = df["method"].astype(str)
    df["ablation"] = df["ablation"].astype(str)
    df["attack_type"] = df["attack_type"].astype(str)

    return df


# ---------------------------------------------------------------------
# Complete grid and recalculated rewards
# ---------------------------------------------------------------------
def complete_round_client_grid(
    df: pd.DataFrame,
    num_clients: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a complete round-client table for plotting.

    Important:
    - has_submission=True means the row existed in the ledger.
    - Missing client-round pairs are added only for zero reward accounting.
    - Aggregation/reward recalculation is applied only to submitted clients,
      matching submissions.keys() in the original aggregation function.
    """
    df = df.copy()
    df["has_submission"] = True

    if num_clients is None:
        finite_clients = pd.to_numeric(df["client"], errors="coerce").dropna()
        if finite_clients.empty:
            raise ValueError("Could not infer number of clients.")
        num_clients = int(finite_clients.max()) + 1

    run_cols = [
        "source_file",
        "trial",
        "method",
        "ablation",
        "attack_type",
        "malicious_ratio",
    ]
    run_cols = [c for c in run_cols if c in df.columns]

    parts = []

    for key_vals, run_df in df.groupby(run_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)

        rounds = sorted(run_df["round"].dropna().astype(int).unique())
        clients = list(range(num_clients))

        full_index = pd.MultiIndex.from_product(
            [rounds, clients],
            names=["round", "client"],
        )

        g = (
            run_df.set_index(["round", "client"])
                  .reindex(full_index)
                  .reset_index()
        )

        for col, val in zip(run_cols, key_vals):
            g[col] = val

        # Preserve each client's DP-noise multiplier when missing rows are inserted.
        noise_map = run_df.groupby("client")["dp_noise_multiplier"].mean().to_dict()
        g["dp_noise_multiplier"] = g.apply(
            lambda row: noise_map.get(row["client"], row.get("dp_noise_multiplier", np.nan)),
            axis=1,
        )

        g["has_submission"] = g["has_submission"].fillna(False).astype(bool)
        g["accepted"] = g["accepted"].fillna(False).astype(bool)
        g["is_malicious"] = g["is_malicious"].fillna(False).astype(bool)

        fill_zero_cols = [
            "contribution_score",
            "raw_shapley",
            "relevance",
            "model_gain",
            "reward_shapley",
            "reward_equal",
            "reward_latency",
            "reward_proposed",
            "compute_time_sec",
            "comm_latency_sec",
            "total_latency_sec",
        ]

        for col in fill_zero_cols:
            if col in g.columns:
                g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)

        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    return out


def recompute_contribution_and_rewards(
    df: pd.DataFrame,
    alpha: float = 0.1,
    beta: float = 0.9,
    temperature: float = 20.0,
    max_selected: int = 0,
) -> pd.DataFrame:
    """
    Recalculate contribution and rewards from the ledger.

    The word "contribution" here refers to the intent-aware contribution score:
        contribution = alpha*contribution_score + beta*relevance + model_gain

    Recalculated proposed reward:
        softmax(contribution / temperature)
    over active clients in each round.
    """
    out = df.copy()

    out["contribution"] = (
        float(alpha) * out["contribution_score"].clip(lower=0.0)
        + float(beta) * out["relevance"].fillna(0.0)
        + out["model_gain"].fillna(0.0)
    )

    out["reward_shapley_calc"] = 0.0
    out["reward_equal_calc"] = 0.0
    out["reward_latency_calc"] = 0.0
    out["reward_proposed_calc"] = 0.0
    out["active_for_reward"] = False

    group_cols = [
        "source_file",
        "trial",
        "method",
        "ablation",
        "attack_type",
        "malicious_ratio",
        "round",
    ]
    group_cols = [c for c in group_cols if c in out.columns]

    for _, g in out.groupby(group_cols, dropna=False):
        submitted_idx = g.index[g["has_submission"].astype(bool)]

        if len(submitted_idx) == 0:
            continue

        accepted_idx = g.index[
            g["has_submission"].astype(bool) & g["accepted"].astype(bool)
        ]

        # Same fallback idea as the original function:
        # if no accepted clients, use all submitted clients.
        if len(accepted_idx) == 0:
            active_idx = submitted_idx
        else:
            active_idx = accepted_idx

        raw = out.loc[active_idx, "contribution"].to_numpy(dtype=float)

        if max_selected and len(raw) > int(max_selected):
            top_local = np.argsort(-raw)[:int(max_selected)]
            active_idx = active_idx[top_local]
            raw = raw[top_local]

        out.loc[active_idx, "active_for_reward"] = True

        # Proposed: softmax over contribution / temperature.
        proposed_w = softmax(raw, temperature)
        out.loc[active_idx, "reward_proposed_calc"] = proposed_w

        # Shapley-only: normalize raw contribution_score over active clients.
        shapley_vals = out.loc[active_idx, "contribution_score"].clip(lower=0.0).to_numpy(dtype=float)
        out.loc[active_idx, "reward_shapley_calc"] = normalize_nonnegative(shapley_vals)

        # Equal: uniform over active clients.
        out.loc[active_idx, "reward_equal_calc"] = 1.0 / len(active_idx)

        # Latency-based: lower total latency gives higher reward.
        latency = out.loc[active_idx, "total_latency_sec"].to_numpy(dtype=float)
        latency = np.nan_to_num(latency, nan=0.0, posinf=0.0, neginf=0.0)
        latency_score = 1.0 / np.maximum(latency, 1e-12)

        if np.isfinite(latency_score).all() and latency_score.sum() > 0:
            out.loc[active_idx, "reward_latency_calc"] = latency_score / latency_score.sum()
        else:
            out.loc[active_idx, "reward_latency_calc"] = 1.0 / len(active_idx)

    return out


# ---------------------------------------------------------------------
# Selection and aggregation for plotting
# ---------------------------------------------------------------------
def select_rounds(df: pd.DataFrame, mode: str = "all", last_k: int = 5) -> pd.DataFrame:
    if mode == "all":
        return df.copy()

    group_cols = [
        "source_file",
        "trial",
        "method",
        "ablation",
        "attack_type",
        "malicious_ratio",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    parts = []

    for _, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("round")
        max_round = g["round"].max()

        if mode == "final":
            parts.append(g[g["round"] == max_round])
        elif mode == "last_k":
            parts.append(g[g["round"] > max_round - int(last_k)])
        else:
            raise ValueError("round_mode must be one of: all, final, last_k")

    return pd.concat(parts, ignore_index=True)


def build_client_level_table(df: pd.DataFrame, client_agg: str = "mean") -> pd.DataFrame:
    """
    Aggregates recalculated round-level reward/contribution values into
    one row per client per run.
    """
    group_cols = [
        "source_file",
        "trial",
        "method",
        "ablation",
        "attack_type",
        "malicious_ratio",
        "client",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    value_cols = [
        "dp_noise_multiplier",
        "contribution",
        "contribution_score",
        "relevance",
        "model_gain",
        "reward_shapley_calc",
        "reward_equal_calc",
        "reward_latency_calc",
        "reward_proposed_calc",
        "has_submission",
        "accepted",
        "active_for_reward",
        "is_malicious",
    ]

    if client_agg == "mean":
        agg_dict = {
            "dp_noise_multiplier": ("dp_noise_multiplier", "mean"),
            "contribution": ("contribution", "mean"),
            "contribution_score": ("contribution_score", "mean"),
            "relevance": ("relevance", "mean"),
            "model_gain": ("model_gain", "mean"),
            "reward_shapley_calc": ("reward_shapley_calc", "mean"),
            "reward_equal_calc": ("reward_equal_calc", "mean"),
            "reward_latency_calc": ("reward_latency_calc", "mean"),
            "reward_proposed_calc": ("reward_proposed_calc", "mean"),
            "submission_rate": ("has_submission", "mean"),
            "acceptance_rate": ("accepted", "mean"),
            "active_reward_rate": ("active_for_reward", "mean"),
            "is_malicious": ("is_malicious", "max"),
        }
        return df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    if client_agg == "final":
        parts = []
        for _, g in df.groupby(group_cols, dropna=False):
            g = g.sort_values("round")
            parts.append(g.tail(1)[group_cols + value_cols])
        out = pd.concat(parts, ignore_index=True)
        out = out.rename(columns={
            "has_submission": "submission_rate",
            "accepted": "acceptance_rate",
            "active_for_reward": "active_reward_rate",
        })
        return out

    raise ValueError("client_agg must be one of: mean, final")


def add_normalized_client_values(client_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds normalized contribution and normalized rewards within each run.
    This is used for:
        normalized reward - normalized contribution
    """
    out = client_df.copy()

    group_cols = [
        "source_file",
        "trial",
        "method",
        "ablation",
        "attack_type",
        "malicious_ratio",
    ]
    group_cols = [c for c in group_cols if c in out.columns]

    out["contribution_norm"] = np.nan

    for reward_col in REWARD_RULES:
        out[f"{reward_col}_norm"] = np.nan
        out[f"{reward_col}_minus_contribution_norm"] = np.nan

    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)

        contribution_norm = normalize_nonnegative(out.loc[idx, "contribution"])
        out.loc[idx, "contribution_norm"] = contribution_norm

        for reward_col in REWARD_RULES:
            reward_norm = normalize_nonnegative(out.loc[idx, reward_col])
            out.loc[idx, f"{reward_col}_norm"] = reward_norm
            out.loc[idx, f"{reward_col}_minus_contribution_norm"] = reward_norm - contribution_norm

    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_reward_against_contribution(client_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Plot 1:
        recalculated reward against recalculated contribution.
    """
    rows = []

    x = client_df["contribution"].to_numpy(dtype=float)
    finite_x = x[np.isfinite(x)]

    if len(finite_x) == 0:
        raise ValueError("No finite contribution values for plotting.")

    x_min = float(np.nanmin(finite_x))
    x_max = float(np.nanmax(finite_x))

    if np.isclose(x_min, x_max):
        x_min = 0.0
        x_max = max(float(x_max), 1e-6)

    xx = np.linspace(x_min, x_max, 200)

    plt.figure(figsize=(8.2, 5.8))

    plt.plot(
        xx,
        xx,
        "--",
        linewidth=1.8,
        label="Reward = contribution",
    )

    for i, (reward_col, label) in enumerate(REWARD_RULES.items()):
        y = client_df[reward_col].to_numpy(dtype=float)
        fit = linear_fit(x, y)

        if np.isfinite(fit["slope"]):
            yy = fit["slope"] * xx + fit["intercept"]
            plt.plot(
                xx,
                yy,
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=2.5,
                label=f"{label} fit, slope={fit['slope']:.2f}",
            )

        plt.scatter(
            x,
            y,
            marker=MARKERS[i % len(MARKERS)],
            s=75,
            facecolors="white",
            linewidths=1.5,
            alpha=0.85,
        )

        rows.append({
            "reward_rule": label,
            "reward_col": reward_col,
            **fit,
        })

    plt.xlabel("Contribution")
    plt.ylabel("Recalculated reward")
    plt.xlim(x_min, x_max)
    plt.ylim(bottom=0.0)
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "reward_against_contribution_recalculated")

    return pd.DataFrame(rows)


def summarize_gap_by_noise(client_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for reward_col, label in REWARD_RULES.items():
        gap_col = f"{reward_col}_minus_contribution_norm"
        reward_norm_col = f"{reward_col}_norm"

        g = client_df.groupby("dp_noise_multiplier", as_index=False).agg(
            gap_mean=(gap_col, "mean"),
            gap_std=(gap_col, "std"),
            reward_norm_mean=(reward_norm_col, "mean"),
            reward_norm_std=(reward_norm_col, "std"),
            contribution_norm_mean=("contribution_norm", "mean"),
            contribution_norm_std=("contribution_norm", "std"),
            contribution_mean=("contribution", "mean"),
            reward_mean=(reward_col, "mean"),
            submission_rate=("submission_rate", "mean"),
            acceptance_rate=("acceptance_rate", "mean"),
            active_reward_rate=("active_reward_rate", "mean"),
            n_clients=("client", "count"),
        )

        g["reward_rule"] = label
        g["reward_col"] = reward_col
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)

    for c in ["gap_std", "reward_norm_std", "contribution_norm_std"]:
        out[c] = out[c].fillna(0.0)

    return out


def plot_gap_against_dp_noise(gap_summary: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8.2, 5.7))

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.6,
        label="Reward = contribution",
    )

    for i, (label, g) in enumerate(gap_summary.groupby("reward_rule", sort=False)):
        g = g.sort_values("dp_noise_multiplier")

        plt.errorbar(
            g["dp_noise_multiplier"],
            g["gap_mean"],
            yerr=g["gap_std"],
            label=label,
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.5,
            linewidth=2.7,
            capsize=4,
        )

    x = gap_summary["dp_noise_multiplier"].to_numpy(dtype=float)
    plt.xlabel("DP noise multiplier")
    plt.ylabel("Normalized reward $-$ normalized contribution")
    plt.xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(dp_tick_formatter))
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "normalized_reward_minus_contribution_against_dp_noise_recalculated")


def plot_check_bar(client_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Check bar using recalculated values.
    """
    bar_df = client_df.groupby("client", as_index=False).agg(
        reward_shapley_calc=("reward_shapley_calc", "mean"),
        reward_equal_calc=("reward_equal_calc", "mean"),
        reward_latency_calc=("reward_latency_calc", "mean"),
        reward_proposed_calc=("reward_proposed_calc", "mean"),
        contribution=("contribution", "mean"),
    ).sort_values("client")

    clients = bar_df["client"].astype(int).to_numpy()
    x = np.arange(len(clients))
    width = 0.15

    bars = [
        ("reward_shapley_calc", "Shapley-only"),
        ("reward_equal_calc", "Equal"),
        ("reward_latency_calc", "Latency-based"),
        ("reward_proposed_calc", "Proposed"),
        ("contribution", "Contribution"),
    ]

    plt.figure(figsize=(11.2, 5.6))

    offsets = np.linspace(-2, 2, len(bars)) * width

    for i, (col, label) in enumerate(bars):
        plt.bar(
            x + offsets[i],
            bar_df[col].to_numpy(dtype=float),
            width=width,
            label=label,
            edgecolor="black",
            linewidth=0.8,
        )

    plt.xlabel("Client")
    plt.ylabel("Contribution--reward balance")
    plt.xticks(x, [str(c) for c in clients])
    plt.legend(frameon=True, framealpha=0.5, ncol=2)
    savefig(out_dir / "check_bar_recalculated_contribution_reward")


def plot_acceptance_and_activity_by_noise(gap_summary: pd.DataFrame, out_dir: Path) -> None:
    """
    Optional diagnostic plot showing whether reward differences are caused by
    acceptance or active aggregation frequency.
    """
    # Use one row per noise value; these fields are same across reward rules,
    # so average over reward rules.
    g = gap_summary.groupby("dp_noise_multiplier", as_index=False).agg(
        submission_rate=("submission_rate", "mean"),
        acceptance_rate=("acceptance_rate", "mean"),
        active_reward_rate=("active_reward_rate", "mean"),
    ).sort_values("dp_noise_multiplier")

    plt.figure(figsize=(8.2, 5.7))

    plt.plot(
        g["dp_noise_multiplier"],
        g["submission_rate"],
        marker="o",
        linestyle="-",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
        label="Submission rate",
    )
    plt.plot(
        g["dp_noise_multiplier"],
        g["acceptance_rate"],
        marker="s",
        linestyle="--",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
        label="Acceptance rate",
    )
    plt.plot(
        g["dp_noise_multiplier"],
        g["active_reward_rate"],
        marker="^",
        linestyle="-.",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
        label="Active reward rate",
    )

    x = g["dp_noise_multiplier"].to_numpy(dtype=float)
    plt.xlabel("DP noise multiplier")
    plt.ylabel("Rate")
    plt.ylim(0.0, 1.05)
    plt.xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(dp_tick_formatter))
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "submission_acceptance_activity_against_dp_noise")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalculate contribution/reward from incentive ledger and plot DP-noise fairness."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--incentive_csv", default="./incentive_ledger.csv", help="Path to one incentive_ledger.csv")
    input_group.add_argument("--results_root", default=None, help="Folder containing incentive_ledger.csv files")

    parser.add_argument("--out_dir", required="./plots_recalculated_reward", help="Output folder")

    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for contribution_score")
    parser.add_argument("--beta", type=float, default=0.9, help="Weight for relevance")
    parser.add_argument("--agg_temperature", type=float, default=20.0, help="Softmax temperature")
    parser.add_argument("--max_selected", type=int, default=0, help="Optional top-k selection before softmax. 0 disables it.")

    parser.add_argument(
        "--round_mode",
        default="all",
        choices=["all", "final", "last_k"],
        help="Which rounds to use after recalculating per-round rewards.",
    )
    parser.add_argument("--last_k", type=int, default=5)
    parser.add_argument(
        "--client_agg",
        default="mean",
        choices=["mean", "final"],
        help="How to aggregate selected rounds per client.",
    )
    parser.add_argument("--num_clients", type=int, default=None, help="Force clients 0...(num_clients-1) to appear.")

    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--font_size", type=int, default=15)
    parser.add_argument("--no_check_bar", action="store_true")
    parser.add_argument("--no_activity_plot", action="store_true")

    args = parser.parse_args()

    set_plot_style(paper=args.paper, font_size=args.font_size)

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    csv_dir = out_dir / "csv"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    df = load_ledgers(args.incentive_csv, args.results_root)

    if "is_malicious" in df.columns and df["is_malicious"].any():
        print("Warning: is_malicious=True found in the ledger.")
    if "malicious_ratio" in df.columns and pd.to_numeric(df["malicious_ratio"], errors="coerce").fillna(0).max() > 0:
        print("Warning: malicious_ratio > 0 found in the ledger.")

    full_df = complete_round_client_grid(df, num_clients=args.num_clients)

    recalculated = recompute_contribution_and_rewards(
        full_df,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.agg_temperature,
        max_selected=args.max_selected,
    )

    selected = select_rounds(
        recalculated,
        mode=args.round_mode,
        last_k=args.last_k,
    )

    client_df = build_client_level_table(
        selected,
        client_agg=args.client_agg,
    )

    client_df = add_normalized_client_values(client_df)

    fit_summary = plot_reward_against_contribution(client_df, plot_dir)
    gap_summary = summarize_gap_by_noise(client_df)
    plot_gap_against_dp_noise(gap_summary, plot_dir)

    if not args.no_check_bar:
        plot_check_bar(client_df, plot_dir)

    if not args.no_activity_plot:
        plot_acceptance_and_activity_by_noise(gap_summary, plot_dir)

    # Save CSV outputs.
    full_df.to_csv(csv_dir / "complete_round_client_grid.csv", index=False)
    recalculated.to_csv(csv_dir / "round_level_recalculated_contribution_reward.csv", index=False)
    selected.to_csv(csv_dir / "selected_round_level_recalculated_values.csv", index=False)
    client_df.to_csv(csv_dir / "client_level_recalculated_contribution_reward.csv", index=False)
    fit_summary.to_csv(csv_dir / "reward_against_contribution_fit_summary.csv", index=False)
    gap_summary.to_csv(csv_dir / "normalized_reward_minus_contribution_by_dp_noise_summary.csv", index=False)

    print("\nSaved recalculated contribution/reward plots.")
    print(f"Plot outputs: {plot_dir}")
    print(f"CSV outputs:  {csv_dir}")

    print("\nMain figures:")
    print(f"  {plot_dir / 'reward_against_contribution_recalculated.pdf'}")
    print(f"  {plot_dir / 'normalized_reward_minus_contribution_against_dp_noise_recalculated.pdf'}")

    print("\nDiagnostic figures:")
    print(f"  {plot_dir / 'check_bar_recalculated_contribution_reward.pdf'}")
    print(f"  {plot_dir / 'submission_acceptance_activity_against_dp_noise.pdf'}")

    print("\nFit summary:")
    print(fit_summary.to_string(index=False))


if __name__ == "__main__":
    main()
