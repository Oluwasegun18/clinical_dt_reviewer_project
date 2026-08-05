#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ledger_rewards_contribution_aligned_proposed.py

Plot incentive fairness from incentive_ledger.csv without recalculating the
benchmark rewards.

This script uses the reward columns already stored in the ledger:

    reward_shapley
    reward_equal
    reward_latency
    reward_proposed

It only recalculates the contribution target used for comparison.

Contribution definition
-----------------------
The contribution used in the plots is the normalized intent-aware contribution:

    q_i = alpha * max(contribution_score_i, 0)
        + beta  * relevance_i
        + model_gain_i

Then, per round over active clients:

    contribution_i = q_i / sum_j q_j

This makes the contribution bar comparable to the reward bars because all are
shares on the same scale.

Active clients
--------------
By default, normalization is over accepted submitted clients. If no client is
accepted in a round, the script falls back to all submitted clients, matching the
adaptive aggregation fallback logic.

Important
---------
This script does NOT recalculate benchmark rewards. It uses the ledger values:
reward_shapley, reward_equal, reward_latency. It also uses reward_proposed from
the ledger by default.

Optional proposed reward modes:
- ledger: use reward_proposed already stored in the ledger.
- aggregation_softmax: recompute proposed reward as the aggregation weight, softmax(q_i / agg_temperature).
- contribution_share: recompute proposed reward directly as the normalized contribution share q_i / sum_j q_j.
- reward_softmax: recompute proposed reward using a separate reward temperature, softmax(q_i / reward_temperature).

The benchmark rewards are never recomputed.

Expected ledger columns:
round,method,ablation,client,accepted,is_malicious,attack_type,malicious_ratio,
dp_noise_multiplier,contribution_score,raw_shapley,relevance,model_gain,
reward_shapley,reward_equal,reward_latency,reward_proposed,
compute_time_sec,comm_latency_sec,total_latency_sec,trial

Example
-------
python plot_ledger_rewards_contribution_aligned_proposed.py \
  --incentive_csv ./incentive_ledger.csv \
  --out_dir ./plots_ledger_rewards \
  --alpha 0.1 \
  --beta 0.9 \
  --round_mode all \
  --client_agg mean \
  --num_clients 10 \
  --paper

Folder example
--------------
python plot_ledger_rewards_contribution_aligned_proposed.py \
  --results_root ./results_dp_iid_noise/physionet2012 \
  --out_dir ./plots_ledger_rewards \
  --alpha 0.1 \
  --beta 0.9 \
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


# These are the plotted reward columns. By default they are taken directly
# from the ledger.
REWARD_RULES: Dict[str, str] = {
    "reward_shapley_plot": "Shapley-only",
    "reward_equal_plot": "Equal",
    "reward_latency_plot": "Latency-based",
    "reward_proposed_plot": "Proposed",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "-", "-", "-", (0, (3, 1, 1, 1)), (0, (5, 2))]


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
        "legend.fontsize": max(9, font_size - 6),
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
        "reward_shapley",
        "reward_equal",
        "reward_latency",
        "reward_proposed",
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
    if "total_latency_sec" not in df.columns:
        df["total_latency_sec"] = np.nan

    df["method"] = df["method"].astype(str)
    df["ablation"] = df["ablation"].astype(str)
    df["attack_type"] = df["attack_type"].astype(str)

    return df


def complete_round_client_grid(
    df: pd.DataFrame,
    num_clients: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add missing round-client rows only for zero-reward accounting.

    has_submission=True means the row existed in the ledger. Missing rows are added
    so every client appears in averaged plots. Missing rows get zero reward and
    zero normalized contribution.
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

        # Preserve each client's DP noise for rows inserted by the grid.
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

    return pd.concat(parts, ignore_index=True)


def compute_normalized_contribution_and_plot_rewards(
    df: pd.DataFrame,
    alpha: float = 0.1,
    beta: float = 0.9,
    active_set: str = "accepted",
    proposed_source: str = "ledger",
    agg_temperature: float = 20.0,
    reward_temperature: float = 0.1,
    normalize_raw_contribution: bool = False,
) -> pd.DataFrame:
    """
    Compute normalized contribution while keeping benchmark rewards from the ledger.

    The benchmark rewards are NOT recomputed.

    active_set:
        accepted  -> normalize contribution among accepted submitted clients,
                     fallback to submitted if none accepted.
        submitted -> normalize contribution among all submitted clients.

    proposed_source:
        ledger              -> use ledger reward_proposed.
        aggregation_softmax -> recompute proposed reward as softmax(q/agg_temperature),
                               i.e., the aggregation influence weight.
        contribution_share  -> recompute proposed reward as normalized contribution q/sum(q).
        reward_softmax      -> recompute proposed reward as softmax(q/reward_temperature),
                               using a separate reward temperature.

        Benchmark rewards always remain ledger-based.
    """
    out = df.copy()

    out["contribution_raw"] = (
        float(alpha) * out["contribution_score"].clip(lower=0.0)
        + float(beta) * out["relevance"].fillna(0.0)
        + out["model_gain"].fillna(0.0)
    )

    out["contribution"] = 0.0
    out["active_for_contribution"] = False

    # Ledger reward columns used for plotting.
    out["reward_shapley_plot"] = out["reward_shapley"].fillna(0.0)
    out["reward_equal_plot"] = out["reward_equal"].fillna(0.0)
    out["reward_latency_plot"] = out["reward_latency"].fillna(0.0)
    out["reward_proposed_plot"] = out["reward_proposed"].fillna(0.0)

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

        if active_set == "submitted":
            active_idx = submitted_idx
        elif active_set == "accepted":
            accepted_idx = g.index[
                g["has_submission"].astype(bool) & g["accepted"].astype(bool)
            ]
            active_idx = accepted_idx if len(accepted_idx) > 0 else submitted_idx
        else:
            raise ValueError("active_set must be one of: accepted, submitted")

        raw = out.loc[active_idx, "contribution_raw"].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        if normalize_raw_contribution:
            rmin = float(np.nanmin(raw))
            rmax = float(np.nanmax(raw))
            if rmax > rmin:
                raw_for_share = (raw - rmin) / (rmax - rmin)
            else:
                raw_for_share = np.ones_like(raw, dtype=float)
        else:
            raw_for_share = raw

        out.loc[active_idx, "active_for_contribution"] = True

        contribution_share = normalize_nonnegative(raw_for_share)
        out.loc[active_idx, "contribution"] = contribution_share

        if proposed_source == "aggregation_softmax":
            # This matches the model aggregation weight. With a large
            # agg_temperature, this can become almost identical to equal reward.
            out.loc[active_idx, "reward_proposed_plot"] = softmax(raw_for_share, agg_temperature)
            inactive_in_round = g.index.difference(active_idx)
            out.loc[inactive_in_round, "reward_proposed_plot"] = 0.0

        elif proposed_source == "contribution_share":
            # This is the most direct contribution-aligned incentive reward.
            # It decouples reward fairness from the smoothing used for model aggregation.
            out.loc[active_idx, "reward_proposed_plot"] = contribution_share
            inactive_in_round = g.index.difference(active_idx)
            out.loc[inactive_in_round, "reward_proposed_plot"] = 0.0

        elif proposed_source == "reward_softmax":
            # This uses a separate reward temperature, so aggregation can remain
            # smooth while reward allocation is more contribution-sensitive.
            out.loc[active_idx, "reward_proposed_plot"] = softmax(raw_for_share, reward_temperature)
            inactive_in_round = g.index.difference(active_idx)
            out.loc[inactive_in_round, "reward_proposed_plot"] = 0.0

        elif proposed_source == "ledger":
            pass

        else:
            raise ValueError(
                "proposed_source must be one of: ledger, aggregation_softmax, contribution_share, reward_softmax"
            )

    return out


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
        "contribution_raw",
        "contribution",
        "contribution_score",
        "relevance",
        "model_gain",
        "reward_shapley_plot",
        "reward_equal_plot",
        "reward_latency_plot",
        "reward_proposed_plot",
        "has_submission",
        "accepted",
        "active_for_contribution",
        "is_malicious",
    ]

    if client_agg == "mean":
        agg_dict = {
            "dp_noise_multiplier": ("dp_noise_multiplier", "mean"),
            "contribution_raw": ("contribution_raw", "mean"),
            "contribution": ("contribution", "mean"),
            "contribution_score": ("contribution_score", "mean"),
            "relevance": ("relevance", "mean"),
            "model_gain": ("model_gain", "mean"),
            "reward_shapley_plot": ("reward_shapley_plot", "mean"),
            "reward_equal_plot": ("reward_equal_plot", "mean"),
            "reward_latency_plot": ("reward_latency_plot", "mean"),
            "reward_proposed_plot": ("reward_proposed_plot", "mean"),
            "submission_rate": ("has_submission", "mean"),
            "acceptance_rate": ("accepted", "mean"),
            "active_contribution_rate": ("active_for_contribution", "mean"),
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
            "active_for_contribution": "active_contribution_rate",
        })
        return out

    raise ValueError("client_agg must be one of: mean, final")


def add_normalized_client_values(client_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize after client-level aggregation so the DP-noise gap plot compares
    client-level reward shares to client-level contribution shares.
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
    group_cols = [c for c in out.columns if c in group_cols]

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


def plot_incentive_bar(client_df: pd.DataFrame, out_dir: Path) -> None:
    bar_df = client_df.groupby("client", as_index=False).agg(
        reward_shapley_plot=("reward_shapley_plot", "mean"),
        reward_equal_plot=("reward_equal_plot", "mean"),
        reward_latency_plot=("reward_latency_plot", "mean"),
        reward_proposed_plot=("reward_proposed_plot", "mean"),
        contribution=("contribution", "mean"),
    ).sort_values("client")

    clients = bar_df["client"].astype(int).to_numpy()
    x = np.arange(len(clients))
    width = 0.15

    bars = [
        ("reward_shapley_plot", "Shapley-only"),
        ("reward_equal_plot", "Equal"),
        ("reward_latency_plot", "Latency-based"),
        ("reward_proposed_plot", "Proposed"),
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
    savefig(out_dir / "incentive_bar_ledger_rewards_normalized_contribution")


def plot_reward_against_contribution(client_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
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
    plt.ylabel("Reward")
    plt.xlim(x_min, x_max)
    plt.ylim(bottom=0.0)
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "reward_against_contribution_ledger_rewards")

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
            active_contribution_rate=("active_contribution_rate", "mean"),
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
    savefig(out_dir / "normalized_reward_minus_contribution_against_dp_noise_ledger_rewards")


def plot_activity_by_noise(gap_summary: pd.DataFrame, out_dir: Path) -> None:
    g = gap_summary.groupby("dp_noise_multiplier", as_index=False).agg(
        submission_rate=("submission_rate", "mean"),
        acceptance_rate=("acceptance_rate", "mean"),
        active_contribution_rate=("active_contribution_rate", "mean"),
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
        g["active_contribution_rate"],
        marker="^",
        linestyle="-.",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.7,
        label="Active contribution rate",
    )

    x = g["dp_noise_multiplier"].to_numpy(dtype=float)
    plt.xlabel("DP noise multiplier")
    plt.ylabel("Rate")
    plt.ylim(0.0, 1.05)
    plt.xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(dp_tick_formatter))
    plt.legend(frameon=True, framealpha=0.5)
    savefig(out_dir / "submission_acceptance_contribution_activity_against_dp_noise")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use ledger rewards and normalized intent-aware contribution for fairness plots."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--incentive_csv", default=None, help="Path to one incentive_ledger.csv")
    input_group.add_argument("--results_root", default=None, help="Folder containing incentive_ledger.csv files")

    parser.add_argument("--out_dir", required=True, help="Output folder")

    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for contribution_score")
    parser.add_argument("--beta", type=float, default=0.9, help="Weight for relevance")
    parser.add_argument(
        "--active_set",
        default="accepted",
        choices=["accepted", "submitted"],
        help="Client set used to normalize contribution per round.",
    )
    parser.add_argument(
        "--proposed_source",
        default="ledger",
        choices=["ledger", "aggregation_softmax", "contribution_share", "reward_softmax"],
        help=(
            "Use ledger reward_proposed, recompute proposed as aggregation softmax, "
            "recompute proposed as contribution share, or recompute proposed with a separate reward softmax. "
            "Benchmarks are never recalculated."
        ),
    )
    parser.add_argument("--agg_temperature", type=float, default=20.0)
    parser.add_argument("--reward_temperature", type=float, default=0.1)
    parser.add_argument(
        "--normalize_raw_contribution",
        action="store_true",
        help="Min-max normalize q_i within each active set before contribution-share normalization.",
    )

    parser.add_argument(
        "--round_mode",
        default="all",
        choices=["all", "final", "last_k"],
        help="Which rounds to use after computing normalized contribution.",
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

    prepared = compute_normalized_contribution_and_plot_rewards(
        full_df,
        alpha=args.alpha,
        beta=args.beta,
        active_set=args.active_set,
        proposed_source=args.proposed_source,
        agg_temperature=args.agg_temperature,
        reward_temperature=args.reward_temperature,
        normalize_raw_contribution=args.normalize_raw_contribution,
    )

    selected = select_rounds(
        prepared,
        mode=args.round_mode,
        last_k=args.last_k,
    )

    client_df = build_client_level_table(
        selected,
        client_agg=args.client_agg,
    )

    client_df = add_normalized_client_values(client_df)

    plot_incentive_bar(client_df, plot_dir)
    fit_summary = plot_reward_against_contribution(client_df, plot_dir)
    gap_summary = summarize_gap_by_noise(client_df)
    plot_gap_against_dp_noise(gap_summary, plot_dir)

    if not args.no_activity_plot:
        plot_activity_by_noise(gap_summary, plot_dir)

    full_df.to_csv(csv_dir / "complete_round_client_grid.csv", index=False)
    prepared.to_csv(csv_dir / "round_level_ledger_rewards_normalized_contribution.csv", index=False)
    selected.to_csv(csv_dir / "selected_round_level_values.csv", index=False)
    client_df.to_csv(csv_dir / "client_level_ledger_rewards_normalized_contribution.csv", index=False)
    fit_summary.to_csv(csv_dir / "reward_against_contribution_fit_summary.csv", index=False)
    gap_summary.to_csv(csv_dir / "normalized_reward_minus_contribution_by_dp_noise_summary.csv", index=False)

    print("\nSaved plots using ledger rewards and normalized contribution.")
    print(f"Plot outputs: {plot_dir}")
    print(f"CSV outputs:  {csv_dir}")

    print("\nMain figures:")
    print(f"  {plot_dir / 'incentive_bar_ledger_rewards_normalized_contribution.pdf'}")
    print(f"  {plot_dir / 'reward_against_contribution_ledger_rewards.pdf'}")
    print(f"  {plot_dir / 'normalized_reward_minus_contribution_against_dp_noise_ledger_rewards.pdf'}")

    print("\nDiagnostic figure:")
    print(f"  {plot_dir / 'submission_acceptance_contribution_activity_against_dp_noise.pdf'}")

    print("\nFit summary:")
    print(fit_summary.to_string(index=False))


if __name__ == "__main__":
    main()
