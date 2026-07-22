#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_results_and_plots.py
--------------------------------
Compile simulation results from many experiment folders and generate publication-ready plots.

Designed for outputs from main8_reviewer_update.py / main8.py, including:
  - all_trials_metrics.csv
  - incentive_ledger.csv
  - incentive_gini*.csv
  - ledger_trial_*.csv
  - rdp_accounting.csv

Typical use:
  python compile_results_and_plots.py \
      --results_root ./local_results \
      --out_dir ./compiled_results \
      --metric acc

Cluster use:
  python compile_results_and_plots.py \
      --results_root /speed-scratch/ol_tal/simulation/clinical_dt_reviewer/results_reviewer \
      --out_dir /speed-scratch/ol_tal/simulation/clinical_dt_reviewer/compiled_results \
      --metric acc \
      --paper
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# -----------------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------------

METHOD_ORDER = [
    "adaptive", "proposed", "pom-dt", "pb-pom", "full",
    "fedavg", "fedprox", "fedsgd",
    "krum", "multikrum", "multi-krum", "median", "trimmed_mean", "trimmed-mean", "bulyan",
    "adaptive_no_consensus", "adaptive_consensus_only", "adaptive_quality_only",
    "adaptive_shapley_only", "adaptive_no_quality",
]

METHOD_LABELS = {
    "adaptive": "Proposed",
    "proposed": "Proposed",
    "pom-dt": "PoM-DT",
    "pb-pom": "pB-PoM",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedsgd": "FedSGD",
    "krum": "Krum",
    "multikrum": "Multi-Krum",
    "multi-krum": "Multi-Krum",
    "median": "Median",
    "coordinate_median": "Median",
    "coord_median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "trimmed-mean": "Trimmed Mean",
    "bulyan": "Bulyan",
    "adaptive_no_consensus": "No consensus",
    "adaptive_consensus_only": "Consensus only",
    "adaptive_quality_only": "Quality only",
    "adaptive_shapley_only": "Shapley only",
    "adaptive_no_quality": "No quality",
}

DATASETS = ["pathmnist", "tissuemnist", "organamnist", "organsmnist", "cifar10", "mnist"]
ATTACKS = ["none", "clean", "label_flip", "sign_flip", "scaling", "random_update", "gaussian_model_poisoning"]


def set_plot_style(paper: bool = False) -> None:
    """Set calm, publication-oriented matplotlib defaults without external style deps."""
    if paper:
        base = 9
        fig_w = 6.6
    else:
        base = 11
        fig_w = 8.0
    plt.rcParams.update({
        "figure.figsize": (fig_w, 4.8),
        "figure.dpi": 130,
        "savefig.dpi": 350 if paper else 220,
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.1,
        "lines.markersize": 5,
        "patch.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def clean_name(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "unknown"
    s = str(x).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s


def label_method(method: object) -> str:
    key = clean_name(method)
    return METHOD_LABELS.get(key, str(method))


def slugify(s: object, max_len: int = 120) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def method_sort_key(m: object) -> Tuple[int, str]:
    key = clean_name(m)
    try:
        return METHOD_ORDER.index(key), key
    except ValueError:
        return 999, key


def metric_label(metric: str) -> str:
    return {
        "acc": "Accuracy",
        "accuracy": "Accuracy",
        "loss": "Loss",
        "f1": "F1-score",
        "precision": "Precision",
        "recall": "Recall",
        "auc": "AUROC",
        "malicious_reward_share": "Malicious reward share",
        "accepted_malicious_rate": "Accepted malicious rate",
        "rejected_malicious_rate": "Rejected malicious rate",
        "accepted_honest_rate": "Accepted honest rate",
        "reward_gini": "Reward Gini coefficient",
    }.get(metric, metric.replace("_", " ").title())


# -----------------------------------------------------------------------------
# Metadata parsing and CSV loading
# -----------------------------------------------------------------------------

def read_optional_metadata(folder: Path) -> Dict[str, object]:
    """Read optional config files if present. Safe if files do not exist."""
    meta: Dict[str, object] = {}

    # Common names that may appear in future runs
    for name in ["config_parameters.csv", "args.csv", "config.csv", "run_config.csv"]:
        fp = folder / name
        if not fp.exists():
            continue
        try:
            cfg = pd.read_csv(fp)
            if {"parameter", "value"}.issubset(cfg.columns):
                meta.update(dict(zip(cfg["parameter"].astype(str), cfg["value"])))
            elif {"key", "value"}.issubset(cfg.columns):
                meta.update(dict(zip(cfg["key"].astype(str), cfg["value"])))
            elif len(cfg) == 1:
                meta.update(cfg.iloc[0].dropna().to_dict())
        except Exception:
            pass

    for name in ["config.json", "args.json", "run_config.json"]:
        fp = folder / name
        if not fp.exists():
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                meta.update(obj)
        except Exception:
            pass

    return meta


def infer_metadata_from_path(csv_path: Path, root: Path) -> Dict[str, object]:
    """Infer dataset, attack type, malicious ratio, etc. from folder names."""
    rel_parts = [p.lower() for p in csv_path.relative_to(root).parts[:-1]]
    joined = "/".join(rel_parts)
    name = "_".join(rel_parts)

    meta: Dict[str, object] = {
        "source_dir": str(csv_path.parent),
        "source_csv": str(csv_path),
        "experiment": csv_path.parent.name,
    }

    for ds in DATASETS:
        if ds in joined:
            meta["dataset"] = ds
            break

    # Attack parsing: longest first so gaussian_model_poisoning is not shortened
    for atk in sorted(ATTACKS, key=len, reverse=True):
        if atk in joined:
            meta["attack_type"] = "none" if atk == "clean" else atk
            break
    if "attack_type" not in meta:
        meta["attack_type"] = "none"

    # Malicious ratio: supports ratio_0.2, malicious_0.2, _20, label_flip_20
    ratio_patterns = [
        r"malicious[_-]?ratio[_-]?(0?\.\d+|1\.0|\d+)",
        r"ratio[_-]?(0?\.\d+|1\.0|\d+)",
        r"mal[_-]?(0?\.\d+|1\.0|\d+)",
        r"(?:label_flip|sign_flip|scaling|random_update|gaussian_model_poisoning)[_-](\d{1,3})(?:\D|$)",
    ]
    for pat in ratio_patterns:
        m = re.search(pat, name)
        if m:
            val = m.group(1)
            try:
                x = float(val)
                if x > 1.0:
                    x = x / 100.0
                meta["malicious_ratio"] = x
                break
            except Exception:
                pass
    if "malicious_ratio" not in meta:
        meta["malicious_ratio"] = 0.0 if meta.get("attack_type") in ["none", "clean"] else np.nan

    # Dirichlet alpha parsing
    m = re.search(r"(?:dirichlet|alpha)[_-]?(0?\.\d+|\d+)", name)
    if m:
        try:
            meta["dirichlet_alpha"] = float(m.group(1))
        except Exception:
            pass

    # Run type parsing
    if "ablation" in joined:
        meta["run_group"] = "ablation"
    elif "heterodp" in joined or "heterogeneous_dp" in joined:
        meta["run_group"] = "heterogeneous_dp"
    elif "benchmark" in joined or "baseline" in joined:
        meta["run_group"] = "benchmark"
    elif "attack" in joined or meta.get("attack_type") not in ["none", "clean"]:
        meta["run_group"] = "byzantine"
    else:
        meta["run_group"] = "general"

    return meta


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in ["method", "dataset", "attack_type", "run_group", "source_dir", "source_csv", "experiment"]:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="ignore")
            df[col] = converted
    return df


def load_metrics(results_root: Path) -> pd.DataFrame:
    files = sorted(results_root.rglob("all_trials_metrics.csv"))
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
            continue
        meta = infer_metadata_from_path(fp, results_root)
        meta.update(read_optional_metadata(fp.parent))
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
        if "round" in df.columns:
            df["round"] = pd.to_numeric(df["round"], errors="coerce")
        if "trial" not in df.columns:
            df["trial"] = 1
        if "method" in df.columns:
            df["method"] = df["method"].astype(str).map(clean_name)
        frames.append(coerce_numeric_columns(df))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def load_incentives(results_root: Path) -> pd.DataFrame:
    files = sorted(results_root.rglob("incentive_ledger.csv"))
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
            continue
        meta = infer_metadata_from_path(fp, results_root)
        meta.update(read_optional_metadata(fp.parent))
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
        if "trial" not in df.columns:
            df["trial"] = 1
        frames.append(coerce_numeric_columns(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_ledgers(results_root: Path) -> pd.DataFrame:
    files = sorted(results_root.rglob("ledger_trial_*.csv"))
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
            continue
        meta = infer_metadata_from_path(fp, results_root)
        meta.update(read_optional_metadata(fp.parent))
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
        m = re.search(r"ledger_trial_(\d+)\.csv", fp.name)
        if m and "trial" not in df.columns:
            df["trial"] = int(m.group(1))
        frames.append(coerce_numeric_columns(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------

def gini_coefficient(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0 or np.allclose(arr.sum(), 0):
        return float("nan")
    arr = np.sort(arr)
    n = arr.size
    return float((2.0 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1.0) / n)


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def final_round_table(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metrics.empty or metric not in metrics.columns:
        return pd.DataFrame()
    keys = ["source_dir", "experiment", "dataset", "attack_type", "malicious_ratio", "run_group", "method", "trial"]
    keys = [k for k in keys if k in metrics.columns]
    final_idx = metrics.groupby(keys, dropna=False)["round"].idxmax() if "round" in metrics.columns else metrics.index
    final = metrics.loc[final_idx].copy()
    group_keys = [k for k in ["experiment", "dataset", "attack_type", "malicious_ratio", "run_group", "method"] if k in final.columns]
    summary = final.groupby(group_keys, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": f"final_{metric}_mean", "std": f"final_{metric}_std", "count": "n_trials"})
    return summary.sort_values([c for c in ["dataset", "attack_type", "malicious_ratio"] if c in summary.columns] + ["method"], key=lambda s: s.map(method_sort_key) if s.name == "method" else s)


def summarize_fairness(incentives: pd.DataFrame) -> pd.DataFrame:
    if incentives.empty:
        return pd.DataFrame()
    reward_cols = [c for c in incentives.columns if c.startswith("reward_")]
    if not reward_cols:
        return pd.DataFrame()

    # Contribution proxy: prefer explicit contribution fields, otherwise use Shapley reward as contribution proxy.
    contrib_col = None
    for c in ["reward_contrib", "contribution", "shapley", "shapley_value", "reward_shapley"]:
        if c in incentives.columns:
            contrib_col = c
            break

    group_cols = [c for c in ["experiment", "dataset", "attack_type", "malicious_ratio", "run_group", "trial"] if c in incentives.columns]
    rows = []
    for keys, grp in incentives.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row_base = dict(zip(group_cols, keys))
        by_client = grp.groupby("client", dropna=False)[reward_cols + ([contrib_col] if contrib_col and contrib_col not in reward_cols else [])].sum().reset_index()
        contrib = by_client[contrib_col].to_numpy(dtype=float) if contrib_col else None

        for rc in reward_cols:
            rewards = by_client[rc].to_numpy(dtype=float)
            row = row_base.copy()
            row.update({
                "scheme": rc,
                "gini": gini_coefficient(rewards),
                "mean_reward": float(np.nanmean(rewards)) if len(rewards) else np.nan,
                "std_reward": float(np.nanstd(rewards)) if len(rewards) else np.nan,
            })
            if contrib is not None:
                row["reward_contribution_corr"] = pearson_corr(contrib, rewards)
                row["reward_contribution_l1_gap"] = float(np.nanmean(np.abs(normalize_vec(rewards) - normalize_vec(contrib))))
            rows.append(row)
    return pd.DataFrame(rows)


def normalize_vec(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0, None)
    s = arr.sum()
    if s <= 1e-12:
        return np.ones_like(arr) / max(len(arr), 1)
    return arr / s


def summarize_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "accepted" not in ledger.columns:
        return pd.DataFrame()
    df = ledger.copy()
    if df["accepted"].dtype == object:
        df["accepted_bool"] = df["accepted"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["accepted_bool"] = df["accepted"].astype(bool)
    group_cols = [c for c in ["experiment", "dataset", "attack_type", "malicious_ratio", "run_group", "trial", "round"] if c in df.columns]
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["acceptance_rate"] = float(grp["accepted_bool"].mean())
        if "is_malicious" in grp.columns:
            mal = grp[grp["is_malicious"].astype(bool)]
            hon = grp[~grp["is_malicious"].astype(bool)]
            row["accepted_malicious_rate"] = float(mal["accepted_bool"].mean()) if len(mal) else np.nan
            row["rejected_malicious_rate"] = float((~mal["accepted_bool"]).mean()) if len(mal) else np.nan
            row["accepted_honest_rate"] = float(hon["accepted_bool"].mean()) if len(hon) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------

def savefig(out_path: Path, formats: Sequence[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fp = out_path.with_suffix(f".{fmt}")
        plt.savefig(fp)
    plt.close()


def plot_learning_curves(metrics: pd.DataFrame, out_dir: Path, metric: str, formats: Sequence[str], separate_experiments: bool = True) -> None:
    if metrics.empty or metric not in metrics.columns or "round" not in metrics.columns:
        return
    group_keys = [c for c in ["experiment", "dataset", "attack_type", "malicious_ratio", "run_group"] if c in metrics.columns]

    if separate_experiments and group_keys:
        groups = metrics.groupby(group_keys, dropna=False)
    else:
        groups = [("all", metrics)]

    for keys, grp in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_keys, keys)) if group_keys else {"experiment": "all"}
        methods = sorted(grp["method"].dropna().unique(), key=method_sort_key)
        if len(methods) == 0:
            continue

        plt.figure(figsize=(8.2, 4.8))
        for method in methods:
            g = grp[grp["method"] == method]
            stat = g.groupby("round")[metric].agg(["mean", "std", "count"]).reset_index()
            x = stat["round"].to_numpy(dtype=float)
            y = stat["mean"].to_numpy(dtype=float)
            yerr = stat["std"].fillna(0).to_numpy(dtype=float)
            plt.plot(x, y, marker="o", label=label_method(method))
            if np.nanmax(yerr) > 0:
                plt.fill_between(x, y - yerr, y + yerr, alpha=0.13)

        title = f"{metric_label(metric)} over communication rounds"
        desc = []
        for k in ["dataset", "attack_type", "malicious_ratio"]:
            if k in meta and not pd.isna(meta[k]):
                desc.append(f"{k.replace('_', ' ')}={meta[k]}")
        if desc:
            title += "\n" + ", ".join(desc)
        plt.title(title)
        plt.xlabel("Communication round")
        plt.ylabel(metric_label(metric))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(ncol=2, frameon=False)
        fname = "learning_curve_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type", "malicious_ratio", "experiment"] if k in meta])) + f"_{metric}"
        savefig(out_dir / "learning_curves" / fname, formats)


def plot_final_bars(summary: pd.DataFrame, out_dir: Path, metric: str, formats: Sequence[str]) -> None:
    if summary.empty:
        return
    mean_col = f"final_{metric}_mean"
    std_col = f"final_{metric}_std"
    group_cols = [c for c in ["experiment", "dataset", "attack_type", "malicious_ratio", "run_group"] if c in summary.columns]
    for keys, grp in summary.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        grp = grp.sort_values("method", key=lambda s: s.map(method_sort_key))
        labels = [label_method(m) for m in grp["method"]]
        x = np.arange(len(grp))
        y = grp[mean_col].to_numpy(dtype=float)
        yerr = grp[std_col].fillna(0).to_numpy(dtype=float) if std_col in grp else None
        width = 0.72

        plt.figure(figsize=(max(7.0, 0.65 * len(grp) + 2), 4.8))
        plt.bar(x, y, width=width, yerr=yerr if np.nanmax(yerr) > 0 else None, capsize=3)
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel(f"Final {metric_label(metric)}")
        title = f"Final-round {metric_label(metric)} by method"
        subtitle = []
        for k in ["dataset", "attack_type", "malicious_ratio"]:
            if k in meta and not pd.isna(meta[k]):
                subtitle.append(f"{k.replace('_', ' ')}={meta[k]}")
        if subtitle:
            title += "\n" + ", ".join(subtitle)
        plt.title(title)
        plt.grid(axis="y", alpha=0.25)
        plt.grid(axis="x", visible=False)
        fname = "final_bar_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type", "malicious_ratio", "experiment"] if k in meta])) + f"_{metric}"
        savefig(out_dir / "final_bars" / fname, formats)


def plot_byzantine_robustness(summary: pd.DataFrame, out_dir: Path, metric: str, formats: Sequence[str]) -> None:
    if summary.empty or "malicious_ratio" not in summary.columns or "attack_type" not in summary.columns:
        return
    mean_col = f"final_{metric}_mean"
    df = summary.copy()
    df["malicious_ratio"] = pd.to_numeric(df["malicious_ratio"], errors="coerce")
    df = df[df["malicious_ratio"].notna()]
    if df.empty:
        return

    group_cols = [c for c in ["dataset", "attack_type"] if c in df.columns]
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        if meta.get("attack_type") in ["none", "clean", np.nan]:
            continue
        plt.figure(figsize=(8.2, 4.8))
        for method in sorted(grp["method"].dropna().unique(), key=method_sort_key):
            g = grp[grp["method"] == method].sort_values("malicious_ratio")
            plt.plot(g["malicious_ratio"].to_numpy(dtype=float) * 100.0, g[mean_col].to_numpy(dtype=float), marker="o", label=label_method(method))
        plt.xlabel("Malicious clients (%)")
        plt.ylabel(f"Final {metric_label(metric)}")
        title = f"Byzantine robustness under {str(meta.get('attack_type')).replace('_', ' ')}"
        if "dataset" in meta:
            title += f"\nDataset={meta['dataset']}"
        plt.title(title)
        plt.legend(ncol=2, frameon=False)
        fname = "byzantine_robustness_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type"]])) + f"_{metric}"
        savefig(out_dir / "byzantine" / fname, formats)


def plot_ablation(summary: pd.DataFrame, out_dir: Path, metric: str, formats: Sequence[str]) -> None:
    if summary.empty or "method" not in summary.columns:
        return
    ablation_methods = {"adaptive", "adaptive_no_consensus", "adaptive_consensus_only", "adaptive_quality_only", "adaptive_shapley_only", "adaptive_no_quality"}
    df = summary[summary["method"].isin(ablation_methods)].copy()
    if df.empty:
        return
    mean_col = f"final_{metric}_mean"
    std_col = f"final_{metric}_std"
    group_cols = [c for c in ["dataset", "attack_type", "malicious_ratio"] if c in df.columns]
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        grp = grp.sort_values("method", key=lambda s: s.map(method_sort_key))
        x = np.arange(len(grp))
        labels = [label_method(m) for m in grp["method"]]
        y = grp[mean_col].to_numpy(dtype=float)
        yerr = grp[std_col].fillna(0).to_numpy(dtype=float) if std_col in grp else None

        plt.figure(figsize=(max(7.0, 0.75 * len(grp) + 1.5), 4.8))
        plt.bar(x, y, yerr=yerr if np.nanmax(yerr) > 0 else None, capsize=3)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel(f"Final {metric_label(metric)}")
        plt.title("Ablation study of the proposed framework" + (f"\nDataset={meta.get('dataset')}, attack={meta.get('attack_type')}, ratio={meta.get('malicious_ratio')}" if meta else ""))
        plt.grid(axis="y", alpha=0.25)
        plt.grid(axis="x", visible=False)
        fname = "ablation_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type", "malicious_ratio"] if k in meta])) + f"_{metric}"
        savefig(out_dir / "ablation" / fname, formats)


def plot_fairness(fairness: pd.DataFrame, out_dir: Path, formats: Sequence[str]) -> None:
    if fairness.empty:
        return
    # Gini bar plots
    group_cols = [c for c in ["dataset", "attack_type", "malicious_ratio", "run_group"] if c in fairness.columns]
    for keys, grp in fairness.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        # Aggregate across trials
        g = grp.groupby("scheme")["gini"].mean().reset_index().sort_values("gini")
        labels = [s.replace("reward_", "").replace("_", " ").title() for s in g["scheme"]]
        x = np.arange(len(g))
        plt.figure(figsize=(max(6.5, 0.75 * len(g) + 1.5), 4.6))
        plt.bar(x, g["gini"].to_numpy(dtype=float))
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Gini coefficient")
        plt.title("Reward inequality by incentive scheme" + (f"\nDataset={meta.get('dataset')}, attack={meta.get('attack_type')}, ratio={meta.get('malicious_ratio')}" if meta else ""))
        plt.grid(axis="y", alpha=0.25)
        plt.grid(axis="x", visible=False)
        fname = "fairness_gini_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type", "malicious_ratio", "run_group"] if k in meta]))
        savefig(out_dir / "fairness" / fname, formats)

    # Correlation bar plots, if available
    if "reward_contribution_corr" in fairness.columns:
        for keys, grp in fairness.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            meta = dict(zip(group_cols, keys))
            g = grp.groupby("scheme")["reward_contribution_corr"].mean().reset_index().sort_values("reward_contribution_corr", ascending=False)
            if g["reward_contribution_corr"].isna().all():
                continue
            labels = [s.replace("reward_", "").replace("_", " ").title() for s in g["scheme"]]
            x = np.arange(len(g))
            plt.figure(figsize=(max(6.5, 0.75 * len(g) + 1.5), 4.6))
            plt.bar(x, g["reward_contribution_corr"].to_numpy(dtype=float))
            plt.xticks(x, labels, rotation=25, ha="right")
            plt.ylabel("Reward–contribution correlation")
            plt.ylim(-1.0, 1.05)
            plt.title("Reward–contribution alignment" + (f"\nDataset={meta.get('dataset')}, attack={meta.get('attack_type')}, ratio={meta.get('malicious_ratio')}" if meta else ""))
            plt.grid(axis="y", alpha=0.25)
            plt.grid(axis="x", visible=False)
            fname = "fairness_corr_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type", "malicious_ratio", "run_group"] if k in meta]))
            savefig(out_dir / "fairness" / fname, formats)


def plot_lorenz_curves(incentives: pd.DataFrame, out_dir: Path, formats: Sequence[str]) -> None:
    if incentives.empty:
        return
    reward_cols = [c for c in incentives.columns if c.startswith("reward_")]
    if not reward_cols or "client" not in incentives.columns:
        return
    group_cols = [c for c in ["dataset", "attack_type", "malicious_ratio", "run_group", "experiment"] if c in incentives.columns]
    for keys, grp in incentives.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        by_client = grp.groupby("client")[reward_cols].sum().reset_index()
        plt.figure(figsize=(6.2, 5.2))
        for rc in reward_cols:
            x = normalize_vec(by_client[rc].to_numpy(dtype=float))
            x = np.sort(x)
            cum = np.cumsum(x)
            lorenz = np.insert(cum, 0, 0)
            pop = np.linspace(0, 1, len(lorenz))
            plt.plot(pop, lorenz, label=rc.replace("reward_", "").replace("_", " ").title())
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Equality")
        plt.xlabel("Cumulative share of clients")
        plt.ylabel("Cumulative share of reward")
        plt.title("Lorenz curves for reward fairness" + (f"\nDataset={meta.get('dataset')}, attack={meta.get('attack_type')}, ratio={meta.get('malicious_ratio')}" if meta else ""))
        plt.legend(frameon=False)
        fname = "lorenz_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type", "malicious_ratio", "experiment"] if k in meta]))
        savefig(out_dir / "fairness" / fname, formats)


def plot_ledger_rates(ledger_summary: pd.DataFrame, out_dir: Path, formats: Sequence[str]) -> None:
    if ledger_summary.empty or "round" not in ledger_summary.columns:
        return
    rate_cols = [c for c in ["acceptance_rate", "accepted_malicious_rate", "rejected_malicious_rate", "accepted_honest_rate"] if c in ledger_summary.columns]
    if not rate_cols:
        return
    group_cols = [c for c in ["dataset", "attack_type", "malicious_ratio"] if c in ledger_summary.columns]
    for keys, grp in ledger_summary.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        plt.figure(figsize=(8.0, 4.8))
        stat = grp.groupby("round")[rate_cols].mean().reset_index()
        for c in rate_cols:
            if stat[c].isna().all():
                continue
            plt.plot(stat["round"], stat[c], marker="o", label=metric_label(c))
        plt.xlabel("Communication round")
        plt.ylabel("Rate")
        plt.ylim(-0.02, 1.02)
        plt.title("Consensus acceptance and rejection rates" + (f"\nDataset={meta.get('dataset')}, attack={meta.get('attack_type')}, ratio={meta.get('malicious_ratio')}" if meta else ""))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(frameon=False)
        fname = "ledger_rates_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type", "malicious_ratio"] if k in meta]))
        savefig(out_dir / "byzantine" / fname, formats)


def plot_heterodp(incentives: pd.DataFrame, out_dir: Path, formats: Sequence[str]) -> None:
    """Plot DP-noise/reward relation when client-level noise columns exist."""
    if incentives.empty:
        return
    noise_cols = [c for c in ["dp_noise_multiplier", "noise_multiplier", "client_noise", "client_noise_multiplier", "sigma_i"] if c in incentives.columns]
    reward_col = "reward_proposed" if "reward_proposed" in incentives.columns else None
    if not noise_cols or reward_col is None:
        return
    noise_col = noise_cols[0]
    df = incentives.copy()
    df[noise_col] = pd.to_numeric(df[noise_col], errors="coerce")
    df[reward_col] = pd.to_numeric(df[reward_col], errors="coerce")
    df = df[df[noise_col].notna() & df[reward_col].notna()]
    if df.empty:
        return
    group_cols = [c for c in ["dataset", "attack_type", "malicious_ratio", "experiment"] if c in df.columns]
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        by_client = grp.groupby(["client", noise_col])[reward_col].sum().reset_index()
        plt.figure(figsize=(6.8, 4.8))
        plt.scatter(by_client[noise_col], by_client[reward_col], s=45, alpha=0.8)
        # fit line when possible
        if len(by_client) >= 2 and by_client[noise_col].nunique() > 1:
            x = by_client[noise_col].to_numpy(dtype=float)
            y = by_client[reward_col].to_numpy(dtype=float)
            coef = np.polyfit(x, y, deg=1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            plt.plot(xs, coef[0] * xs + coef[1], linestyle="--", linewidth=1.5)
        plt.xlabel("DP noise multiplier")
        plt.ylabel("Total proposed reward")
        plt.title("Heterogeneous-DP fairness diagnostic" + (f"\nDataset={meta.get('dataset')}, attack={meta.get('attack_type')}, ratio={meta.get('malicious_ratio')}" if meta else ""))
        fname = "heterodp_reward_vs_noise_" + slugify("_".join([str(meta.get(k, "")) for k in ["dataset", "attack_type", "malicious_ratio", "experiment"] if k in meta]))
        savefig(out_dir / "heterogeneous_dp" / fname, formats)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compile FL simulation CSV results and create publication-ready plots.")
    ap.add_argument("--results_root", type=str, required=True, help="Root folder containing experiment subfolders.")
    ap.add_argument("--out_dir", type=str, default=None, help="Folder where compiled CSVs and plots will be saved.")
    ap.add_argument("--metric", type=str, default="acc", help="Primary metric for learning curves and final comparison plots, e.g., acc, f1, loss.")
    ap.add_argument("--formats", type=str, default="png,pdf", help="Comma-separated output formats, e.g., png,pdf or png.")
    ap.add_argument("--paper", action="store_true", help="Use compact paper-oriented plot style.")
    ap.add_argument("--no_separate_experiments", action="store_true", help="Also useful when all CSVs belong to one experiment.")
    args = ap.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if args.out_dir is None:
        out_dir = results_root / "compiled_results"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip().lower().lstrip(".") for x in args.formats.split(",") if x.strip()]
    set_plot_style(args.paper)

    print(f"[INFO] Searching results under: {results_root}")
    metrics = load_metrics(results_root)
    incentives = load_incentives(results_root)
    ledger = load_ledgers(results_root)

    if metrics.empty:
        print("[WARN] No all_trials_metrics.csv files found.")
    else:
        metrics.to_csv(out_dir / "compiled_metrics_long.csv", index=False)
        print(f"[OK] Saved compiled metrics: {out_dir / 'compiled_metrics_long.csv'} ({len(metrics)} rows)")

    if not incentives.empty:
        incentives.to_csv(out_dir / "compiled_incentives_long.csv", index=False)
        print(f"[OK] Saved compiled incentives: {out_dir / 'compiled_incentives_long.csv'} ({len(incentives)} rows)")

    if not ledger.empty:
        ledger.to_csv(out_dir / "compiled_ledger_long.csv", index=False)
        print(f"[OK] Saved compiled ledger: {out_dir / 'compiled_ledger_long.csv'} ({len(ledger)} rows)")

    # Summaries
    if not metrics.empty and args.metric in metrics.columns:
        summary = final_round_table(metrics, args.metric)
        summary.to_csv(out_dir / f"summary_final_{args.metric}.csv", index=False)
        print(f"[OK] Saved final summary: {out_dir / f'summary_final_{args.metric}.csv'}")
    else:
        summary = pd.DataFrame()
        print(f"[WARN] Metric '{args.metric}' was not found in metrics CSVs.")

    fairness = summarize_fairness(incentives)
    if not fairness.empty:
        fairness.to_csv(out_dir / "summary_fairness.csv", index=False)
        print(f"[OK] Saved fairness summary: {out_dir / 'summary_fairness.csv'}")

    ledger_summary = summarize_ledger(ledger)
    if not ledger_summary.empty:
        ledger_summary.to_csv(out_dir / "summary_ledger_rates.csv", index=False)
        print(f"[OK] Saved ledger-rate summary: {out_dir / 'summary_ledger_rates.csv'}")

    # Plots
    plot_learning_curves(metrics, out_dir, args.metric, formats, separate_experiments=not args.no_separate_experiments)
    plot_final_bars(summary, out_dir, args.metric, formats)
    plot_byzantine_robustness(summary, out_dir, args.metric, formats)
    plot_ablation(summary, out_dir, args.metric, formats)
    plot_fairness(fairness, out_dir, formats)
    plot_lorenz_curves(incentives, out_dir, formats)
    plot_ledger_rates(ledger_summary, out_dir, formats)
    plot_heterodp(incentives, out_dir, formats)

    print(f"[DONE] Plots and compiled CSVs saved in: {out_dir}")


if __name__ == "__main__":
    main()
