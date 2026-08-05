#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_separate_run_results.py
---------------------------------
Compile and plot FL simulation results when:
  1) benchmark methods were run together in one folder, and
  2) the adaptive/proposed method was run separately in another folder, and
  3) ablation variants may also be stored in a separate folder.

Expected files inside each result folder:
  - all_trials_metrics.csv
  - config_parameters.csv  (optional, but recommended)

Typical usage:
  python plot_separate_run_results.py \
      --adaptive_dir results1/organsmnist/trial3c \
      --benchmark_dir results1/organsmnist/trial2 \
      --out_dir plots1 \
      --smooth rolling \
      --window 5

The generated line plots use different markers and line styles for each method.
Final bar plots are skipped for accuracy by default.

With ablation:
  python plot_separate_run_results.py \
      --adaptive_dir results1/organsmnist/trial3c \
      --benchmark_dir results1/organsmnist/trial2 \
      --ablation_dir results1/organsmnist/ablation1 \
      --out_dir plots1 \
      --smooth ema
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Editable default paths, matching your example style
# -----------------------------------------------------------------------------
ADAPTIVE_DIR = None #"results1/organsmnist/trial3c"
BENCHMARK_DIR = None #"results1/organsmnist/trial2"
ABLATION_DIR = None  # e.g., "results1/organsmnist/ablation1"
OUT_DIR = "plots1"


# -----------------------------------------------------------------------------
# Plot appearance
# -----------------------------------------------------------------------------
def setup_matplotlib(font_size: int = 20, paper: bool = True) -> None:
    """Set paper-friendly matplotlib defaults."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["axes.linewidth"] = 1.2
    matplotlib.rcParams["xtick.major.width"] = 1.2
    matplotlib.rcParams["ytick.major.width"] = 1.2
    matplotlib.rcParams["legend.frameon"] = True
    matplotlib.rcParams["legend.framealpha"] = 0.92
    matplotlib.rcParams["legend.edgecolor"] = "0.75"
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["axes.grid"] = False
    matplotlib.rcParams["xtick.direction"] = "out"
    matplotlib.rcParams["ytick.direction"] = "out"
    plt.rcParams.update({"font.size": font_size})
    if paper:
        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["savefig.dpi"] = 300


METHOD_LABELS = {
    "adaptive": "Proposed",
    "proposed": "Proposed",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedsgd": "FedSGD",
    "krum": "Krum",
    "multikrum": "Multi-Krum",
    "multi_krum": "Multi-Krum",
    "median": "Median",
    "coordinate_median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "trimmedmean": "Trimmed Mean",
    "bulyan": "Bulyan",
    "adaptive_no_consensus": "No consensus",
    "adaptive_consensus_only": "Consensus only",
    "adaptive_quality_only": "Quality only",
    "adaptive_shapley_only": "Shapley only",
    "adaptive_no_quality": "No quality weighting",
    "adaptive_no_incentive": "No incentive",
}

METHOD_ORDER = [
    "adaptive",
    "fedavg",
    "fedprox",
    "fedsgd",
    "krum",
    "multikrum",
    "median",
    "trimmed_mean",
    "bulyan",
    "adaptive_no_consensus",
    "adaptive_consensus_only",
    "adaptive_quality_only",
    "adaptive_shapley_only",
    "adaptive_no_quality",
    "adaptive_no_incentive",
]

# Distinct markers and line styles make the curves readable even in grayscale.
METHOD_MARKERS = {
    "adaptive": "o",
    "fedavg": "s",
    "fedprox": "^",
    "fedsgd": "D",
    "krum": "v",
    "multikrum": "P",
    "median": "X",
    "trimmed_mean": "*",
    "bulyan": "h",
    "adaptive_no_consensus": "<",
    "adaptive_consensus_only": ">",
    "adaptive_quality_only": "p",
    "adaptive_shapley_only": "8",
    "adaptive_no_quality": "d",
    "adaptive_no_incentive": "H",
}

METHOD_LINESTYLES = {
    "adaptive": "-",
    "fedavg": "--",
    "fedprox": "-.",
    "fedsgd": ":",
    "krum": "--",
    "multikrum": "-.",
    "median": ":",
    "trimmed_mean": "--",
    "bulyan": "-.",
    "adaptive_no_consensus": "--",
    "adaptive_consensus_only": "-.",
    "adaptive_quality_only": ":",
    "adaptive_shapley_only": "--",
    "adaptive_no_quality": "-.",
    "adaptive_no_incentive": ":",
}

MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "8", "d", "H"]
LINESTYLE_CYCLE = ["-", "--", "-.", ":"]

METRIC_LABELS = {
    "acc": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "loss": "Loss",
    "test_acc": "Accuracy",
    "test_loss": "Loss",
    "malicious_reward_share": "Malicious reward share",
    "accepted_malicious_rate": "Accepted malicious rate",
    "rejected_malicious_rate": "Rejected malicious rate",
    "accepted_honest_rate": "Accepted honest rate",
    "reward_gini": "Reward Gini coefficient",
    "reward_contribution_pearson": "Reward-contribution Pearson correlation",
    "reward_contribution_spearman": "Reward-contribution Spearman correlation",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def sanitize(s: object) -> str:
    """Make a value safe for filenames."""
    text = str(s)
    text = text.replace(".", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_method_name(method: object) -> str:
    m = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "multi_krum": "multikrum",
        "trimmedmean": "trimmed_mean",
        "coordinate_wise_median": "median",
        "coord_median": "median",
        "proposed": "adaptive",
    }
    return aliases.get(m, m)


def display_method(method: object) -> str:
    m = normalize_method_name(method)
    return METHOD_LABELS.get(m, str(method))


def method_sort_key(method: object) -> Tuple[int, str]:
    m = normalize_method_name(method)
    try:
        return METHOD_ORDER.index(m), m
    except ValueError:
        return len(METHOD_ORDER), m


def marker_for_method(method: object, index: int = 0) -> str:
    m = normalize_method_name(method)
    return METHOD_MARKERS.get(m, MARKER_CYCLE[index % len(MARKER_CYCLE)])


def linestyle_for_method(method: object, index: int = 0) -> str:
    m = normalize_method_name(method)
    return METHOD_LINESTYLES.get(m, LINESTYLE_CYCLE[index % len(LINESTYLE_CYCLE)])


def read_config(folder: Path) -> Dict[str, object]:
    path = folder / "config_parameters.csv"
    if not path.exists():
        return {}
    try:
        cfg = pd.read_csv(path)
        if cfg.empty:
            return {}
        return {col: cfg[col].iloc[0] for col in cfg.columns}
    except Exception as exc:
        print(f"Warning: could not read config file {path}: {exc}")
        return {}


def get_experiment_tag(config: Dict[str, object], fallback_folder: Path) -> Dict[str, str]:
    dataset = sanitize(config.get("dataset", fallback_folder.name))
    alpha = sanitize(config.get("dirichlet_alpha", "na"))
    base_noise = sanitize(config.get("base_noise", "na"))
    attack_type = sanitize(config.get("attack_type", "none"))
    malicious_ratio = sanitize(config.get("malicious_ratio", "0"))
    return {
        "dataset": dataset,
        "alpha": alpha,
        "base_noise": base_noise,
        "attack_type": attack_type,
        "malicious_ratio": malicious_ratio,
    }


def load_metrics_from_folder(folder: str | Path, source_label: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    folder = Path(folder)
    csv_path = folder / "all_trials_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Metrics file is empty: {csv_path}")

    if "method" not in df.columns:
        df["method"] = source_label

    if "round" not in df.columns:
        # Some outputs use zero-based row order; keep a safe fallback.
        df["round"] = np.arange(1, len(df) + 1)

    # Optional trial column. If not present, create one.
    if "trial" not in df.columns:
        df["trial"] = 0

    df["method"] = df["method"].apply(normalize_method_name)
    df["method_label"] = df["method"].apply(display_method)
    df["source"] = source_label
    df["source_folder"] = str(folder)
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"])
    df["round"] = df["round"].astype(int)

    cfg = read_config(folder)
    tags = get_experiment_tag(cfg, folder)
    for key, value in tags.items():
        if key not in df.columns:
            df[key] = value

    return df, cfg


def combine_runs(adaptive_dir: str, benchmark_dir: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    adaptive_df, adaptive_cfg = load_metrics_from_folder(adaptive_dir, "adaptive")
    benchmark_df, benchmark_cfg = load_metrics_from_folder(benchmark_dir, "benchmark")

    combined = pd.concat([benchmark_df, adaptive_df], ignore_index=True)
    combined = combined.sort_values(by=["method", "trial", "round"]).reset_index(drop=True)

    # Prefer adaptive config because it is the proposed method run.
    cfg = dict(benchmark_cfg)
    cfg.update(adaptive_cfg)
    return combined, cfg


def available_metrics(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    out = []
    for m in requested:
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
            out.append(m)
        else:
            print(f"Skipping metric '{m}' because it is not available or not numeric.")
    return out


def aggregate_by_round(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_cols = ["method", "method_label", "round"]
    agg = (
        df.groupby(group_cols, as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def smooth_values(values: Sequence[float], smooth: str = "rolling", window: int = 5, ema_alpha: float = 0.25) -> np.ndarray:
    y = pd.Series(values, dtype="float64")
    if smooth == "none" or len(y) <= 1:
        return y.to_numpy()
    if smooth == "rolling":
        return y.rolling(window=max(1, window), min_periods=1, center=False).mean().to_numpy()
    if smooth == "rolling_center":
        return y.rolling(window=max(1, window), min_periods=1, center=True).mean().to_numpy()
    if smooth == "ema":
        return y.ewm(alpha=float(ema_alpha), adjust=False).mean().to_numpy()
    if smooth == "savgol":
        try:
            from scipy.signal import savgol_filter
            w = max(3, int(window))
            if w % 2 == 0:
                w += 1
            if w >= len(y):
                w = len(y) if len(y) % 2 == 1 else len(y) - 1
            if w < 3:
                return y.to_numpy()
            return savgol_filter(y.to_numpy(), window_length=w, polyorder=min(2, w - 1))
        except Exception:
            print("Warning: scipy is not available or Savitzky-Golay failed. Falling back to rolling smoothing.")
            return y.rolling(window=max(1, window), min_periods=1).mean().to_numpy()
    raise ValueError(f"Unknown smoothing method: {smooth}")


def metric_y_limits(metric: str, values: np.ndarray) -> Optional[Tuple[float, float]]:
    if metric in {"acc", "auc", "precision", "recall", "f1", "test_acc"}:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return 0.0, 1.0
        ymin = max(0.0, float(np.nanmin(finite)) - 0.05)
        ymax = min(1.0, float(np.nanmax(finite)) + 0.05)
        if ymax - ymin < 0.1:
            ymax = min(1.0, ymax + 0.05)
            ymin = max(0.0, ymin - 0.05)
        return ymin, ymax
    return None


def plot_metric_curves(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: Optional[str] = None,
    smooth: str = "rolling",
    window: int = 5,
    ema_alpha: float = 0.25,
    max_round: Optional[int] = None,
    min_round: Optional[int] = None,
    show_std: bool = True,
    marker_every_points: int = 8,
) -> pd.DataFrame:
    data = df.copy()
    if min_round is not None:
        data = data[data["round"] >= int(min_round)]
    if max_round is not None:
        data = data[data["round"] <= int(max_round)]

    agg = aggregate_by_round(data, metric)
    if agg.empty:
        print(f"No data to plot for metric {metric}")
        return agg

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    summary_rows = []
    all_smoothed = []

    methods = sorted(agg["method"].unique(), key=method_sort_key)
    for method_index, method in enumerate(methods):
        subset = agg[agg["method"] == method].sort_values("round")
        x = subset["round"].to_numpy()
        y_raw = subset["mean"].to_numpy(dtype=float)
        y_smooth = smooth_values(y_raw, smooth=smooth, window=window, ema_alpha=ema_alpha)
        all_smoothed.append(y_smooth)

        label = display_method(method)
        marker = marker_for_method(method, method_index)
        linestyle = linestyle_for_method(method, method_index)
        markevery = max(1, len(x) // max(1, int(marker_every_points)))
        line, = ax.plot(
            x,
            y_smooth,
            label=label,
            linewidth=2.3,
            # linestyle=linestyle,
            marker=marker,
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=1.4,
            markevery=markevery,
            alpha=0.98,
        )

        if show_std and subset["count"].max() > 1:
            std = subset["std"].to_numpy(dtype=float)
            ax.fill_between(x, y_smooth - std, y_smooth + std, alpha=0.12, color=line.get_color(), linewidth=0)

        final_idx = -1
        if len(y_raw) > 0:
            if metric == "loss":
                best_idx = int(np.nanargmin(y_raw))
            else:
                best_idx = int(np.nanargmax(y_raw))
            summary_rows.append({
                "metric": metric,
                "method": method,
                "method_label": label,
                "final_round": int(x[final_idx]),
                "final_raw": float(y_raw[final_idx]),
                "final_smoothed": float(y_smooth[final_idx]),
                "best_round": int(x[best_idx]),
                "best_raw": float(y_raw[best_idx]),
            })

    ylabel = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # ax.grid(False, axis="y", alpha=0.25, linewidth=0.8)
    # ax.grid(False, axis="x", alpha=0.10, linewidth=0.6)
    legend = ax.legend(loc="best", fontsize=18, ncol=1, handlelength=2.8, borderpad=0.6, labelspacing=0.35,framealpha=0.5)
    legend.get_frame().set_linewidth(0.8)

    if max_round is not None:
        ax.set_xlim(left=min_round if min_round is not None else 1, right=max_round)
    else:
        ax.set_xlim(left=max(1, int(agg["round"].min())))

    if all_smoothed:
        concat = np.concatenate(all_smoothed)
        limits = metric_y_limits(metric, concat)
        if limits is not None:
            ax.set_ylim(*limits)

    ax.tick_params(axis="both", which="major", length=5, width=1.1)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(summary_rows)


def plot_final_bar(
    summary: pd.DataFrame,
    metric: str,
    out_path: Path,
    use_smoothed: bool = True,
) -> None:
    if summary.empty:
        return
    sub = summary[summary["metric"] == metric].copy()
    if sub.empty:
        return
    sub["sort_key"] = sub["method"].apply(method_sort_key)
    sub = sub.sort_values("sort_key")
    value_col = "final_smoothed" if use_smoothed else "final_raw"

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    labels = sub["method_label"].tolist()
    values = sub[value_col].to_numpy(dtype=float)
    x = np.arange(len(labels))
    ax.bar(x, values, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)

    if metric in {"acc", "auc", "precision", "recall", "f1", "test_acc"}:
        ax.set_ylim(0, min(1.0, max(0.1, float(np.nanmax(values)) + 0.1)))

    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def save_method_metric_wide(summary: pd.DataFrame, path: Path, value_col: str = "final_smoothed") -> None:
    if summary.empty:
        return
    wide = summary.pivot_table(index="method_label", columns="metric", values=value_col, aggfunc="first")
    wide = wide.reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(path, index=False)


def run_comparison_block(
    name: str,
    df: pd.DataFrame,
    config: Dict[str, object],
    out_root: Path,
    metrics: Sequence[str],
    smooth: str,
    window: int,
    ema_alpha: float,
    max_round: Optional[int],
    min_round: Optional[int],
    show_std: bool,
    skip_bar_metrics: Sequence[str],
    no_final_bars: bool,
    marker_every_points: int,
) -> pd.DataFrame:
    fallback_folder = Path(str(df["source_folder"].iloc[0])) if "source_folder" in df.columns and len(df) else Path("results")
    tags = get_experiment_tag(config, fallback_folder)
    dataset_dir = out_root / tags["dataset"] /name/f"alpha_{tags['alpha']}_dpnoise_{tags['base_noise']}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(dataset_dir / f"combined_{name}.csv", index=False)

    summary_parts = []
    for metric in available_metrics(df, metrics):
        filename = f"{tags['dataset']}_{name}_{metric}_alpha_{tags['alpha']}_dpnoise_{tags['base_noise']}"
        if tags.get("attack_type", "none") != "none" or tags.get("malicious_ratio", "0") not in {"0", "0_0"}:
            filename += f"_attack_{tags['attack_type']}_mal_{tags['malicious_ratio']}"
        title = None  # keep manuscript plots clean; set a string if you want titles.
        summary = plot_metric_curves(
            df=df,
            metric=metric,
            out_path=dataset_dir / filename,
            title=title,
            smooth=smooth,
            window=window,
            ema_alpha=ema_alpha,
            max_round=max_round,
            min_round=min_round,
            show_std=show_std,
            marker_every_points=marker_every_points,
        )
        if not summary.empty:
            summary_parts.append(summary)
            if (not no_final_bars) and (metric not in set(skip_bar_metrics)):
                plot_final_bar(
                    summary=summary,
                    metric=metric,
                    out_path=dataset_dir / f"{filename}_final_bar",
                    use_smoothed=True,
                )

    if summary_parts:
        final_summary = pd.concat(summary_parts, ignore_index=True)
    else:
        final_summary = pd.DataFrame()

    final_summary.to_csv(dataset_dir / f"summary_{name}.csv", index=False)
    save_method_metric_wide(final_summary, dataset_dir / f"summary_{name}_wide.csv")
    print(f"Saved {name} plots and summaries to: {dataset_dir}")
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adaptive, benchmark, and ablation results from separate folders.")
    parser.add_argument("--adaptive_dir", default=ADAPTIVE_DIR, help="Folder containing adaptive all_trials_metrics.csv")
    parser.add_argument("--benchmark_dir", default=BENCHMARK_DIR, help="Folder containing benchmark all_trials_metrics.csv")
    parser.add_argument("--ablation_dir", default=ABLATION_DIR, help="Optional folder containing ablation all_trials_metrics.csv")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Output folder for plots and compiled CSVs")
    parser.add_argument("--metrics", nargs="+", default=["acc", "auc", "precision", "recall", "f1", "loss"], help="Metrics to plot")
    parser.add_argument("--smooth", choices=["none", "rolling", "rolling_center", "ema", "savgol"], default="rolling", help="Smoothing method")
    parser.add_argument("--window", type=int, default=5, help="Rolling/Savitzky-Golay smoothing window")
    parser.add_argument("--ema_alpha", type=float, default=0.25, help="EMA smoothing alpha")
    parser.add_argument("--max_round", type=int, default=50, help="Optional maximum round for x-axis and filtering")
    parser.add_argument("--min_round", type=int, default=None, help="Optional minimum round for filtering")
    parser.add_argument("--font_size", type=int, default=24, help="Plot font size")
    parser.add_argument("--no_std", action="store_true", help="Disable standard deviation bands when multiple trials exist")
    parser.add_argument("--skip_bar_metrics", nargs="+", default=["acc", "test_acc"], help="Metrics for which final bar plots should not be generated. Default skips accuracy bars.")
    parser.add_argument("--no_final_bars", action="store_true", help="Disable all final bar plots and only save line plots.")
    parser.add_argument("--marker_every_points", type=int, default=50, help="Approximate number of visible markers per line.")
    parser.add_argument("--paper", action="store_true", help="Use paper-ready matplotlib settings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib(font_size=args.font_size, paper=args.paper)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Section 1: Proposed/adaptive vs benchmarks
    # ------------------------------------------------------------------
    comparison_df, comparison_cfg = combine_runs(args.adaptive_dir, args.benchmark_dir)
    comparison_summary = run_comparison_block(
        name="adaptive_vs_benchmarks",
        df=comparison_df,
        config=comparison_cfg,
        out_root=out_root,
        metrics=args.metrics,
        smooth=args.smooth,
        window=args.window,
        ema_alpha=args.ema_alpha,
        max_round=args.max_round,
        min_round=args.min_round,
        show_std=not args.no_std,
        skip_bar_metrics=args.skip_bar_metrics,
        no_final_bars=args.no_final_bars,
        marker_every_points=args.marker_every_points,
    )

    # ------------------------------------------------------------------
    # Section 2: Proposed/adaptive vs ablation variants
    # ------------------------------------------------------------------
    if args.ablation_dir:
        adaptive_df, adaptive_cfg = load_metrics_from_folder(args.adaptive_dir, "adaptive")
        ablation_df, ablation_cfg = load_metrics_from_folder(args.ablation_dir, "ablation")

        # Avoid duplicate full adaptive if the ablation file already includes it.
        if "adaptive" in set(ablation_df["method"].unique()):
            ablation_combined = ablation_df.copy()
            ablation_cfg_final = ablation_cfg
        else:
            ablation_combined = pd.concat([adaptive_df, ablation_df], ignore_index=True)
            ablation_cfg_final = dict(ablation_cfg)
            ablation_cfg_final.update(adaptive_cfg)

        run_comparison_block(
            name="adaptive_vs_ablation",
            df=ablation_combined,
            config=ablation_cfg_final,
            out_root=out_root,
            metrics=args.metrics,
            smooth=args.smooth,
            window=args.window,
            ema_alpha=args.ema_alpha,
            max_round=args.max_round,
            min_round=args.min_round,
            show_std=not args.no_std,
            skip_bar_metrics=args.skip_bar_metrics,
            no_final_bars=args.no_final_bars,
            marker_every_points=args.marker_every_points,
        )

    print("Done.")
    print(f"Adaptive vs benchmark summary rows: {len(comparison_summary)}")
    print(f"All plots saved under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
