r"""
Evaluation reporting: figures and markdown summary for model comparison.
"""

from __future__ import annotations

import logging
import os
import textwrap
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("f1_evaluation.reporter")

# Colour palette for models
MODEL_COLORS: Dict[str, str] = {
    "TrueSkill": "#E10600",   # F1 red
    "Grid": "#1E3A5F",        # dark blue
    "Elo": "#FFA500",         # orange
    "PrevSeason": "#2E8B57",  # sea green
    "Random": "#888888",      # grey
}

METRIC_LABELS: Dict[str, str] = {
    "pairwise_accuracy": "Pairwise Accuracy",
    "top_1_accuracy": "Top-1 (Winner) Accuracy",
    "top_3_accuracy": "Top-3 (Podium) Accuracy",
    "top_5_accuracy": "Top-5 Accuracy",
    "spearman_rho": "Spearman rho",
    "mrr": "Mean Reciprocal Rank",
    "mse_position": "MSE (finish position)",
}


def set_style() -> None:
    """Apply consistent figure styling."""
    sns.set_theme(style="darkgrid", context="notebook", font_scale=1.0)


def _save(fig: plt.Figure, name: str, save_dir: str) -> str:
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("  ✓ saved %s", path)
    return path


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------


def plot_model_comparison(
    metrics_df: pd.DataFrame, metric: str, save_dir: str
) -> str:
    """Bar chart showing mean ± std of *metric* across folds per model.

    Args:
        metrics_df: Long‑form metrics DataFrame from ``ChronologicalValidator.run``.
        metric: Column name (e.g. ``"pairwise_accuracy"``).
        save_dir: Output directory.
    """
    agg = metrics_df.groupby("model")[metric].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=False)
    label = METRIC_LABELS.get(metric, metric)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [MODEL_COLORS.get(m, "#888") for m in agg["model"]]
    bars = ax.bar(agg["model"], agg["mean"], yerr=agg["std"], color=colors,
                  edgecolor="white", capsize=5, width=0.55)
    ax.set_title(f"Model Comparison: {label}")
    ax.set_ylabel(label)
    for bar, mean in zip(bars, agg["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    return _save(fig, f"model_comparison_{metric}", save_dir)


# ---------------------------------------------------------------------------
# Fold‑consistency line plot
# ---------------------------------------------------------------------------


def plot_fold_consistency(
    metrics_df: pd.DataFrame, metric: str, save_dir: str
) -> str:
    """Line plot of *metric* across test years for each model.

    Args:
        metrics_df: Long‑form metrics DataFrame.
        metric: Metric column.
        save_dir: Output directory.
    """
    label = METRIC_LABELS.get(metric, metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    for model_name in sorted(metrics_df["model"].unique()):
        sub = metrics_df[metrics_df["model"] == model_name]
        color = MODEL_COLORS.get(model_name, "#888")
        ax.plot(
            sub["fold_test_year"],
            sub[metric],
            "o-",
            color=color,
            label=model_name,
            linewidth=1.5,
            markersize=5,
        )
    ax.set_xlabel("Test Year (held‑out season)")
    ax.set_ylabel(label)
    ax.set_title(f"Fold Consistency: {label} over Time")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _save(fig, f"fold_consistency_{metric}", save_dir)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def compute_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return mean ± std for each model across all folds.

    Columns: model, then mean_pairwise_accuracy, std_pairwise_accuracy, …
    """
    agg = metrics_df.groupby("model").agg(["mean", "std"]).reset_index()
    # Flatten MultiIndex columns
    agg.columns = ["model"] + [
        f"{col[0]}_{col[1]}" for col in agg.columns[1:]
    ]
    return agg


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_report(
    summary: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: str,
) -> str:
    """Write a ``validation_report.md`` summarising all findings.

    Args:
        summary: Per‑model mean ± std DataFrame.
        metrics_df: Raw fold‑level metrics.
        output_dir: Directory for the report file.

    Returns:
        Path to the generated report.
    """
    # Rank models by pairwise accuracy (the most discriminative metric)
    if "pairwise_accuracy_mean" in summary.columns:
        ranked = summary.sort_values("pairwise_accuracy_mean", ascending=False)
    else:
        ranked = summary

    lines: List[str] = []
    lines.append("# F1 TrueSkill Model Validation Report")
    lines.append("")
    lines.append(
        "Chronological cross‑validation over **10 seasons** (2015–2024), "
        "training on all prior years.  Five models evaluated head‑to‑head."
    )
    lines.append("")

    # --- Summary table ---
    lines.append("## Model Performance Summary")
    lines.append("")
    lines.append("| Model | Pairwise Acc | Top‑1 Win | Top‑3 Podium | Spearman ρ | MRR | MSE ↓ |")
    lines.append("|-------|-------------|-----------|-------------|------------|-----|-------|")

    for _, row in ranked.iterrows():
        lines.append(
            f"| **{row['model']}** "
            f"| {row.get('pairwise_accuracy_mean', 0):.3f} ± {row.get('pairwise_accuracy_std', 0):.3f} "
            f"| {row.get('top_1_accuracy_mean', 0):.3f} "
            f"| {row.get('top_3_accuracy_mean', 0):.3f} "
            f"| {row.get('spearman_rho_mean', 0):.3f} "
            f"| {row.get('mrr_mean', 0):.3f} "
            f"| {row.get('mse_position_mean', 0):.1f} "
            f"|"
        )

    lines.append("")

    # --- Per‑fold breakdown ---
    lines.append("## Per‑Fold Breakdown (Top‑1 Accuracy)")
    lines.append("")
    lines.append("| Test Year | " + " | ".join(sorted(metrics_df["model"].unique())) + " |")
    lines.append("|" + "|".join(["-----------"] * (metrics_df["model"].nunique() + 1)) + "|")

    for year in sorted(metrics_df["fold_test_year"].unique()):
        row_data = [str(int(year))]
        for model in sorted(metrics_df["model"].unique()):
            sub = metrics_df[
                (metrics_df["model"] == model) & (metrics_df["fold_test_year"] == year)
            ]
            val = sub["top_1_accuracy"].values[0] if len(sub) else 0
            row_data.append(f"{val:.3f}")
        lines.append("| " + " | ".join(row_data) + " |")

    lines.append("")

    # --- Key findings ---
    lines.append("## Key Findings")
    lines.append("")

    # Find best model
    if not ranked.empty:
        best = ranked.iloc[0]
        lines.append(
            f"- **{best['model']}** achieves the highest pairwise accuracy "
            f"({best.get('pairwise_accuracy_mean', 0):.3f})."
        )

    # Grid baseline context
    if "Grid" in ranked["model"].values:
        grid_row = ranked[ranked["model"] == "Grid"].iloc[0]
        lines.append(
            f"- The **Grid** baseline achieves {grid_row.get('pairwise_accuracy_mean', 0):.3f} "
            f"pairwise accuracy, confirming that qualifying pace is a strong predictor."
        )

    # Compare TrueSkill vs baselines
    ts_rows = ranked[ranked["model"] == "TrueSkill"]
    if not ts_rows.empty:
        ts_row = ts_rows.iloc[0]
        lines.append(
            f"- **TrueSkill** (pairwise acc {ts_row.get('pairwise_accuracy_mean', 0):.3f}) "
            f"provides full posterior uncertainty estimates (σ) in addition to point "
            f"predictions (μ) — a key advantage over point‑estimate baselines."
        )

    lines.append("")

    # --- Figures ---
    lines.append("## Generated Figures")
    lines.append("")
    for fig_name in sorted(os.listdir(output_dir)):
        if fig_name.endswith(".png"):
            lines.append(f"- `{fig_name}`")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated from {len(metrics_df)} fold‑model combinations.*")

    report_text = "\n".join(lines)
    path = os.path.join(output_dir, "validation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Validation report written to %s", path)
    return path
