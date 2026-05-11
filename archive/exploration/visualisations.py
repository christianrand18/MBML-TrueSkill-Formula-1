r"""
Visualisation functions for F1 data exploration.

Produces publication‑quality figures using ``matplotlib`` and ``seaborn``.
All functions accept a ``save_dir`` parameter and write PNG files; the
calling orchestrator is responsible for creating the directory.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("f1_exploration.visualisations")

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

F1_RED = "#E10600"
F1_DARK = "#15151E"
F1_WHITE = "#FFFFFF"
PALETTE = "viridis"


def set_style() -> None:
    """Apply a consistent, clean matplotlib style."""
    sns.set_theme(style="darkgrid", context="notebook", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": F1_WHITE,
            "axes.facecolor": "#F5F5F5",
        }
    )


def _save(fig: plt.Figure, name: str, save_dir: str) -> str:
    """Save figure and close it.  Returns the file path."""
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("  ✓ saved %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Data overview
# ---------------------------------------------------------------------------


def plot_races_per_year(overview: pd.DataFrame, save_dir: str) -> str:
    """Bar chart of number of races per year."""
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        overview["year"],
        overview["n_races"],
        color=sns.color_palette(PALETTE, 1)[0],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Number of Races per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Races")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    return _save(fig, "01_races_per_year", save_dir)


def plot_participants_per_year(overview: pd.DataFrame, save_dir: str) -> str:
    """Dual‑axis line plot: unique drivers and constructors per year."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color1 = sns.color_palette("tab10")[0]
    color2 = sns.color_palette("tab10")[1]
    ax1.plot(overview["year"], overview["n_drivers"], "o-", color=color1, label="Drivers")
    ax1.plot(overview["year"], overview["n_constructors"], "s--", color=color2, label="Constructors")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Unique Participants", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.fill_between(overview["year"], overview["n_entries"], alpha=0.15, color="gray", label="Entries")
    ax2.set_ylabel("Total Entries", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    fig.suptitle("Unique Drivers, Constructors & Total Entries per Year")
    fig.tight_layout()
    return _save(fig, "02_participants_per_year", save_dir)


# ---------------------------------------------------------------------------
# 2. Drivers
# ---------------------------------------------------------------------------


def plot_top_drivers_wins(wins_df: pd.DataFrame, save_dir: str) -> str:
    """Horizontal bar chart of top drivers by wins."""
    fig, ax = plt.subplots(figsize=(10, 6))
    wins_df = wins_df.sort_values("wins")
    colors = sns.color_palette("rocket_r", len(wins_df))
    ax.barh(wins_df["driverName"], wins_df["wins"], color=colors, edgecolor="white")
    ax.set_title("Top Drivers by Race Wins (2011–2024)")
    ax.set_xlabel("Wins")
    for i, (_, row) in enumerate(wins_df.iterrows()):
        ax.text(row["wins"] + 0.5, i, str(int(row["wins"])), va="center", fontsize=9)
    return _save(fig, "03_top_drivers_wins", save_dir)


def plot_driver_careers(spans: pd.DataFrame, save_dir: str) -> str:
    """Gantt‑like bar chart of driver career spans (top 25 by races)."""
    top = spans.head(25).sort_values("first_year")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("viridis", len(top))
    for i, (_, row) in enumerate(top.iterrows()):
        ax.barh(
            row["driverName"],
            row["seasons"],
            left=row["first_year"],
            color=colors[i],
            edgecolor="white",
            height=0.7,
        )
    ax.set_xlabel("Year")
    ax.set_title("Driver Career Spans (Top 25 by Race Entries)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    return _save(fig, "04_driver_career_spans", save_dir)


# ---------------------------------------------------------------------------
# 3. Constructors
# ---------------------------------------------------------------------------


def plot_top_constructors_wins(wins_df: pd.DataFrame, save_dir: str) -> str:
    """Horizontal bar chart of top constructors by wins."""
    fig, ax = plt.subplots(figsize=(10, 6))
    wins_df = wins_df.sort_values("wins")
    colors = sns.color_palette("mako_r", len(wins_df))
    ax.barh(wins_df["constructorName"], wins_df["wins"], color=colors, edgecolor="white")
    ax.set_title("Top Constructors by Race Wins (2011–2024)")
    ax.set_xlabel("Wins")
    for i, (_, row) in enumerate(wins_df.iterrows()):
        ax.text(row["wins"] + 0.3, i, str(int(row["wins"])), va="center", fontsize=9)
    return _save(fig, "05_top_constructors_wins", save_dir)


def plot_constructor_heatmap(
    pivot: pd.DataFrame, save_dir: str
) -> str:
    """Heatmap of constructor wins per year."""
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot.T,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Wins"},
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Constructor Wins per Year (Top 8 Teams)")
    ax.set_ylabel("Constructor")
    ax.set_xlabel("Year")
    return _save(fig, "06_constructor_win_heatmap", save_dir)


# ---------------------------------------------------------------------------
# 4. Grid vs Finish
# ---------------------------------------------------------------------------


def plot_grid_vs_finish(df: pd.DataFrame, save_dir: str, sample: int = 2000) -> str:
    """Scatter plot of grid position versus finishing position with
    regression line and jitter.

    Args:
        df: Preprocessed race DataFrame.
        save_dir: Output directory.
        sample: Subsample size for readability.
    """
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 7))
    # Jitter
    jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(df))
    ax.scatter(
        df["grid"] + jitter,
        df["positionOrder"] + jitter,
        alpha=0.4,
        s=15,
        c="#E10600",
        edgecolors="none",
    )
    ax.plot([1, 25], [1, 25], "k--", alpha=0.3, label="No change")
    ax.set_xlabel("Grid Position")
    ax.set_ylabel("Finish Position")
    ax.set_title("Grid vs Finishing Position")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.legend()
    # Annotate quadrants
    ax.text(2, 23, "Overtook field", fontsize=8, color="green", alpha=0.7)
    ax.text(23, 2, "Lost positions", fontsize=8, color="red", alpha=0.7)
    return _save(fig, "07_grid_vs_finish", save_dir)


def plot_position_change_distribution(df: pd.DataFrame, save_dir: str) -> str:
    """Histogram of positions gained (grid − finish)."""
    gains = df["grid"] - df["positionOrder"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gains, bins=range(int(gains.min()), int(gains.max()) + 2), color=F1_RED, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="Net zero")
    ax.set_xlabel("Positions Gained (positive = improvement)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Position Changes (Grid → Finish)")
    ax.legend()
    return _save(fig, "08_position_change_distribution", save_dir)


# ---------------------------------------------------------------------------
# 5. Pit stops
# ---------------------------------------------------------------------------


def plot_pit_stop_trends(trends: pd.DataFrame, save_dir: str) -> str:
    """Dual‑axis time series: average pit stops and average pit duration per year."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color1 = "#E10600"
    color2 = "#1E3A5F"
    ax1.plot(trends["year"], trends["avg_stops"], "o-", color=color1, linewidth=2)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average Stops per Race", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.plot(trends["year"], trends["avg_duration_ms"] / 1000, "s--", color=color2, linewidth=2)
    ax2.set_ylabel("Average Pit Duration (seconds)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    fig.suptitle("Pit Stop Strategy Evolution (2011–2024)")
    fig.tight_layout()
    return _save(fig, "09_pit_stop_trends", save_dir)


# ---------------------------------------------------------------------------
# 6. DNF analysis
# ---------------------------------------------------------------------------


def plot_dnf_rate(dnf_rate: pd.DataFrame, save_dir: str) -> str:
    """Line plot of DNF rate per season."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(dnf_rate["year"], dnf_rate["dnf_rate"], alpha=0.2, color=F1_RED)
    ax.plot(dnf_rate["year"], dnf_rate["dnf_rate"], "o-", color=F1_RED, linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("DNF Rate")
    ax.set_title("Technical Failure / DNF Rate per Season")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    for _, row in dnf_rate.iterrows():
        ax.text(row["year"], row["dnf_rate"] + 0.005, f"{row['dnf_rate']:.1%}", ha="center", fontsize=7)
    return _save(fig, "10_dnf_rate", save_dir)


def plot_dnf_reasons(reasons: pd.DataFrame, save_dir: str) -> str:
    """Horizontal bar chart of top DNF reasons."""
    fig, ax = plt.subplots(figsize=(10, 5))
    reasons = reasons.sort_values("count")
    colors = sns.color_palette("Reds_r", len(reasons))
    ax.barh(reasons["statusName"], reasons["count"], color=colors, edgecolor="white")
    ax.set_title("Most Common Non‑Finish Reasons (2011–2024)")
    ax.set_xlabel("Occurrences")
    for i, (_, row) in enumerate(reasons.iterrows()):
        ax.text(row["count"] + 1, i, str(int(row["count"])), va="center", fontsize=8)
    return _save(fig, "11_dnf_reasons", save_dir)


# ---------------------------------------------------------------------------
# 7. Circuits
# ---------------------------------------------------------------------------


def plot_circuit_overtaking(circuit_df: pd.DataFrame, save_dir: str) -> str:
    """Horizontal bar of circuits by average overtaking (position gain)."""
    df = circuit_df.sort_values("avg_position_gain")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [F1_RED if v > 0 else "#888888" for v in df["avg_position_gain"]]
    ax.barh(df["circuitName"], df["avg_position_gain"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Average Positions Gained (positive = more overtaking)")
    ax.set_title("Circuit Overtaking Friendliness")
    return _save(fig, "12_circuit_overtaking", save_dir)


def plot_circuit_map(
    df: pd.DataFrame, circuit_names: Dict[int, str], save_dir: str
) -> str:
    """World map scatter of circuits, colour‑coded by latitude."""
    world = df.groupby("circuitId").agg(
        lat=("lat", "first"), lng=("lng", "first")
    ).reset_index()
    world["circuitName"] = world["circuitId"].map(
        lambda cid: circuit_names.get(cid, str(cid))
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sc = ax.scatter(
        world["lng"],
        world["lat"],
        c=world["lat"],
        cmap="coolwarm",
        s=80,
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )
    for _, row in world.iterrows():
        ax.annotate(
            row["circuitName"],
            (row["lng"], row["lat"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
            alpha=0.8,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("F1 Circuit Locations (2011–2024)")
    cbar = fig.colorbar(sc, ax=ax, label="Latitude")
    return _save(fig, "13_circuit_map", save_dir)


# ---------------------------------------------------------------------------
# 8. Rating trajectories  (from TrueSkill output)
# ---------------------------------------------------------------------------


def plot_driver_rating_trajectories(
    trajectories: pd.DataFrame, save_dir: str, top_n: int = 8
) -> str:
    """Line chart of mu trajectories for the top drivers."""
    fig, ax = plt.subplots(figsize=(14, 6))
    top = trajectories.groupby("driverId")["mu"].last().sort_values(ascending=False).head(top_n)
    colors = sns.color_palette("tab10", top_n)
    for idx, (d_id, _) in enumerate(top.items()):
        sub = trajectories[trajectories["driverId"] == d_id]
        name = sub["driverName"].iloc[0]
        ax.plot(sub["date"], sub["mu"], linewidth=1.2, color=colors[idx], label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Skill (mu)")
    ax.set_title(f"Top {top_n} Driver Skill Trajectories (TrueSkill)")
    ax.legend(fontsize=9, loc="upper left")
    return _save(fig, "14_driver_rating_trajectories", save_dir)


def plot_sigma_vs_races(final: pd.DataFrame, save_dir: str) -> str:
    """Scatter plot of final sigma vs number of races for each driver."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(final["n_races"], final["sigma_final"], alpha=0.7, s=40, c=F1_RED, edgecolors="white")
    ax.set_xlabel("Number of Races")
    ax.set_ylabel("Final Sigma (uncertainty)")
    ax.set_title("Uncertainty Reduction: Sigma vs Career Races")
    # Fit a trend
    try:
        z = np.polyfit(final["n_races"], final["sigma_final"], 1)
        x_line = np.linspace(final["n_races"].min(), final["n_races"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.6, label="Trend")
        ax.legend()
    except Exception:
        pass
    # Annotate outliers
    for _, row in final.iterrows():
        if row["sigma_final"] > 1.5 or row["n_races"] < 5:
            ax.annotate(
                row["driverName"],
                (row["n_races"], row["sigma_final"]),
                fontsize=6,
                alpha=0.7,
            )
    return _save(fig, "15_sigma_vs_races", save_dir)


# ---------------------------------------------------------------------------
# 9. Teammate comparison
# ---------------------------------------------------------------------------


def plot_teammate_comparison(
    teammate_df: pd.DataFrame,
    constructor_map: Dict[int, str],
    driver_map: Dict[int, str],
    save_dir: str,
    top_n: int = 15,
) -> str:
    """Horizontal bar of the most dominant teammate match‑ups."""
    df = teammate_df.sort_values("avg_finish_delta").tail(top_n)
    df = df.copy()
    df["label"] = df.apply(
        lambda r: f"{driver_map.get(r['driver_ahead'], str(r['driver_ahead']))} "
        f"(ahead in {constructor_map.get(r['constructorId'], str(r['constructorId']))})",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("rocket_r", len(df))
    ax.barh(df["label"], df["avg_finish_delta"], color=colors, edgecolor="white")
    ax.set_xlabel("Average Finishing Positions Ahead of Teammate")
    ax.set_title(f"Most Dominant Teammate Pairings (Top {top_n})")
    return _save(fig, "16_teammate_comparison", save_dir)
