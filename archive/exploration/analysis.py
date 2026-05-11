r"""
Statistical analysis functions for F1 data exploration.

Computes summary statistics, distributions, and derived metrics from the
preprocessed race DataFrame and auxiliary source CSVs.  All functions
return ``pd.DataFrame`` objects suitable for plotting or tabular output.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_exploration.analysis")

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

NA_VALUES = [r"\N"]


def load_auxiliary_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load supplementary CSV tables for name lookups and context.

    Args:
        data_dir: Path to the ``data/`` folder.

    Returns:
        Dictionary with keys ``drivers``, ``constructors``, ``races``,
        ``status``, ``circuits``.
    """
    tables: Dict[str, pd.DataFrame] = {}
    for name in ["drivers", "constructors", "races", "status", "circuits"]:
        path = f"{data_dir}/{name}.csv"
        try:
            tables[name] = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)
        except FileNotFoundError:
            logger.warning("Auxiliary table %s.csv not found.", name)
            tables[name] = pd.DataFrame()
    return tables


# ---------------------------------------------------------------------------
# Data overview
# ---------------------------------------------------------------------------


def compute_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return per‑year counts of races, entries, drivers, and constructors.

    Args:
        df: Preprocessed race DataFrame.

    Returns:
        DataFrame with columns ``year``, ``n_races``, ``n_entries``,
        ``n_drivers``, ``n_constructors``.
    """
    grouped = df.groupby("year").agg(
        n_races=("raceId", "nunique"),
        n_entries=("raceId", "count"),
        n_drivers=("driverId", "nunique"),
        n_constructors=("constructorId", "nunique"),
    ).reset_index()
    groups_series = grouped.sort_values("year")
    return groups_series


# ---------------------------------------------------------------------------
# Driver analysis
# ---------------------------------------------------------------------------


def top_drivers_by_wins(
    df: pd.DataFrame, driver_map: Dict[int, str], top_n: int = 15
) -> pd.DataFrame:
    """Rank drivers by number of race wins (``positionOrder == 1``).

    Args:
        df: Preprocessed race DataFrame.
        driver_map: Mapping from ``driverId`` to ``"forename surname"``.
        top_n: Number of drivers to return.

    Returns:
        DataFrame with ``driverId``, ``driverName``, ``wins``, sorted
        descending.
    """
    wins = df[df["positionOrder"] == 1].groupby("driverId").size()
    wins = wins.sort_values(ascending=False).head(top_n).reset_index(name="wins")
    wins["driverName"] = wins["driverId"].map(
        lambda did: driver_map.get(did, f"Driver_{did}")
    )
    return wins[["driverId", "driverName", "wins"]]


def podium_counts(
    df: pd.DataFrame, driver_map: Dict[int, str], top_n: int = 15
) -> pd.DataFrame:
    """Rank drivers by total podium finishes (``positionOrder <= 3``).

    Args:
        df: Preprocessed race DataFrame.
        driver_map: Name map.
        top_n: Number of drivers.

    Returns:
        DataFrame with ``driverId``, ``driverName``, ``podiums``.
    """
    podiums = df[df["positionOrder"] <= 3].groupby("driverId").size()
    podiums = (
        podiums.sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="podiums")
    )
    podiums["driverName"] = podiums["driverId"].map(
        lambda did: driver_map.get(did, f"Driver_{did}")
    )
    return podiums[["driverId", "driverName", "podiums"]]


def driver_career_spans(
    df: pd.DataFrame, driver_map: Dict[int, str]
) -> pd.DataFrame:
    """Compute career length (first–last year) and total races per driver.

    Args:
        df: Preprocessed race DataFrame.
        driver_map: Name map.

    Returns:
        DataFrame with ``driverId``, ``driverName``, ``first_year``,
        ``last_year``, ``seasons``, ``total_races``.
    """
    spans = df.groupby("driverId").agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        total_races=("raceId", "count"),
    )
    spans["seasons"] = spans["last_year"] - spans["first_year"] + 1
    spans["driverName"] = spans.index.map(
        lambda did: driver_map.get(did, f"Driver_{did}")
    )
    return (
        spans.reset_index()
        .sort_values("total_races", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Constructor analysis
# ---------------------------------------------------------------------------


def top_constructors_by_wins(
    df: pd.DataFrame, constructor_map: Dict[int, str], top_n: int = 15
) -> pd.DataFrame:
    """Rank constructors by race wins.

    Args:
        df: Preprocessed race DataFrame.
        constructor_map: ``constructorId`` → team name.
        top_n: Number to return.

    Returns:
        DataFrame with ``constructorId``, ``constructorName``, ``wins``.
    """
    wins = df[df["positionOrder"] == 1].groupby("constructorId").size()
    wins = wins.sort_values(ascending=False).head(top_n).reset_index(name="wins")
    wins["constructorName"] = wins["constructorId"].map(
        lambda cid: constructor_map.get(cid, f"Constructor_{cid}")
    )
    return wins[["constructorId", "constructorName", "wins"]]


def constructor_wins_per_year(
    df: pd.DataFrame, constructor_map: Dict[int, str]
) -> pd.DataFrame:
    """Pivot table of constructor wins by year.

    Args:
        df: Preprocessed race DataFrame.
        constructor_map: Name map.

    Returns:
        DataFrame with year as index and top constructor columns (top 8).
    """
    wins_df = df[df["positionOrder"] == 1].copy()
    wins_df["constructorName"] = wins_df["constructorId"].map(constructor_map)
    pivot = wins_df.pivot_table(
        index="year", columns="constructorName", values="raceId", aggfunc="count"
    ).fillna(0)
    # Keep only top 8 constructors (by total wins)
    top_cons = pivot.sum().sort_values(ascending=False).head(8).index.tolist()
    return pivot[top_cons].astype(int)


def constructor_dominance(
    df: pd.DataFrame, constructor_map: Dict[int, str]
) -> pd.DataFrame:
    """Yearly win share for each constructor.

    Returns a long‑form DataFrame with ``year``, ``constructorName``,
    ``wins``, ``total_races``, ``win_share``.
    """
    yearly_races = df.groupby("year")["raceId"].nunique().rename("total_races")
    wins = df[df["positionOrder"] == 1].copy()
    wins["constructorName"] = wins["constructorId"].map(constructor_map)
    yearly_wins = wins.groupby(["year", "constructorName"]).size().rename("wins")
    result = yearly_wins.reset_index()
    result = result.merge(yearly_races, on="year", how="left")
    result["win_share"] = result["wins"] / result["total_races"]
    return result.sort_values(["year", "wins"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Grid vs Finish
# ---------------------------------------------------------------------------


def grid_vs_finish_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per‑grid‑position finishing statistics.

    Returns:
        DataFrame with ``grid``, ``avg_finish``, ``avg_gain``, ``n_entries``,
        ``win_rate``.
    """
    agg = df.groupby("grid").agg(
        avg_finish=("positionOrder", "mean"),
        median_finish=("positionOrder", "median"),
        n_entries=("raceId", "count"),
        win_rate=("positionOrder", lambda x: (x == 1).mean()),
    ).reset_index()
    agg["avg_gain"] = agg["grid"] - agg["avg_finish"]
    return agg


def positions_gained(df: pd.DataFrame) -> pd.Series:
    """Return ``grid - positionOrder`` for every entry (positive = gain)."""
    return df["grid"] - df["positionOrder"]


# ---------------------------------------------------------------------------
# Pit stop analysis
# ---------------------------------------------------------------------------


def pit_stop_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Average pit stop duration and count per year.

    Returns:
        DataFrame with ``year``, ``avg_duration_ms``, ``avg_stops``,
        ``total_pit_duration_ms`` standard deviation, etc.
    """
    trends = df.groupby("year").agg(
        avg_duration_ms=("total_pit_duration_ms", "mean"),
        avg_stops=("num_pit_stops", "mean"),
        n_entries=("raceId", "count"),
    ).reset_index()
    return trends


# ---------------------------------------------------------------------------
# DNF / Status analysis
# ---------------------------------------------------------------------------


def dnf_rate_per_season(
    df: pd.DataFrame, status_map: Dict[int, str]
) -> pd.DataFrame:
    """Compute DNF (non‑finish) counts and rates per season.

    A *classified finish* includes ``"Finished"``, ``"+N Laps"`` statuses
    (the driver completed the race distance) and ``"Disqualified"``.
    Everything else (accidents, collisions, mechanical failures, …) is
    considered a DNF.

    Args:
        df: Preprocessed race DataFrame.
        status_map: ``statusId`` → status description.

    Returns:
        DataFrame with ``year``, ``entries``, ``dnfs``, ``dnf_rate``.
    """
    df = df.copy()
    df["statusName"] = df["statusId"].map(status_map)

    CLASSIFIED = {"Finished", "Disqualified"}
    # Statuses like "+1 Lap", "+2 Laps", … are classified finishes.
    def _is_classified(name: str) -> bool:
        if not isinstance(name, str):
            return False
        if name in CLASSIFIED:
            return True
        return name.startswith("+")

    yearly = df.groupby("year").agg(
        entries=("raceId", "count"),
        dnfs=("statusName", lambda s: (~s.apply(_is_classified)).sum()),
    ).reset_index()
    yearly["dnf_rate"] = yearly["dnfs"] / yearly["entries"]
    return yearly


def top_dnf_reasons(
    df: pd.DataFrame, status_map: Dict[int, str], top_n: int = 12
) -> pd.DataFrame:
    """Return the most common non‑finish status codes (excluding classified).

    Args:
        df: Preprocessed race DataFrame.
        status_map: statusId → label.
        top_n: Number of reasons.

    Returns:
        DataFrame with ``statusName``, ``count``.
    """
    df = df.copy()
    df["statusName"] = df["statusId"].map(status_map)

    CLASSIFIED = {"Finished", "Disqualified"}

    def _is_classified(name: str) -> bool:
        if not isinstance(name, str):
            return False
        if name in CLASSIFIED:
            return True
        return name.startswith("+")

    dnf = df[~df["statusName"].apply(_is_classified)]
    counts = dnf["statusName"].value_counts().head(top_n).reset_index()
    counts.columns = ["statusName", "count"]
    return counts


# ---------------------------------------------------------------------------
# Circuit analysis
# ---------------------------------------------------------------------------


def circuit_overtaking_metric(df: pd.DataFrame, circuit_map: Dict[int, str]) -> pd.DataFrame:
    """Rank circuits by average position gain (overtaking friendliness).

    Positive values indicate circuits where grid position matters less.

    Args:
        df: Preprocessed race DataFrame.
        circuit_map: ``circuitId`` → circuit name.

    Returns:
        DataFrame with ``circuitId``, ``circuitName``, ``avg_position_gain``,
        ``avg_grid``, ``avg_finish``, ``n_races``, ``avg_dnf_rate``.
    """
    df = df.copy()
    df["gain"] = df["grid"] - df["positionOrder"]
    agg = df.groupby("circuitId").agg(
        avg_position_gain=("gain", "mean"),
        avg_grid=("grid", "mean"),
        avg_finish=("positionOrder", "mean"),
        n_races=("raceId", "nunique"),
    ).reset_index()
    agg["circuitName"] = agg["circuitId"].map(
        lambda cid: circuit_map.get(cid, f"Circuit_{cid}")
    )
    return agg.sort_values("avg_position_gain", ascending=False)


# ---------------------------------------------------------------------------
# Rating trajectory helpers
# ---------------------------------------------------------------------------


def load_rating_history(history_path: str) -> Optional[pd.DataFrame]:
    """Load the driver rating history CSV produced by the TrueSkill model.

    Args:
        history_path: Path to ``driver_rating_history.csv``.

    Returns:
        DataFrame or *None* if the file does not exist.
    """
    try:
        df = pd.read_csv(history_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        logger.warning("Rating history not found at %s.  Run the model first.", history_path)
        return None


def top_driver_trajectories(
    history_df: pd.DataFrame,
    top_n: int = 8,
    metric: str = "mu",
) -> pd.DataFrame:
    """Extract the trajectories of the *top_n* drivers by final *metric*.

    Args:
        history_df: Full rating history DataFrame.
        top_n: Number of drivers.
        metric: Column to rank by (``"mu"`` or ``"sigma"``).

    Returns:
        Filtered history containing only the selected drivers.
    """
    final = history_df.groupby("driverId").last().reset_index()
    top_ids = final.sort_values(metric, ascending=False).head(top_n)["driverId"].tolist()
    return history_df[history_df["driverId"].isin(top_ids)]


def sigma_vs_races(history_df: pd.DataFrame) -> pd.DataFrame:
    """Compute each driver's sigma as a function of career race count.

    Returns:
        DataFrame with ``driverId``, ``driverName``, ``mu_final``,
        ``sigma_final``, ``n_races``.
    """
    final = history_df.groupby("driverId").agg(
        mu_final=("mu", "last"),
        sigma_final=("sigma", "last"),
        driverName=("driverName", "last"),
    ).reset_index()
    # Count races with > 5 drivers (proxy for "career races observed")
    race_counts = history_df.groupby("driverId").size().reset_index(name="n_race_snapshots")
    final = final.merge(race_counts, on="driverId")
    # Use number of unique races the driver appeared in
    n_unique = history_df.groupby("driverId")["raceId"].nunique().reset_index(name="n_races")
    final = final.merge(n_unique, on="driverId")
    return final


# ---------------------------------------------------------------------------
# Teammate analysis
# ---------------------------------------------------------------------------


def teammate_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """For each race, compare teammates (same constructor) by finish delta.

    Returns a DataFrame with the average teammate finish gap per
    constructor‑driver pair.
    """
    rows: List[Dict[str, object]] = []
    for race_id, group in df.groupby("raceId"):
        for c_id, team in group.groupby("constructorId"):
            if len(team) != 2:
                continue
            team = team.sort_values("positionOrder")
            driver_1 = team.iloc[0]
            driver_2 = team.iloc[1]
            rows.append(
                {
                    "raceId": race_id,
                    "constructorId": c_id,
                    "year": int(driver_1["year"]),
                    "driver_ahead": int(driver_1["driverId"]),
                    "driver_behind": int(driver_2["driverId"]),
                    "position_ahead": int(driver_1["positionOrder"]),
                    "position_behind": int(driver_2["positionOrder"]),
                    "finish_delta": int(driver_2["positionOrder"] - driver_1["positionOrder"]),
                }
            )
    teammate_df = pd.DataFrame(rows)
    if teammate_df.empty:
        return teammate_df
    # Aggregate per constructor‑driver combo
    agg = teammate_df.groupby(["constructorId", "driver_ahead"]).agg(
        avg_finish_delta=("finish_delta", "mean"),
        n_races_together=("raceId", "count"),
        beat_teammate_rate=("finish_delta", lambda x: (x > 0).mean()),
    ).reset_index()
    return agg.sort_values("avg_finish_delta", ascending=False)
