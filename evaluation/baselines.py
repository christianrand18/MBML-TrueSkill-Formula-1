r"""
Baseline skill‑rating models for F1 race prediction.

All models implement a common ``SkillPredictor`` interface and can be
plugged into the ``ChronologicalValidator`` for head‑to‑head comparison
with the TrueSkill Bayesian model.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("f1_evaluation.baselines")

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SkillPredictor(ABC):
    """Interface for a skill‑rating model.

    Subclasses must implement ``fit`` and ``predict_driver_skills``.
    """

    name: str = "BasePredictor"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Learn from historical training data.

        Args:
            train_df: DataFrame containing all columns from
                ``f1_model_ready.csv`` for the training seasons.
        """
        ...

    @abstractmethod
    def predict_driver_skills(self, race_df: pd.DataFrame) -> Dict[int, float]:
        """Return a skill score for each driver in a single race.

        Higher scores mean stronger drivers (analogous to TrueSkill μ).

        Args:
            race_df: One race's rows from ``f1_model_ready.csv``.

        Returns:
            Mapping ``driverId → skill_score``.
        """
        ...


# ---------------------------------------------------------------------------
# 1. Grid position baseline
# ---------------------------------------------------------------------------


class GridPredictor(SkillPredictor):
    """Predict finish from starting grid position — no training needed.

    Skill = ``-grid`` (i.e. pole = strongest).
    """

    name = "Grid"

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # No training

    def predict_driver_skills(self, race_df: pd.DataFrame) -> Dict[int, float]:
        return {
            int(row["driverId"]): -float(row["grid"])
            for _, row in race_df.iterrows()
        }


# ---------------------------------------------------------------------------
# 2. Elo rating baseline
# ---------------------------------------------------------------------------


class EloPredictor(SkillPredictor):
    """Pairwise Elo rating system (K = 32, initial rating = 1500).

    After each race every driver‑pair is evaluated.  Higher Elo = stronger
    driver.  Constructor skill is implicitly captured through the driver's
    accumulated rating.
    """

    name = "Elo"

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        scale: float = 400.0,
    ) -> None:
        self._initial = initial_rating
        self._k = k_factor
        self._scale = scale
        self._ratings: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Chronologically process every training race."""
        self._ratings = {}
        races = train_df.groupby("raceId", sort=False)
        for race_id, race in races:
            race = race.sort_values("date")  # ensure within‑race order
            self._update_from_race(race)

    def predict_driver_skills(self, race_df: pd.DataFrame) -> Dict[int, float]:
        return {
            int(row["driverId"]): self._ratings.get(
                int(row["driverId"]), self._initial
            )
            for _, row in race_df.iterrows()
        }

    # -- internal ----------------------------------------------------------

    def _update_from_race(self, race: pd.DataFrame) -> None:
        """Apply pairwise Elo updates for one race."""
        entries = []
        for _, row in race.iterrows():
            d_id = int(row["driverId"])
            pos = int(row["positionOrder"])
            rating = self._ratings.get(d_id, self._initial)
            entries.append((d_id, rating, pos))
            self._ratings[d_id] = rating  # ensure initialised

        n = len(entries)
        # Accumulate deltas
        deltas: Dict[int, float] = defaultdict(float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_a, r_a, p_a = entries[i]
                d_b, r_b, p_b = entries[j]
                # Expected score
                expected = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / self._scale))
                # Actual score: 1 if a finishes ahead of b
                actual = 1.0 if p_a < p_b else (0.5 if p_a == p_b else 0.0)
                delta = self._k * (actual - expected) / (n - 1)
                deltas[d_a] += delta

        for d_id, delta in deltas.items():
            self._ratings[d_id] = self._ratings.get(d_id, self._initial) + delta


# ---------------------------------------------------------------------------
# 3. Previous‑season championship points baseline
# ---------------------------------------------------------------------------


class PreviousSeasonPredictor(SkillPredictor):
    """Skill = the driver's championship points from the preceding season.

    Points are loaded from the raw ``results.csv`` because they are not
    included in ``f1_model_ready.csv``.
    """

    name = "PrevSeason"

    def __init__(self, results_df: pd.DataFrame) -> None:
        """Pre‑compute per‑driver per‑year point totals.

        Args:
            results_df: Raw ``results.csv`` DataFrame (must contain
                ``driverId``, ``raceId``, ``points`` columns, plus a
                ``year`` column obtained via a pre‑join with ``races.csv``).
        """
        self._season_points = self._compute_points(results_df)
        self._last_year: int = 0

    @staticmethod
    def _compute_points(results_df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
        """Return ``{(driverId, year): total_points}``."""
        pts = results_df.groupby(["driverId", "year"])["points"].sum()
        return pts.to_dict()

    def fit(self, train_df: pd.DataFrame) -> None:
        """Record the last training year."""
        self._last_year = int(train_df["year"].max())

    def predict_driver_skills(self, race_df: pd.DataFrame) -> Dict[int, float]:
        """Use previous‑season points as skill (0 for rookies)."""
        prev_year = self._last_year
        skills: Dict[int, float] = {}
        for _, row in race_df.iterrows():
            d_id = int(row["driverId"])
            pts = self._season_points.get((d_id, prev_year), 0.0)
            skills[d_id] = pts
        return skills


# ---------------------------------------------------------------------------
# 4. Random baseline  (lower bound)
# ---------------------------------------------------------------------------


class RandomPredictor(SkillPredictor):
    """Assign random skill scores — absolute lower bound for metrics."""

    name = "Random"

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def predict_driver_skills(self, race_df: pd.DataFrame) -> Dict[int, float]:
        return {
            int(row["driverId"]): float(self._rng.uniform(0, 1))
            for _, row in race_df.iterrows()
        }


# ---------------------------------------------------------------------------
# 5. TrueSkill predictor (wraps rating history)
# ---------------------------------------------------------------------------


class TrueSkillHistoryPredictor(SkillPredictor):
    """Skill = the TrueSkill μ at the end of the training window.

    Reads the ``driver_rating_history.csv`` produced by the TrueSkill
    model.  For each test race, returns the ratings from the last training
    race (which represent all information before the test season).
    """

    name = "TrueSkill"

    def __init__(self, history_df: pd.DataFrame) -> None:
        self._history: pd.DataFrame = history_df
        self._last_ratings: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Extract ratings at the end of the training window."""
        train_race_ids = set(int(r) for r in train_df["raceId"].unique())
        # Get the last history entry for each driver within training races
        hist_train = self._history[
            self._history["raceId"].isin(train_race_ids)
        ]
        # For each driver, take the last entry (largest raceId in training)
        last_per_driver = hist_train.sort_values("raceId").groupby("driverId").last()
        self._last_ratings = {}
        for d_id, row in last_per_driver.iterrows():
            self._last_ratings[int(d_id)] = float(row["mu"])

    def predict_driver_skills(self, race_df: pd.DataFrame) -> Dict[int, float]:
        """Return frozen training‑end ratings for all entrants.

        Drivers who never appeared in training get the default mu = 25.
        """
        return {
            int(row["driverId"]): self._last_ratings.get(
                int(row["driverId"]), 25.0
            )
            for _, row in race_df.iterrows()
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_all_predictors(
    results_with_year: Optional[pd.DataFrame] = None,
    rating_history: Optional[pd.DataFrame] = None,
) -> Dict[str, SkillPredictor]:
    """Return a dict of ``{name: SkillPredictor}`` for all five models.

    Args:
        results_with_year: Raw ``results.csv`` joined with ``races.csv``
            to add a ``year`` column (needed by ``PreviousSeasonPredictor``).
        rating_history: TrueSkill rating‑history DataFrame.

    Returns:
        Model name → predictor instance.
    """
    predictors: Dict[str, SkillPredictor] = {
        "Grid": GridPredictor(),
        "Elo": EloPredictor(),
        "Random": RandomPredictor(),
    }
    if results_with_year is not None:
        predictors["PrevSeason"] = PreviousSeasonPredictor(results_with_year)
    else:
        logger.warning("No results_with_year provided; skipping PrevSeason baseline.")
    if rating_history is not None:
        predictors["TrueSkill"] = TrueSkillHistoryPredictor(rating_history)
    else:
        logger.warning("No rating_history provided; skipping TrueSkill predictor.")
    return predictors
