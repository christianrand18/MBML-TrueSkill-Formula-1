r"""
F1 TrueSkill Baseline Model
=============================

A chronological Bayesian skill-rating pipeline for Formula 1 drivers and
constructors.  Races are modelled as free-for-all matches where each entry is a
two-player team :math:`[driver, constructor]`.

Architecture
------------
* ``SkillEvaluator``      – abstract backend (swap TrueSkill for Pyro later).
* ``TrueSkillEvaluator``  – concrete ``trueskill`` backend.
* ``F1RatingEnvironment`` – stateful store of per-entity ``trueskill.Rating``.
* ``RaceProcessor``       – builds teams from a single race and invokes the
  evaluator.
* ``F1SkillPipeline``     – orchestrator: load data, process chronologically,
  export results.

Output
------
* ``outputs/ratings/driver_ratings.csv``      – final mu / sigma per driver.
* ``outputs/ratings/constructor_ratings.csv`` – final mu / sigma per constructor.
* ``outputs/history/driver_rating_history.csv`` – per-race rating snapshots.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from trueskill import Rating, TrueSkill, rate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("f1_trueskill")

# ---------------------------------------------------------------------------
# Abstract skill evaluator  (extensibility hook for Pyro)
# ---------------------------------------------------------------------------


class SkillEvaluator(ABC):
    """Abstract interface for a skill-rating backend.

    Subclass this to swap ``trueskill`` for a custom Bayesian backend
    (e.g. Pyro) while keeping the rest of the pipeline unchanged.
    """

    @abstractmethod
    def update_skills(
        self,
        teams: List[Tuple[Rating, ...]],
        ranks: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Rating, ...]]:
        """Compute posterior team ratings given observed ranks.

        Args:
            teams: List of teams, where each team is a tuple of
                ``trueskill.Rating`` objects (one per entity).
            ranks: List of integer ranks for each team (0 = 1st place).
                Teams sharing the same rank value are considered tied.
            context: Optional dictionary of contextual features
                (weather, tyre compound, …).  Ignored by the baseline
                TrueSkill evaluator but available to future Pyro backends.

        Returns:
            List of posterior teams with the same structure as *teams*.
        """
        ...


# ---------------------------------------------------------------------------
# TrueSkill backend
# ---------------------------------------------------------------------------


class TrueSkillEvaluator(SkillEvaluator):
    """Skill evaluator that delegates to the ``trueskill`` library.

    Args:
        mu: Initial mean skill for new players (default 25.0).
        sigma: Initial skill uncertainty (default 25/3).
        beta: Skill difference for an 80 % win probability (default 25/6).
        tau: Per-match skill dynamics factor (default 25/300).
        draw_probability: Probability of a draw (default 0.0 – F1 has no draws
            in the traditional sense).
    """

    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25.0 / 3,
        beta: float = 25.0 / 6,
        tau: float = 25.0 / 300,
        draw_probability: float = 0.0,
    ) -> None:
        self._params = {
            "mu": mu,
            "sigma": sigma,
            "beta": beta,
            "tau": tau,
            "draw_probability": draw_probability,
        }
        self._env = TrueSkill(**self._params)

    @property
    def env(self) -> TrueSkill:
        """The underlying ``TrueSkill`` environment instance."""
        return self._env

    def update_skills(
        self,
        teams: List[Tuple[Rating, ...]],
        ranks: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Rating, ...]]:
        """Delegate to ``trueskill.rate``.

        The *context* dictionary is accepted for interface compatibility but
        is not used by the baseline TrueSkill model.
        """
        return rate(teams, ranks=ranks)


# ---------------------------------------------------------------------------
# Rating environment
# ---------------------------------------------------------------------------


class F1RatingEnvironment:
    """Persistent store of TrueSkill ratings for drivers and constructors.

    Provides factory methods that lazily initialise new entities with a
    default ``Rating()`` and supports batched posterior updates after each
    race.

    Args:
        mu: Default initial mu for new entrants.
        sigma: Default initial sigma for new entrants.
    """

    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25.0 / 3,
    ) -> None:
        self._mu = mu
        self._sigma = sigma
        self._driver_ratings: Dict[int, Rating] = {}
        self._constructor_ratings: Dict[int, Rating] = {}
        # Human-readable name maps (populated by the pipeline)
        self._driver_names: Dict[int, str] = {}
        self._constructor_names: Dict[int, str] = {}

    # -- Factory helpers --------------------------------------------------

    def get_or_create_driver(self, driver_id: int) -> Rating:
        """Return the current ``Rating`` for *driver_id*, creating it if new.

        Args:
            driver_id: Unique driver identifier.

        Returns:
            The driver's current ``Rating``.
        """
        if driver_id not in self._driver_ratings:
            self._driver_ratings[driver_id] = Rating(
                mu=self._mu, sigma=self._sigma
            )
        return self._driver_ratings[driver_id]

    def get_or_create_constructor(self, constructor_id: int) -> Rating:
        """Return the current ``Rating`` for *constructor_id*, creating it if new.

        Args:
            constructor_id: Unique constructor identifier.

        Returns:
            The constructor's current ``Rating``.
        """
        if constructor_id not in self._constructor_ratings:
            self._constructor_ratings[constructor_id] = Rating(
                mu=self._mu, sigma=self._sigma
            )
        return self._constructor_ratings[constructor_id]

    def clone_constructor(self, constructor_id: int) -> Rating:
        """Return a **copy** of the constructor's current rating.

        Constructors may field two cars in the same race.  Because the
        ``trueskill`` library returns per-team (rather than per-player)
        posteriors when a player appears on multiple teams, we clone the
        constructor rating for each car and average the returned
        posteriors.

        Args:
            constructor_id: Constructor to clone.

        Returns:
            A new ``Rating`` with the same mu and sigma.
        """
        current = self.get_or_create_constructor(constructor_id)
        return Rating(mu=current.mu, sigma=current.sigma)

    # -- Name registry ----------------------------------------------------

    def register_names(
        self,
        driver_map: Dict[int, str],
        constructor_map: Dict[int, str],
    ) -> None:
        """Attach human-readable names for output reports.

        Args:
            driver_map: Mapping from ``driverId`` to ``"forename surname"``.
            constructor_map: Mapping from ``constructorId`` to team name.
        """
        self._driver_names = driver_map
        self._constructor_names = constructor_map

    # -- Batch posterior update -------------------------------------------

    def apply_driver_posteriors(
        self, updates: Dict[int, Rating]
    ) -> None:
        """Overwrite stored driver ratings with posterior values.

        Args:
            updates: Mapping of ``driverId`` → posterior ``Rating``.
        """
        for d_id, rating in updates.items():
            if d_id in self._driver_ratings:
                self._driver_ratings[d_id] = rating

    def apply_constructor_posterior(
        self, constructor_id: int, rating: Rating
    ) -> None:
        """Overwrite a single constructor's stored rating.

        Args:
            constructor_id: Constructor identifier.
            rating: Posterior ``Rating`` (e.g. averaged over cars).
        """
        self._constructor_ratings[constructor_id] = rating

    # -- Snapshot exports ------------------------------------------------

    def driver_snapshot(self) -> pd.DataFrame:
        """Return the current driver ratings as a sorted DataFrame.

        Columns: ``driverId``, ``driverName``, ``mu``, ``sigma``.
        Sorted descending by *mu*.
        """
        rows: List[Dict[str, Any]] = []
        for d_id, rating in self._driver_ratings.items():
            rows.append(
                {
                    "driverId": d_id,
                    "driverName": self._driver_names.get(
                        d_id, f"Driver_{d_id}"
                    ),
                    "mu": float(rating.mu),
                    "sigma": float(rating.sigma),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("mu", ascending=False)
            .reset_index(drop=True)
        )

    def constructor_snapshot(self) -> pd.DataFrame:
        """Return the current constructor ratings as a sorted DataFrame.

        Columns: ``constructorId``, ``constructorName``, ``mu``, ``sigma``.
        Sorted descending by *mu*.
        """
        rows: List[Dict[str, Any]] = []
        for c_id, rating in self._constructor_ratings.items():
            rows.append(
                {
                    "constructorId": c_id,
                    "constructorName": self._constructor_names.get(
                        c_id, f"Constructor_{c_id}"
                    ),
                    "mu": float(rating.mu),
                    "sigma": float(rating.sigma),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("mu", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Race processor
# ---------------------------------------------------------------------------


class RaceProcessor:
    """Translates a single race's result rows into a TrueSkill match.

    Because a constructor may field two cars in the same race, we clone
    the constructor's rating for each entry and average the returned
    posteriors after rating.

    Args:
        env: The shared rating environment.
        evaluator: The skill-rating backend.
    """

    def __init__(
        self,
        env: F1RatingEnvironment,
        evaluator: SkillEvaluator,
    ) -> None:
        self._env = env
        self._evaluator = evaluator

    def process_race(
        self,
        race_df: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Process a single race and update all entity ratings in-place.

        Args:
            race_df: DataFrame slice containing one race (all competitors).
                Must include columns ``driverId``, ``constructorId``,
                ``positionOrder``.
            context: Optional dictionary of per-race features (weather,
                tyres, …).  Passed through to the evaluator.
        """
        n_entries = len(race_df)
        if n_entries < 2:
            logger.debug("Skipping race with < 2 competitors.")
            return

        teams: List[Tuple[Rating, ...]] = []
        ranks: List[int] = []
        driver_ids: List[int] = []
        constructor_ids: List[int] = []

        # Track which team indices belong to each constructor for averaging
        cons_team_indices: Dict[int, List[int]] = defaultdict(list)

        for idx, (_, row) in enumerate(race_df.iterrows()):
            d_id = int(row["driverId"])
            c_id = int(row["constructorId"])
            pos_order = int(row["positionOrder"])

            driver_rating = self._env.get_or_create_driver(d_id)
            # Clone constructor so each car is an independent team
            constructor_clone = self._env.clone_constructor(c_id)

            teams.append((driver_rating, constructor_clone))
            ranks.append(pos_order - 1)  # 1st place → rank 0
            driver_ids.append(d_id)
            constructor_ids.append(c_id)
            cons_team_indices[c_id].append(idx)

        # -- Call the TrueSkill backend ----------------------------------
        posterior_teams = self._evaluator.update_skills(teams, ranks, context)

        # -- Update drivers (one posterior per driver) -------------------
        driver_updates: Dict[int, Rating] = {}
        for i, d_id in enumerate(driver_ids):
            driver_updates[d_id] = posterior_teams[i][0]
        self._env.apply_driver_posteriors(driver_updates)

        # -- Update constructors (average across their cars) -------------
        for c_id, team_indices in cons_team_indices.items():
            mus = [posterior_teams[i][1].mu for i in team_indices]
            sigmas = [posterior_teams[i][1].sigma for i in team_indices]
            avg_rating = Rating(
                mu=float(np.mean(mus)),
                sigma=float(np.mean(sigmas)),
            )
            self._env.apply_constructor_posterior(c_id, avg_rating)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class F1SkillPipeline:
    """End-to-end orchestrator for the F1 TrueSkill baseline model.

    Loads the preprocessed CSV, processes every race chronologically,
    and exports final ratings together with a per-race rating history.

    Args:
        data_path: Path to ``f1_model_ready.csv``.
        output_dir: Root directory for output files (default ``outputs/``).
        evaluator: Optional pre-configured ``SkillEvaluator``.  If *None*,
            a default ``TrueSkillEvaluator`` is created.
    """

    def __init__(
        self,
        data_path: str,
        output_dir: str = "outputs",
        evaluator: Optional[SkillEvaluator] = None,
    ) -> None:
        self._data_path = data_path
        self._output_dir = output_dir
        self._env = F1RatingEnvironment()
        self._evaluator = evaluator or TrueSkillEvaluator()
        self._processor = RaceProcessor(self._env, self._evaluator)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_and_prepare(self) -> pd.DataFrame:
        """Load the preprocessed race data and attach human-readable names.

        Returns:
            DataFrame sorted by *date* ascending, ready for chronological
            processing.
        """
        logger.info("Loading preprocessed data from %s", self._data_path)
        df = pd.read_csv(self._data_path)
        df["date"] = pd.to_datetime(df["date"])

        # -- Merge driver & constructor names ----------------------------
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        )

        drivers_csv = os.path.join(data_dir, "drivers.csv")
        constructors_csv = os.path.join(data_dir, "constructors.csv")

        driver_names: Dict[int, str] = {}
        if os.path.exists(drivers_csv):
            drivers_df = pd.read_csv(
                drivers_csv, na_values=[r"\N"], keep_default_na=True
            )
            for _, row in drivers_df.iterrows():
                forename = str(row.get("forename", ""))
                surname = str(row.get("surname", ""))
                driver_names[int(row["driverId"])] = f"{forename} {surname}".strip()
        else:
            logger.warning("drivers.csv not found – names will be IDs.")

        constructor_names: Dict[int, str] = {}
        if os.path.exists(constructors_csv):
            cons_df = pd.read_csv(
                constructors_csv, na_values=[r"\N"], keep_default_na=True
            )
            for _, row in cons_df.iterrows():
                constructor_names[int(row["constructorId"])] = str(
                    row.get("name", "")
                )
        else:
            logger.warning("constructors.csv not found – names will be IDs.")

        self._env.register_names(driver_names, constructor_names)

        # Ensure chronological order
        df = df.sort_values(["date", "raceId"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full pipeline: load → process → export."""
        df = self._load_and_prepare()

        race_ids = df["raceId"].unique()
        n_races = len(race_ids)
        logger.info(
            "Starting chronological processing of %d races (%.2d–%.2d).",
            n_races,
            int(df["year"].min()),
            int(df["year"].max()),
        )

        history: List[pd.DataFrame] = []
        n_entries_total = 0

        for i, race_id in enumerate(race_ids):
            race_df = df[df["raceId"] == race_id]
            n_entries_total += len(race_df)

            # Process one race
            self._processor.process_race(race_df)

            # Append post-race snapshot
            snap = self._env.driver_snapshot()
            year_val = race_df["year"].iloc[0]
            date_val = race_df["date"].iloc[0]
            snap.insert(0, "date", date_val)
            snap.insert(0, "year", year_val)
            snap.insert(0, "raceId", race_id)
            history.append(snap)

            # Progress logging every 50 races
            if (i + 1) % 50 == 0 or i == 0 or i == n_races - 1:
                logger.info(
                    "Race %4d/%d  |  raceId=%4d  |  %s  |  "
                    "entries in race: %2d  |  total entries: %d",
                    i + 1,
                    n_races,
                    int(race_id),
                    str(date_val.date()),
                    len(race_df),
                    n_entries_total,
                )

        logger.info(
            "Processing complete: %d races, %d driver-race entries.",
            n_races,
            n_entries_total,
        )

        self._export_results(history)
        self._log_leaderboard()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_results(self, history: List[pd.DataFrame]) -> None:
        """Write final ratings and full rating history to CSV.

        Args:
            history: List of per-race driver-snapshot DataFrames.
        """
        ratings_dir = os.path.join(self._output_dir, "ratings")
        history_dir = os.path.join(self._output_dir, "history")
        os.makedirs(ratings_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)

        # Final ratings
        drivers_df = self._env.driver_snapshot()
        constructors_df = self._env.constructor_snapshot()

        driver_path = os.path.join(ratings_dir, "driver_ratings.csv")
        constructor_path = os.path.join(ratings_dir, "constructor_ratings.csv")
        drivers_df.to_csv(driver_path, index=False)
        constructors_df.to_csv(constructor_path, index=False)
        logger.info("Exported final ratings to %s", ratings_dir)

        # Rating history
        history_df = pd.concat(history, ignore_index=True)
        # Reorder columns for readability
        col_order = [
            "raceId",
            "year",
            "date",
            "driverId",
            "driverName",
            "mu",
            "sigma",
        ]
        history_df = history_df[col_order]
        history_path = os.path.join(history_dir, "driver_rating_history.csv")
        history_df.to_csv(history_path, index=False)
        logger.info(
            "Exported rating history (%d snapshots) to %s",
            len(history),
            history_dir,
        )

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def _log_leaderboard(self) -> None:
        """Print the top 10 drivers and constructors by mu to the log."""
        drivers_df = self._env.driver_snapshot()
        constructors_df = self._env.constructor_snapshot()

        logger.info("=" * 70)
        logger.info("ALL-TIME TOP 10 DRIVERS (by posterior mu):")
        for rank, (_, row) in enumerate(drivers_df.head(10).iterrows(), 1):
            logger.info(
                "  %2d.  %-28s  mu = %7.3f  sigma = %6.3f",
                rank,
                row["driverName"],
                row["mu"],
                row["sigma"],
            )

        logger.info("-" * 70)
        logger.info("ALL-TIME TOP 10 CONSTRUCTORS (by posterior mu):")
        for rank, (_, row) in enumerate(constructors_df.head(10).iterrows(), 1):
            logger.info(
                "  %2d.  %-28s  mu = %7.3f  sigma = %6.3f",
                rank,
                row["constructorName"],
                row["mu"],
                row["sigma"],
            )
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: configure logging and launch the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir, "..", "data_preprocessing", "f1_model_ready.csv"
    )
    output_dir = os.path.join(base_dir, "..", "outputs")

    pipeline = F1SkillPipeline(
        data_path=os.path.normpath(data_path),
        output_dir=os.path.normpath(output_dir),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
