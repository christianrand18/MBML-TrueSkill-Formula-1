r"""
Convert F1 race data into pairwise comparison tensors for Pyro training.

Each race produces :math:`N \cdot (N-1)/2` ordered pairs (i finished
ahead of j).  Weather features (shared by all drivers in a race) are
attached per pair so they can modulate the performance noise scale.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("f1_pyro.data")

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

MEAN_FEATURES: List[str] = [
    "grid",
    "temp_mean_c",
    "wind_max_kmh",
    "humidity_mean_pct",
]

BINARY_FEATURES: List[str] = [
    "is_wet",
    "is_very_wet",
    "is_hot",
    "is_cold",
    "is_windy",
    "is_humid",
]

ALL_FEATURES: List[str] = MEAN_FEATURES + BINARY_FEATURES


# ---------------------------------------------------------------------------
# Pairwise dataset builder
# ---------------------------------------------------------------------------


class PairwiseDataset:
    """Holds pairwise comparison data for SVI training.

    Attributes:
        n_drivers: Number of unique drivers.
        n_constructors: Number of unique constructors.
        n_seasons: Number of seasons (years).
        n_covariates: Number of per‑driver covariate features.
        n_pairs: Total number of pairwise comparisons.
        driver_i: LongTensor [n_pairs] — index of winning driver.
        driver_j: LongTensor [n_pairs] — index of losing driver.
        cons_i: LongTensor [n_pairs] — index of winner's constructor.
        cons_j: LongTensor [n_pairs] — index of loser's constructor.
        season: LongTensor [n_pairs] — season index.
        feats_i: FloatTensor [n_pairs, n_covariates] — winner features.
        feats_j: FloatTensor [n_pairs, n_covariates] — loser features.
        weather: FloatTensor [n_pairs, n_weather] — shared weather.
    """

    def __init__(
        self,
        n_drivers: int,
        n_constructors: int,
        n_seasons: int,
        n_covariates: int,
    ):
        self.n_drivers = n_drivers
        self.n_constructors = n_constructors
        self.n_seasons = n_seasons
        self.n_covariates = n_covariates

        # Will be filled
        self.driver_i: torch.LongTensor = torch.empty(0, dtype=torch.long)
        self.driver_j: torch.LongTensor = torch.empty(0, dtype=torch.long)
        self.cons_i: torch.LongTensor = torch.empty(0, dtype=torch.long)
        self.cons_j: torch.LongTensor = torch.empty(0, dtype=torch.long)
        self.season: torch.LongTensor = torch.empty(0, dtype=torch.long)
        self.feats_i: torch.Tensor = torch.empty(0)
        self.feats_j: torch.Tensor = torch.empty(0)
        self.weather: torch.Tensor = torch.empty(0)
        self.n_pairs: int = 0

    @property
    def n_weather(self) -> int:
        return self.weather.shape[1] if self.weather.numel() > 0 else 0


class DataPreparer:
    """Load enriched CSV, normalise features, and build pairwise datasets.

    Args:
        enriched_path: Path to ``f1_enriched.csv``.
        mean_feats: Continuous features (z‑score normalised per season).
        binary_feats: Binary features (used as‑is).
    """

    def __init__(
        self,
        enriched_path: str,
        mean_feats: List[str] | None = None,
        binary_feats: List[str] | None = None,
    ):
        self._path = enriched_path
        self._mean_feats = mean_feats or MEAN_FEATURES
        self._binary_feats = binary_feats or BINARY_FEATURES
        self._all_feats = self._mean_feats + self._binary_feats

        # Mappings (set by _build_mappings)
        self.driver_map: Dict[int, int] = {}
        self.constructor_map: Dict[int, int] = {}
        self.season_map: Dict[int, int] = {}
        self._reverse_driver: Dict[int, int] = {}
        self._reverse_constructor: Dict[int, int] = {}

        # Normalisation stats
        self._norm_mean: Dict[str, float] = {}
        self._norm_std: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Tuple[PairwiseDataset, pd.DataFrame]:
        """Load data, normalise, and build pairwise tensors.

        Returns:
            ``(dataset, enriched_df)`` where *enriched_df* has been
            normalised and includes index columns (``_d_idx``, ``_c_idx``,
            ``_s_idx``).
        """
        df = pd.read_csv(self._path)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("Loaded %d rows from %s.", len(df), self._path)

        self._build_mappings(df)
        df = self._add_indices(df)
        df = self._normalise(df)

        dataset = self._build_pairwise_tensors(df)
        logger.info(
            "Pairwise dataset: %d pairs across %d drivers, "
            "%d constructors, %d seasons.",
            dataset.n_pairs,
            self.n_drivers,
            self.n_constructors,
            self.n_seasons,
        )
        return dataset, df

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_drivers(self) -> int:
        return len(self.driver_map)

    @property
    def n_constructors(self) -> int:
        return len(self.constructor_map)

    @property
    def n_seasons(self) -> int:
        return len(self.season_map)

    @property
    def n_covariates(self) -> int:
        return len(self._all_feats)

    def reverse_driver(self, idx: int) -> int:
        return self._reverse_driver.get(idx, -1)

    def reverse_constructor(self, idx: int) -> int:
        return self._reverse_constructor.get(idx, -1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_mappings(self, df: pd.DataFrame) -> None:
        """Create 0‑based index mappings for drivers, constructors, seasons."""
        driver_ids = sorted(df["driverId"].unique())
        cons_ids = sorted(df["constructorId"].unique())
        seasons = sorted(df["year"].unique())

        self.driver_map = {int(d): i for i, d in enumerate(driver_ids)}
        self.constructor_map = {int(c): i for i, c in enumerate(cons_ids)}
        self.season_map = {int(y): i for i, y in enumerate(seasons)}

        self._reverse_driver = {v: k for k, v in self.driver_map.items()}
        self._reverse_constructor = {v: k for k, v in self.constructor_map.items()}

        logger.info(
            "Mappings: %d drivers, %d constructors, %d seasons.",
            len(self.driver_map),
            len(self.constructor_map),
            len(self.season_map),
        )

    def _add_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``_d_idx``, ``_c_idx``, ``_s_idx`` columns."""
        df = df.copy()
        df["_d_idx"] = df["driverId"].map(self.driver_map)
        df["_c_idx"] = df["constructorId"].map(self.constructor_map)
        df["_s_idx"] = df["year"].map(self.season_map)
        return df

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z‑score normalise *mean_feats* per season; keep binary as‑is."""
        df = df.copy()
        for col in self._mean_feats:
            if col not in df.columns:
                continue
            grp = df.groupby("_s_idx")[col]
            mean = grp.transform("mean")
            std = grp.transform("std").replace(0.0, 1.0)
            df[col] = (df[col] - mean) / std
            self._norm_mean[col] = float(df[col].mean())
            self._norm_std[col] = float(df[col].std())
        return df

    def _build_pairwise_tensors(self, df: pd.DataFrame) -> PairwiseDataset:
        """Iterate races, create all ordered pairs."""
        dataset = PairwiseDataset(
            n_drivers=self.n_drivers,
            n_constructors=self.n_constructors,
            n_seasons=self.n_seasons,
            n_covariates=self.n_covariates,
        )

        driver_i_list: List[int] = []
        driver_j_list: List[int] = []
        cons_i_list: List[int] = []
        cons_j_list: List[int] = []
        season_list: List[int] = []
        feats_i_list: List[List[float]] = []
        feats_j_list: List[List[float]] = []
        weather_list: List[List[float]] = []

        for race_id, race in df.groupby("raceId", sort=False):
            race = race.sort_values("positionOrder")
            entries = []
            for _, row in race.iterrows():
                feats = [float(row.get(c, 0)) for c in self._all_feats]
                entries.append({
                    "d_idx": int(row["_d_idx"]),
                    "c_idx": int(row["_c_idx"]),
                    "s_idx": int(row["_s_idx"]),
                    "feats": feats,
                })

            n = len(entries)
            # Weather features: same for all entries (use first)
            weather_row = {c: float(race.iloc[0].get(c, 0)) for c in self._mean_feats + self._binary_feats}
            weather_vec = [weather_row.get(c, 0.0) for c in self._all_feats]

            for a in range(n):
                for b in range(a + 1, n):
                    # a finished ahead of b
                    driver_i_list.append(entries[a]["d_idx"])
                    driver_j_list.append(entries[b]["d_idx"])
                    cons_i_list.append(entries[a]["c_idx"])
                    cons_j_list.append(entries[b]["c_idx"])
                    season_list.append(entries[a]["s_idx"])
                    feats_i_list.append(entries[a]["feats"])
                    feats_j_list.append(entries[b]["feats"])
                    weather_list.append(weather_vec)

        dataset.driver_i = torch.tensor(driver_i_list, dtype=torch.long)
        dataset.driver_j = torch.tensor(driver_j_list, dtype=torch.long)
        dataset.cons_i = torch.tensor(cons_i_list, dtype=torch.long)
        dataset.cons_j = torch.tensor(cons_j_list, dtype=torch.long)
        dataset.season = torch.tensor(season_list, dtype=torch.long)
        dataset.feats_i = torch.tensor(feats_i_list, dtype=torch.float)
        dataset.feats_j = torch.tensor(feats_j_list, dtype=torch.float)
        dataset.weather = torch.tensor(weather_list, dtype=torch.float)
        dataset.n_pairs = len(driver_i_list)

        return dataset
