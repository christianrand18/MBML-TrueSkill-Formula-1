"""Data preparation for the F1 TrueSkill PGM.

Loads ``data_preprocessing/f1_enriched.csv``, applies constructor rebranding
merges, classifies DNFs, and emits a verified ``F1RankingDataset`` dataclass.
"""

from dataclasses import dataclass

import pandas as pd
import torch

CONSTRUCTOR_REMAP = {
    211: 10,  # Racing Point -> Force India
    117: 10,  # Aston Martin -> Force India
    214: 4,   # Alpine -> Renault
    213: 5,   # AlphaTauri -> Toro Rosso
    215: 5,   # Racing Bulls -> Toro Rosso
    51: 15,   # Alfa Romeo -> Sauber
}

MECHANICAL_STATUS_IDS = frozenset({
    5, 6, 7, 8, 9, 10, 21, 22, 26, 28, 29, 31, 36,
    40, 41, 43, 44, 54, 61, 65, 66, 67, 72, 75, 82, 104, 107, 108, 131,
})

FINISHED_STATUS_IDS = frozenset({
    1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
})


@dataclass
class F1RankingDataset:
    """Ranking entries only (N_entries = finishers + driver-fault DNFs, NO mechanical DNFs)."""

    driver_idx: torch.Tensor       # (N_entries,) LongTensor
    cons_idx: torch.Tensor         # (N_entries,) LongTensor
    season_idx: torch.Tensor       # (N_entries,) LongTensor
    circuit_idx: torch.Tensor      # (N_entries,) LongTensor
    race_idx: torch.Tensor         # (N_entries,) LongTensor

    pit_norm: torch.Tensor         # (N_entries,) FloatTensor — normalised per season
    wet: torch.Tensor              # (N_races,) FloatTensor — binary wet indicator
    race_order: torch.Tensor       # (N_entries,) LongTensor — 0=winner within-race
    race_lengths: torch.Tensor     # (N_races,) LongTensor — entries per race

    n_drivers: int
    n_constructors: int
    n_seasons: int
    n_circuits: int
    n_races: int

    driver_map: dict   # int_idx -> driverId
    constructor_map: dict  # int_idx -> constructorId


def load_dataset(csv_path: str = "data_preprocessing/f1_enriched.csv") -> F1RankingDataset:
    """Load and pre-process the F1 enriched dataset into a verified F1RankingDataset."""
    df = pd.read_csv(csv_path)

    # ---- 1. Apply constructor rebranding merges BEFORE building indices ----
    df["constructorId"] = df["constructorId"].replace(CONSTRUCTOR_REMAP)

    # ---- 2. Build integer-index maps ----
    unique_drivers = sorted(df["driverId"].unique())
    driver_lookup = {d: i for i, d in enumerate(unique_drivers)}
    driver_map = dict(enumerate(unique_drivers))

    unique_constructors = sorted(df["constructorId"].unique())
    constructor_lookup = {c: i for i, c in enumerate(unique_constructors)}
    constructor_map = dict(enumerate(unique_constructors))

    unique_seasons = sorted(df["year"].unique())
    season_lookup = {s: i for i, s in enumerate(unique_seasons)}

    unique_circuits = sorted(df["circuitId"].unique())
    circuit_lookup = {c: i for i, c in enumerate(unique_circuits)}

    unique_races = sorted(df["raceId"].unique())
    race_lookup = {r: i for i, r in enumerate(unique_races)}

    # ---- 3. DNF classification ----
    is_finished = df["statusId"].isin(FINISHED_STATUS_IDS)
    is_mechanical = df["statusId"].isin(MECHANICAL_STATUS_IDS)

    include_rank = ~is_mechanical
    ranking = df.loc[include_rank].copy()

    # ---- 4. Build per-race covariates (over all rows) ----
    race_wet = df.groupby("raceId")["is_wet"].first()
    wet_tensor = torch.tensor(
        [race_wet[r] for r in unique_races], dtype=torch.float32
    )

    # ---- 5. race_lengths: number of ranking entries per race ----
    race_lengths_series = ranking.groupby("raceId").size()
    race_lengths_tensor = torch.tensor(
        [race_lengths_series.get(r, 0) for r in unique_races], dtype=torch.long
    )

    # ---- 6. race_order: 0 = winner within each race ----
    df_sorted = ranking.sort_values(["raceId", "positionOrder"])
    df_sorted["_order"] = df_sorted.groupby("raceId").cumcount()
    race_order_tensor = torch.tensor(df_sorted["_order"].values, dtype=torch.long)

    driver_idx_tensor = torch.tensor(
        df_sorted["driverId"].map(driver_lookup).values, dtype=torch.long
    )
    cons_idx_tensor = torch.tensor(
        df_sorted["constructorId"].map(constructor_lookup).values, dtype=torch.long
    )
    season_idx_tensor = torch.tensor(
        df_sorted["year"].map(season_lookup).values, dtype=torch.long
    )
    circuit_idx_tensor = torch.tensor(
        df_sorted["circuitId"].map(circuit_lookup).values, dtype=torch.long
    )
    race_idx_tensor = torch.tensor(
        df_sorted["raceId"].map(race_lookup).values, dtype=torch.long
    )

    # ---- 7. Pit normalisation: robust z-score per season ----
    # Zero entries (no pit stops) are excluded from mean/std and set to pit_norm=0.
    # Extreme outliers (>99th %ile) are winsorised before computing mean/std to
    # prevent the 382 entries with spuriously large values (race-timing artefacts
    # in the source data, not actual pit durations) from inflating the std and
    # distorting all z-scores within a season.
    def _robust_pit_z(pit_ms):
        nonzero = pit_ms > 0
        if nonzero.sum() <= 1:
            return pd.Series(0.0, index=pit_ms.index)
        vals = pit_ms[nonzero].copy()
        cap = vals.quantile(0.99)
        clipped = pit_ms.clip(upper=cap)
        mean = vals.clip(upper=cap).mean()
        std = vals.clip(upper=cap).std()
        result = (clipped - mean) / (std + 1e-8)
        result[pit_ms == 0] = 0.0
        return result

    df_sorted["_pit_z"] = df_sorted.groupby("year")["total_pit_duration_ms"].transform(
        _robust_pit_z
    )
    pit_norm_tensor = torch.tensor(df_sorted["_pit_z"].values, dtype=torch.float32)

    # ---- 8. Assemble dataset ----
    ds = F1RankingDataset(
        driver_idx=driver_idx_tensor,
        cons_idx=cons_idx_tensor,
        season_idx=season_idx_tensor,
        circuit_idx=circuit_idx_tensor,
        race_idx=race_idx_tensor,
        pit_norm=pit_norm_tensor,
        wet=wet_tensor,
        race_order=race_order_tensor,
        race_lengths=race_lengths_tensor,
        n_drivers=len(unique_drivers),
        n_constructors=len(unique_constructors),
        n_seasons=len(unique_seasons),
        n_circuits=len(unique_circuits),
        n_races=len(unique_races),
        driver_map=driver_map,
        constructor_map=constructor_map,
    )

    # ---- 9. Assertions ----
    assert ds.n_races == 286, f"Expected 286 races, got {ds.n_races}"

    assert ds.race_lengths.sum().item() == ds.driver_idx.shape[0], (
        "race_lengths.sum() must equal N_entries"
    )

    for name in ["pit_norm", "wet"]:
        t = getattr(ds, name)
        assert not torch.isnan(t).any(), f"{name} contains NaN"
        assert not torch.isinf(t).any(), f"{name} contains inf"

    assert ds.driver_idx.dtype == torch.long, "driver_idx must be LongTensor"
    assert ds.cons_idx.dtype == torch.long, "cons_idx must be LongTensor"
    assert ds.season_idx.dtype == torch.long, "season_idx must be LongTensor"
    assert ds.circuit_idx.dtype == torch.long, "circuit_idx must be LongTensor"
    assert ds.race_idx.dtype == torch.long, "race_idx must be LongTensor"
    assert ds.race_order.dtype == torch.long, "race_order must be LongTensor"
    assert ds.race_lengths.dtype == torch.long, "race_lengths must be LongTensor"
    assert ds.pit_norm.dtype == torch.float32, "pit_norm must be FloatTensor"
    assert ds.wet.dtype == torch.float32, "wet must be FloatTensor"

    return ds
