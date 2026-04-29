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
    5, 6, 7, 8, 9, 10, 18, 19, 21, 22, 26, 28, 29, 31, 36,
    40, 41, 43, 44, 54, 61, 65, 66, 67, 72, 75, 82, 104, 107, 108, 130, 131,
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

    is_mech: torch.Tensor          # (N_all,) BoolTensor — True if mechanical DNF
    cons_idx_all: torch.Tensor     # (N_all,) LongTensor — constructor index for all rows
    season_idx_all: torch.Tensor   # (N_all,) LongTensor — season index for all original rows

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
    # driver-fault = everything else (not finished, not mechanical)

    # Ranking entries: finished + driver-fault DNFs, exclude mechanical DNFs.
    # Since driver-fault = ~finished & ~mechanical, and mechanical DNFs are
    # explicitly excluded from the Plackett-Luce ranking, the ranking mask is
    # simply all rows that are NOT mechanical DNFs.
    include_rank = ~is_mechanical
    ranking = df.loc[include_rank].copy()

    # ---- 4. Build per-race covariates (over all rows, before ranking filter) ----
    # wet: one value per race (consistent within a race)
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
    # Sort ranking entries within each race by positionOrder ascending
    ranking = ranking.sort_values(["raceId", "positionOrder"])
    ranking["_order"] = ranking.groupby("raceId").cumcount()
    race_order_tensor = torch.tensor(ranking["_order"].values, dtype=torch.long)

    # After sorting, indices are stable; build entry-level tensors from sorted df
    driver_idx_tensor = torch.tensor(
        ranking["driverId"].map(driver_lookup).values, dtype=torch.long
    )
    cons_idx_tensor = torch.tensor(
        ranking["constructorId"].map(constructor_lookup).values, dtype=torch.long
    )
    season_idx_tensor = torch.tensor(
        ranking["year"].map(season_lookup).values, dtype=torch.long
    )
    circuit_idx_tensor = torch.tensor(
        ranking["circuitId"].map(circuit_lookup).values, dtype=torch.long
    )
    race_idx_tensor = torch.tensor(
        ranking["raceId"].map(race_lookup).values, dtype=torch.long
    )

    season_idx_all_tensor = torch.tensor(
        df["year"].map(season_lookup).values, dtype=torch.long
    )

    # ---- 7. Pit normalisation: (x - mean) / (std + 1e-8) per season ----
    ranking["_pit_z"] = ranking.groupby("year")["total_pit_duration_ms"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    pit_norm_tensor = torch.tensor(ranking["_pit_z"].values, dtype=torch.float32)

    # ---- 8. Model 3 fields (N_all = all original rows) ----
    is_mech_tensor = torch.tensor(is_mechanical.values, dtype=torch.bool)
    cons_idx_all_tensor = torch.tensor(
        df["constructorId"].map(constructor_lookup).values, dtype=torch.long
    )

    # ---- 9. Assemble dataset ----
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
        is_mech=is_mech_tensor,
        cons_idx_all=cons_idx_all_tensor,
        season_idx_all=season_idx_all_tensor,
        n_drivers=len(unique_drivers),
        n_constructors=len(unique_constructors),
        n_seasons=len(unique_seasons),
        n_circuits=len(unique_circuits),
        n_races=len(unique_races),
        driver_map=driver_map,
        constructor_map=constructor_map,
    )

    # ---- 10. Assertions ----
    assert ds.n_races == 286, f"Expected 286 races, got {ds.n_races}"

    mech_mean = ds.is_mech.float().mean().item()
    assert 0.05 <= mech_mean <= 0.25, (
        f"is_mech mean {mech_mean:.4f} outside [0.05, 0.25]"
    )

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
    assert ds.is_mech.dtype == torch.bool, "is_mech must be BoolTensor"
    assert ds.cons_idx_all.dtype == torch.long, "cons_idx_all must be LongTensor"
    assert ds.season_idx_all.dtype == torch.long, "season_idx_all must be LongTensor"

    return ds
