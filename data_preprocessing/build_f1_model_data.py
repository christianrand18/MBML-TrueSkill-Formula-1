r"""
Build a modeling-ready DataFrame from the Kaggle Formula 1 dataset.

Performs the following pipeline:
1. Load CSVs, replacing \N with NaN.
2. Filter results to races from 2011 onwards.
3. Aggregate pit stop strategy features per race/driver.
4. Merge circuit lat/lng and convert race date to datetime.
5. Select only the 13 columns needed for the Bayesian PGM.
6. Clean numeric types and drop rows missing critical targets.

Output: f1_model_ready.csv
"""

import os
import numpy as np
import pandas as pd

NA_VALUES = [r"\N"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from DATA_DIR, treating backslash-N as NaN."""
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)


def main() -> None:
    # ----------------------------------------------------------------
    # Step 1: Data Loading & Null Handling
    # ----------------------------------------------------------------
    circuits = load_csv("circuits.csv")
    constructors = load_csv("constructors.csv")  # kept for schema validation
    drivers = load_csv("drivers.csv")  # kept for schema validation
    pit_stops = load_csv("pit_stops.csv")
    races = load_csv("races.csv")
    results = load_csv("results.csv")

    # ----------------------------------------------------------------
    # Step 2: Filter by Year  (pit_stops only starts in 2011)
    # ----------------------------------------------------------------
    races_subset = races[["raceId", "year", "circuitId", "date"]]
    df = results.merge(races_subset, on="raceId", how="inner")
    df = df[df["year"] >= 2011].copy()
    print(f"[Step 2] Filtered to {len(df)} result rows (year >= 2011).")

    # ----------------------------------------------------------------
    # Step 3: Process Strategy — Pit Stop Aggregation
    # ----------------------------------------------------------------
    pit_agg = pit_stops.groupby(["raceId", "driverId"], as_index=False).agg(
        total_pit_duration_ms=("milliseconds", "sum"),
        num_pit_stops=("stop", "count"),
    )

    df = df.merge(pit_agg, on=["raceId", "driverId"], how="left")

    # Drivers with zero pit stops (e.g. DNF on lap 1) get 0
    df["total_pit_duration_ms"] = df["total_pit_duration_ms"].fillna(0).astype(int)
    df["num_pit_stops"] = df["num_pit_stops"].fillna(0).astype(int)
    print(f"[Step 3] Pit stop aggregation merged.")

    # ----------------------------------------------------------------
    # Step 4: Merge Circuit coords & convert date
    # ----------------------------------------------------------------
    circuits_subset = circuits[["circuitId", "lat", "lng"]]
    df = df.merge(circuits_subset, on="circuitId", how="left")

    df["date"] = pd.to_datetime(df["date"])

    # Ensure ID columns stay as integers
    df["driverId"] = df["driverId"].astype(int)
    df["constructorId"] = df["constructorId"].astype(int)
    print(f"[Step 4] Merged circuit lat/lng, parsed dates.")

    # ----------------------------------------------------------------
    # Step 5: Feature Selection — keep only modelling columns
    # ----------------------------------------------------------------
    FINAL_COLUMNS = [
        "raceId",
        "year",
        "date",
        "circuitId",
        "lat",
        "lng",
        "driverId",
        "constructorId",
        "grid",
        "total_pit_duration_ms",
        "num_pit_stops",
        "positionOrder",
        "statusId",
    ]
    df = df[FINAL_COLUMNS]
    print(f"[Step 5] Kept {len(FINAL_COLUMNS)} columns: {FINAL_COLUMNS}")

    # ----------------------------------------------------------------
    # Step 6: Final Cleanup — numeric coercion & drop missing targets
    # ----------------------------------------------------------------
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["positionOrder", "grid"])
    print(
        f"[Step 6] Dropped {before - len(df)} rows with missing positionOrder/grid. "
        f"Remaining: {len(df)}."
    )

    # ----------------------------------------------------------------
    # Write output
    # ----------------------------------------------------------------
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "f1_model_ready.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows × {len(df.columns)} columns to {output_path}")


if __name__ == "__main__":
    main()
