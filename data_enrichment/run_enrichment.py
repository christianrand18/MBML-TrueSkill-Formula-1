r"""
Weather Data Enrichment — Entry Point
=======================================

Fetches historical weather for every F1 race (2011–2024) from the free
`Open-Meteo Archive API`_, merges it into the preprocessed race DataFrame,
and creates a new ``f1_enriched.csv`` ready for downstream models.

.. _Open-Meteo Archive API: https://open-meteo.com/en/docs/historical-weather-api

Usage
-----
.. code-block:: bash

    .venv/Scripts/python.exe -m data_enrichment.run_enrichment
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import pandas as pd

from data_enrichment.enrich_features import enrich
from data_enrichment.fetch_weather import fetch_all

logger = logging.getLogger("f1_enrichment")


class WeatherEnrichmentPipeline:
    """Orchestrates weather fetching, merging, and feature engineering.

    Args:
        model_data_path: Path to ``f1_model_ready.csv``.
        output_path: Where to write ``f1_enriched.csv``.
        cache_dir: Directory for ``weather_cache.csv``.
    """

    def __init__(
        self,
        model_data_path: str,
        output_path: str,
        cache_dir: str,
    ) -> None:
        self._model_data_path = model_data_path
        self._output_path = output_path
        self._cache_path = os.path.join(cache_dir, "weather_cache.csv")

    def run(self) -> None:
        """Execute the full enrichment pipeline."""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)

        # -- Load data ----------------------------------------------------
        logger.info("Loading preprocessed data from %s", self._model_data_path)
        df = pd.read_csv(self._model_data_path)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("  → %d rows, %d unique races.", len(df), df["raceId"].nunique())

        # -- Extract unique (lat, lng, date) queries ----------------------
        queries = self._extract_queries(df)
        logger.info("Unique weather queries: %d.", len(queries))

        # -- Fetch weather (cached) ---------------------------------------
        weather_df = fetch_all(queries, self._cache_path)

        # -- Merge + engineer features ------------------------------------
        enriched = enrich(df, weather_df)

        # -- Save ---------------------------------------------------------
        enriched.to_csv(self._output_path, index=False)
        logger.info(
            "Enriched dataset saved to %s (%d rows × %d columns).",
            self._output_path,
            len(enriched),
            len(enriched.columns),
        )

        # Summary
        self._print_summary(enriched)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_queries(df: pd.DataFrame) -> List[Tuple[float, float, str]]:
        """Return unique ``(lat, lng, date_str)`` tuples, one per race.

        Groups by ``raceId`` and takes the first lat/lng/date (all rows
        within a race share the same circuit and date).
        """
        races = df.groupby("raceId").agg(
            lat=("lat", "first"),
            lng=("lng", "first"),
            date=("date", "first"),
        ).reset_index()
        queries: List[Tuple[float, float, str]] = []
        seen: set = set()
        for _, row in races.iterrows():
            key = (round(row["lat"], 4), round(row["lng"], 4), str(row["date"].date()))
            if key not in seen:
                seen.add(key)
                queries.append(key)
        return queries

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print a summary of the enriched dataset."""
        weather_cols = [
            "temp_mean_c", "precip_sum_mm", "rain_sum_mm",
            "wind_max_kmh", "humidity_mean_pct", "cloud_cover_mean_pct",
        ]
        engineered = ["is_wet", "is_very_wet", "is_hot", "is_cold", "is_windy", "is_humid"]

        n_races = df["raceId"].nunique()
        n_wet = df.groupby("raceId")["is_wet"].first().sum() if "is_wet" in df.columns else 0

        logger.info("=" * 55)
        logger.info("WEATHER ENRICHMENT SUMMARY")
        logger.info("=" * 55)
        logger.info("  Races enriched:        %d", n_races)
        logger.info("  New weather columns:   %d", len([c for c in weather_cols if c in df.columns]))
        logger.info("  Engineered features:   %d", len([c for c in engineered if c in df.columns]))
        logger.info("  Wet race fraction:     %.1f %%", n_wet * 100)

        for col in weather_cols:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    logger.info(
                        "  %-22s  mean=%6.1f  min=%6.1f  max=%6.1f",
                        col, vals.mean(), vals.min(), vals.max(),
                    )
        logger.info("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure logging and launch the enrichment pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, ".."))

    pipeline = WeatherEnrichmentPipeline(
        model_data_path=os.path.join(
            project_root, "data_preprocessing", "f1_model_ready.csv"
        ),
        output_path=os.path.join(
            project_root, "data_preprocessing", "f1_enriched.csv"
        ),
        cache_dir=os.path.join(project_root, "data_enrichment"),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
