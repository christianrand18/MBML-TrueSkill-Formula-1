r"""
Merge weather data into the F1 race DataFrame and engineer domain‑aware
weather features.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("f1_enrichment.features")

# ---------------------------------------------------------------------------
# Weather column mapping  (Open‑Meteo → clean name)
# ---------------------------------------------------------------------------

WEATHER_RENAME: Dict[str, str] = {
    "weather_temperature_2m_mean": "temp_mean_c",
    "weather_precipitation_sum": "precip_sum_mm",
    "weather_rain_sum": "rain_sum_mm",
    "weather_wind_speed_10m_max": "wind_max_kmh",
    "weather_relative_humidity_2m_mean": "humidity_mean_pct",
    "weather_cloud_cover_mean": "cloud_cover_mean_pct",
}


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------


def enrich(
    race_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join weather observations into the race DataFrame and add engineered
    weather features.

    Args:
        race_df: ``f1_model_ready.csv`` DataFrame (must have ``lat``,
            ``lng``, and a ``date`` column convertible to datetime).
        weather_df: Cached weather DataFrame from ``fetch_weather``
            (must have ``lat``, ``lng``, ``date`` columns plus ``weather_*``
            measurement columns).

    Returns:
        Enriched DataFrame with original columns + weather columns +
        engineered features (``is_wet``, ``weather_type``, …).
    """
    df = race_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Round lat/lng to match cache keys
    if "lat_raw" not in df.columns:
        df["_lat_key"] = df["lat"].round(4)
        df["_lng_key"] = df["lng"].round(4)
        df["_date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    else:
        df["_lat_key"] = df["lat_raw"].round(4)
        df["_lng_key"] = df["lng_raw"].round(4)
        df["_date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Normalise weather cache keys
    weather = weather_df.copy()
    weather["_lat_key"] = weather["lat"].round(4)
    weather["_lng_key"] = weather["lng"].round(4)
    weather["_date_str"] = weather["date"].astype(str)

    # Drop any duplicate rows in weather (keep first)
    weather = weather.drop_duplicates(
        subset=["_lat_key", "_lng_key", "_date_str"], keep="first"
    )

    # Merge
    orig_len = len(df)
    df = df.merge(
        weather.drop(columns=["lat", "lng", "date"]),
        on=["_lat_key", "_lng_key", "_date_str"],
        how="left",
    )

    # Clean up temp keys
    df = df.drop(columns=["_lat_key", "_lng_key", "_date_str"])

    if len(df) != orig_len:
        logger.warning("Row count changed after merge: %d → %d", orig_len, len(df))

    # Rename weather columns
    df = df.rename(columns=WEATHER_RENAME)

    # Log coverage
    for col in WEATHER_RENAME.values():
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                logger.warning("  %s: %d/%d missing", col, missing, len(df))

    # ---- Engineered features --------------------------------------------

    df = _add_engineered_features(df)

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain‑aware weather binary/categorical features.

    Args:
        df: Merged DataFrame with ``precip_sum_mm``, ``temp_mean_c``,
            ``wind_max_kmh`` columns.

    Returns:
        DataFrame with added columns.
    """
    # -- Wet‑race flags
    if "precip_sum_mm" in df.columns:
        precip = df["precip_sum_mm"].fillna(0)
        df["is_wet"] = (precip > 0).astype(int)
        df["is_very_wet"] = (precip >= 5).astype(int)

        conditions = [
            (precip == 0),
            (precip > 0) & (precip < 5),
            (precip >= 5),
        ]
        choices = ["dry", "wet", "very_wet"]
        df["weather_type"] = pd.Series(
            index=df.index, dtype="object"
        )
        for cond, choice in zip(conditions, choices):
            df.loc[cond, "weather_type"] = choice

    # -- Temperature flags
    if "temp_mean_c" in df.columns:
        temp = df["temp_mean_c"].fillna(20)
        df["is_hot"] = (temp >= 30).astype(int)
        df["is_cold"] = (temp <= 10).astype(int)

    # -- Wind flag
    if "wind_max_kmh" in df.columns:
        wind = df["wind_max_kmh"].fillna(0)
        df["is_windy"] = (wind >= 40).astype(int)

    # -- Humidity flag
    if "humidity_mean_pct" in df.columns:
        hum = df["humidity_mean_pct"].fillna(50)
        df["is_humid"] = (hum >= 80).astype(int)

    return df
