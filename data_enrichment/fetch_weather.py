r"""
Open‑Meteo weather client with disk‑based caching.

Queries the `Open-Meteo Archive API`_ for historical daily weather at
a given latitude / longitude / date.  Results are cached in a CSV file
so that repeated runs only fetch new data.

.. _Open-Meteo Archive API: https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger("f1_enrichment.weather")

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES: List[str] = [
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "cloud_cover_mean",
]

REQUEST_DELAY_SEC = 1.0   # courtesy delay between API calls
REQUEST_TIMEOUT_SEC = 30
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(lat: float, lng: float, date_str: str) -> Tuple[float, float, str]:
    """Round lat/lng to 4 decimal places for cache stability."""
    return (round(lat, 4), round(lng, 4), date_str)


def load_cache(path: str) -> pd.DataFrame:
    """Load existing cache CSV, or return an empty DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(
        columns=["lat", "lng", "date"]
        + [f"weather_{v}" for v in DAILY_VARIABLES]
    )


def save_cache(cache_df: pd.DataFrame, path: str) -> None:
    """Persist the cache DataFrame to disk."""
    cache_df.to_csv(path, index=False)
    logger.info("Weather cache saved to %s (%d entries).", path, len(cache_df))


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def fetch_one(
    lat: float, lng: float, date_str: str
) -> Dict[str, Optional[float]]:
    """Query Open‑Meteo for one location and date.

    Args:
        lat: Latitude.
        lng: Longitude.
        date_str: ``"YYYY-MM-DD"``.

    Returns:
        Dictionary mapping ``weather_<variable>`` → value.  Missing
        values are ``None`` (e.g. when the API returns no data).
    """
    params: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lng,
        "start_date": date_str,
        "end_date": date_str,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT_SEC
            )
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            result: Dict[str, Optional[float]] = {}

            for var in DAILY_VARIABLES:
                values = daily.get(var, [])
                if values and values[0] is not None:
                    result[f"weather_{var}"] = float(values[0])
                else:
                    result[f"weather_{var}"] = None

            return result

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Open-Meteo API attempt %d/%d failed for (%.4f, %.4f, %s): %s",
                attempt, MAX_RETRIES, lat, lng, date_str, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                # Return all-None on final failure
                return {f"weather_{v}": None for v in DAILY_VARIABLES}


# ---------------------------------------------------------------------------
# Bulk fetch with caching
# ---------------------------------------------------------------------------


def fetch_all(
    queries: List[Tuple[float, float, str]],
    cache_path: str,
) -> pd.DataFrame:
    """Fetch weather for many (lat, lng, date) tuples, leveraging cache.

    Args:
        queries: List of ``(lat, lng, date_str)`` tuples.
        cache_path: Path to ``weather_cache.csv``.

    Returns:
        DataFrame with columns ``lat``, ``lng``, ``date``, and one
        ``weather_*`` column per variable.  Rows are keyed by the
        original query triple.
    """
    cache_df = load_cache(cache_path)
    cached_keys = {
        _cache_key(float(r["lat"]), float(r["lng"]), str(r["date"]))
        for _, r in cache_df.iterrows()
    }

    new_rows: List[Dict[str, Any]] = []
    n_total = len(queries)
    n_new = 0
    n_cached = 0

    for i, (lat, lng, date_str) in enumerate(queries):
        key = _cache_key(lat, lng, date_str)

        if key in cached_keys:
            n_cached += 1
            continue

        n_new += 1
        weather = fetch_one(key[0], key[1], key[2])
        row: Dict[str, Any] = {
            "lat": key[0],
            "lng": key[1],
            "date": key[2],
        }
        row.update(weather)
        new_rows.append(row)

        # Progress logging
        if n_new % 20 == 0:
            logger.info(
                "  Fetched %d/%d new weather entries (%d cached, %d total queries).",
                n_new, n_total, n_cached, n_total,
            )

        if i < n_total - 1:
            time.sleep(REQUEST_DELAY_SEC)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        save_cache(cache_df, cache_path)

    logger.info(
        "Weather fetch complete: %d from cache, %d new, %d total.",
        n_cached, n_new, len(cache_df),
    )
    return cache_df
