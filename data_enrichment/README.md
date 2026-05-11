# F1 Weather Data Enrichment

Fetches historical weather observations for every F1 race (2011–2024) from the
free [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api)
(no API key required).  Merges weather into the preprocessed race DataFrame
and adds engineered domain‑aware features ready for downstream modelling.

## Quick Start

```bash
.venv/Scripts/python.exe -m data_enrichment.run_enrichment
```

**First run:** fetches all 286 race‑weather entries (~5 min with 1s rate limit).  
**Subsequent runs:** loads from `weather_cache.csv` (instant).

## Architecture

| File | Role |
|------|------|
| `run_enrichment.py` | Orchestrator — extracts queries, runs fetch, merges, saves |
| `fetch_weather.py` | Open-Meteo client with disk‑based CSV caching |
| `enrich_features.py` | Join weather → DataFrame + engineered binary/categorical features |
| `weather_cache.csv` | Auto‑generated cache (286 rows) |

## Weather Variables Fetched

| Column | Source (Open-Meteo) | Mean | Min | Max |
|--------|---------------------|------|-----|-----|
| `temp_mean_c` | `temperature_2m_mean` | 19.7 °C | 6.0 | 32.8 |
| `precip_sum_mm` | `precipitation_sum` | 2.5 mm | 0.0 | 70.3 |
| `rain_sum_mm` | `rain_sum` | 2.5 mm | 0.0 | 70.3 |
| `wind_max_kmh` | `wind_speed_10m_max` | 18.2 km/h | 5.7 | 51.1 |
| `humidity_mean_pct` | `relative_humidity_2m_mean` | 71.2 % | 28 | 96 |
| `cloud_cover_mean_pct` | `cloud_cover_mean` | 52.7 % | 0 | 100 |

## Engineered Features

| Feature | Logic |
|---------|-------|
| `is_wet` | `precip_sum_mm > 0` |
| `is_very_wet` | `precip_sum_mm >= 5` |
| `weather_type` | `dry` / `wet` / `very_wet` |
| `is_hot` | `temp_mean_c >= 30` |
| `is_cold` | `temp_mean_c <= 10` |
| `is_windy` | `wind_max_kmh >= 40` |
| `is_humid` | `humidity_mean_pct >= 80` |

## Race Weather Distribution

| Weather Type | Races | Share |
|-------------|-------|-------|
| dry | 143 | 50.0 % |
| wet | 102 | 35.7 % |
| very_wet | 41 | 14.3 % |

## Output

`data_preprocessing/f1_enriched.csv` — 5,980 rows × 26 columns (original 13 +
6 weather + 7 engineered).  Ready for the TrueSkill `context` dictionary hook
and the upcoming Pyro Bayesian model with weather covariates.
