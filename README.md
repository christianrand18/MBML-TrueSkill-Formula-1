# MBML-TrueSkill-Formula-1

Bayesian probabilistic modelling of Formula 1 driver and constructor skill
(2011–2024).  Built with `trueskill`, `pyro`, `pandas`, and `torch`.

---

## Project Overview

This project implements a full model‑based machine learning pipeline:

1. **Data preprocessing** — merges, cleans, and transforms 14 Kaggle CSV files
   into a modelling‑ready DataFrame.
2. **Baseline TrueSkill model** — online Bayesian rating system from Microsoft
   Research, treating each F1 entry as a two‑player team [driver, constructor].
3. **Exploratory data analysis** — 16 publication‑quality visualisations
   covering drivers, constructors, circuits, race dynamics, and skill
   trajectories.
4. **Model evaluation** — chronological cross‑validation across 5 competing
   models (TrueSkill, Grid, Elo, PreviousSeason, Random) with 7 predictive
   metrics.
5. **Weather data enrichment** — fetches historical weather from the free
   Open‑Meteo API for all 286 races (no API key required).
6. **Pyro Bayesian model** — pairwise‑ranking SVI model with grid‑position
   covariates and weather‑dependent noise, outperforming both TrueSkill and the
   Grid baseline.

All models are evaluated head‑to‑head on the same chronological test folds.

---

## Project Structure

```
├── pyproject.toml                       # Dependencies (uv)
├── main.py                              # Placeholder entry point
│
├── data_preprocessing/                  # Stage 1 — clean dataset
│   ├── build_f1_model_data.py           # Merge 6 CSVs, handle \N, filter 2011+, aggregate pit stops
│   ├── read_columns.py                  # Quick column inspector (nrows=0)
│   ├── f1_model_ready.csv               # 5,980 rows × 13 columns
│   └── f1_enriched.csv                  # +13 weather/engineered columns (26 total)
│
├── models/                              # Stage 2 — skill‑rating models
│   ├── f1_trueskill_baseline.py         # TrueSkill: online EP, driver+constructor teams, 286 races
│   └── pyro_backend/                    # Phase B — Pyro Bayesian model
│       ├── run_pyro_model.py            # Orchestrator: train + export + compare
│       ├── pyro_model.py                # Pairwise ranking model + SVI guide (static + temporal)
│       ├── pyro_evaluator.py            # PyroSkillPredictor — plugs into evaluation CV
│       └── data_preparation.py          # Converts 5,980 entries → 59,839 pairwise training examples
│
├── exploration/                         # Stage 3 — EDA & visualisation
│   ├── f1_data_exploration.py           # Orchestrator: 16 analyses + figures
│   ├── analysis.py                      # 20 statistical computation functions
│   └── visualisations.py               # 16 seaborn/matplotlib plotting functions
│
├── evaluation/                          # Phase A — model validation
│   ├── run_evaluation.py                # Orchestrator: 10‑fold chronological CV
│   ├── metrics.py                       # 7 prediction metrics (pairwise acc, Spearman, MRR, MSE, …)
│   ├── baselines.py                     # 5 SkillPredictor classes (Grid, Elo, PrevSeason, Random, TrueSkill)
│   ├── validator.py                     # ChronologicalValidator — train/test split by season
│   └── reporter.py                      # 8 comparison figures + Markdown report
│
├── data_enrichment/                     # Phase C — weather integration
│   ├── run_enrichment.py                # Orchestrator: fetch → merge → engineer
│   ├── fetch_weather.py                 # Open‑Meteo API client with CSV caching
│   ├── enrich_features.py               # Merge + 7 engineered weather features
│   └── weather_cache.csv                # 286 rows × 9 columns (cached API responses)
│
├── outputs/                             # All generated outputs
│   ├── ratings/                         # TrueSkill final ratings
│   │   ├── driver_ratings.csv
│   │   └── constructor_ratings.csv
│   ├── history/                         # TrueSkill per‑race snapshots
│   │   └── driver_rating_history.csv    # 15,921 rows
│   ├── exploration/figures/             # 16 EDA figures (.png)
│   ├── evaluation/                      # CV metrics + comparison figures
│   └── pyro_model/                      # Pyro posteriors + comparison metrics
│
├── prompts/                             # Task specifications
│   ├── prompt_01.md                     # Data preprocessing requirements
│   └── prompt_02.md                     # TrueSkill model requirements
│
└── data/                                # Raw Kaggle F1 dataset (14 CSVs, not tracked in git)
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager

```bash
# Clone and set up
git clone <repo-url>
cd MBML-TrueSkill-Formula-1
uv sync
```

### Pipeline (in order)

```bash
# 1. Build the clean dataset
.venv/Scripts/python.exe data_preprocessing/build_f1_model_data.py
# → data_preprocessing/f1_model_ready.csv

# 2. Run the TrueSkill baseline model
.venv/Scripts/python.exe -m models.f1_trueskill_baseline
# → outputs/ratings/ + outputs/history/

# 3. Explore the data
.venv/Scripts/python.exe -m exploration.f1_data_exploration
# → outputs/exploration/figures/ (16 plots)

# 4. Evaluate all models
.venv/Scripts/python.exe -m evaluation.run_evaluation
# → outputs/evaluation/ (metrics + comparison figures)

# 5. Fetch weather data (first run: ~5 min; cached thereafter)
.venv/Scripts/python.exe -m data_enrichment.run_enrichment
# → data_preprocessing/f1_enriched.csv

# 6. Run the Pyro Bayesian model
.venv/Scripts/python.exe -m models.pyro_backend.run_pyro_model
# → outputs/pyro_model/ (posteriors + comparison)
```

---

## Mathematical Foundation

The project implements two Bayesian approaches to the F1 skill‑rating problem.

### TrueSkill (Online Expectation Propagation)

Each driver and constructor has a latent skill $\theta \sim \mathcal{N}(\mu, \sigma^2)$.
For a race with $N$ competitors, the forward model is:

$$t_j = \theta_{\text{driver}(j)} + \theta_{\text{constructor}(j)} + \varepsilon_j, \qquad \varepsilon_j \sim \mathcal{N}(0, \beta^2)$$

The observed ranking constrains $t_1 > t_2 > \dots > t_N$.  The joint posterior
$p(\boldsymbol{\theta} \mid \text{ranks})$ is intractable for $N \geq 3$, so
TrueSkill approximates it via Expectation Propagation (moment‑matching on the
factor graph).  Between races, skills drift via $\theta_{t+1} \sim \mathcal{N}(\theta_t, \tau^2)$.

**Key parameters:** μ₀ = 25, σ₀ = 25/3 ≈ 8.33, β = 25/6 ≈ 4.17, τ = 25/300 ≈ 0.083.

See `models/README.md` for the full mathematical exposition including worked
examples, the EP factor graph, and citations.

### Pyro SVI (Pairwise‑Ranking with Covariates)

The Pyro model reframes the problem as batch inference on all pairwise
comparisons.  For every ordered pair $(i, j)$ where $i$ finished ahead of $j$
in a race:

$$\mathbb{P}(i \succ j) = \Phi\!\left(
    \frac{s_i - s_j + \beta_g (g_i - g_j)}{\sqrt{2}\,\beta_{\text{perf}}}
\right)$$

where $s_i = \theta_{\text{driver}(i)} + \theta_{\text{constructor}(i)}$,
$g_i$ is the normalised grid position, and $\Phi$ is the standard Normal CDF.

Inference uses Stochastic Variational Inference (SVI) with a mean‑field
Gaussian guide.  The static variant learns 100 latent parameters from 59,839
pairwise observations in ~12 seconds.  A temporal variant with per‑season
random‑walk skills (1,400+ parameters) is implemented but computationally heavy.

---

## Key Results

### Model Comparison (10‑fold Chronological CV, 2015–2024)

| Model | Pairwise Acc | Top‑1 Win | Spearman ρ | MSE ↓ |
|-------|:-----------:|:---------:|:----------:|:-----:|
| **Pyro (SVI)** | **0.755** | **0.475** | **0.640** | **24.3** |
| Grid | 0.730 | 0.422 | 0.569 | 29.1 |
| Elo | 0.678 | 0.389 | 0.496 | 34.2 |
| PrevSeason | 0.669 | 0.543 | 0.517 | 31.5 |
| TrueSkill | 0.659 | 0.389 | 0.455 | 36.8 |
| Random | 0.506 | 0.032 | 0.012 | 66.9 |

**Takeaways:**

- **Pyro is the only model to beat the Grid baseline** — the grid‑position
  covariate ($\beta_g = -3.55$) strongly modulates predictions.
- TrueSkill's strength is not in point predictions alone, but in providing
  **full posterior uncertainties** ($\mu, \sigma$) for risk estimation.
- Grid position is the single strongest predictor of race outcome in F1,
  explaining why all models that ignore it underperform.
- The pairwise‑ranking SVI approach achieves better predictive accuracy than
  TrueSkill's online EP on this dataset.

### Data Summary (2011–2024)

| Metric | Value |
|--------|-------|
| Races | 286 |
| Unique drivers | 77 |
| Unique constructors | 23 |
| Winningest driver | Lewis Hamilton (91 wins) |
| Winningest constructor | Mercedes (120 wins) |
| DNF rate | 17.1% |
| Avg pit stops/race | 1.90 |
| Wet races | 50% (143/286) |
| Very wet races | 14% (41/286) |

---

## Running Individual Modules

Each module has its own README with detailed documentation:

| Module | Command | Description |
|--------|---------|-------------|
| Data prep | `python data_preprocessing/build_f1_model_data.py` | Build clean CSV from raw data |
| TrueSkill | `python -m models.f1_trueskill_baseline` | Train TrueSkill, export ratings |
| Exploration | `python -m exploration.f1_data_exploration` | 16 EDA figures |
| Evaluation | `python -m evaluation.run_evaluation` | 10‑fold CV, all models |
| Weather | `python -m data_enrichment.run_enrichment` | Fetch + merge weather |
| Pyro model | `python -m models.pyro_backend.run_pyro_model` | Train Pyro, compare |

All commands should be run from the project root.  Use the `.venv` Python
interpreter (`.venv/Scripts/python.exe` on Windows).

---

## Dependencies

```
numpy, pandas          — data handling
trueskill              — baseline Bayesian rating
pyro-ppl, torch        — Pyro Bayesian model + SVI
matplotlib, seaborn    — visualisation
requests               — weather API calls
```

Managed by [uv](https://docs.astral.sh/uv/) via `pyproject.toml`.

---

## License

This project uses the [Kaggle Formula 1 dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
(via Ergast API).  Weather data is from [Open‑Meteo](https://open-meteo.com/)
(free, no API key).
