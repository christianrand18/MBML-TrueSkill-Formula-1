# F1 TrueSkill Baseline Model

A production-ready Bayesian skill-rating pipeline that evaluates Formula 1 driver
and constructor performance using the `trueskill` Python package. Races are
modelled as free-for-all matches where each entry is a two‑player team:
`[driver, constructor]`.

## Quick Start

```bash
# Install dependencies (once)
uv add trueskill

# Run from the project root
.venv/Scripts/python.exe models/f1_trueskill_baseline.py
```

The pipeline reads `data_preprocessing/f1_model_ready.csv` and writes results to
`outputs/`.

---

## Architecture

| Component | File | Role |
|-----------|------|------|
| `SkillEvaluator` (ABC) | `f1_trueskill_baseline.py:53` | Abstract backend interface — swap `trueskill` for `Pyro` later |
| `TrueSkillEvaluator` | `f1_trueskill_baseline.py:81` | Concrete `trueskill` wrapper; accepts a `context` dict for future weather/tyre features |
| `F1RatingEnvironment` | `f1_trueskill_baseline.py:126` | Stateful store of per-entity `Rating` objects with lazy initialisation and a name registry |
| `RaceProcessor` | `f1_trueskill_baseline.py:258` | Builds `(driver, constructor)` teams per race, maps `positionOrder` → ranks, clones constructor ratings for multi‑car teams, averages posteriors |
| `F1SkillPipeline` | `f1_trueskill_baseline.py:344` | Orchestrator: loads data → sorts chronologically → processes 286 races → exports results |

### Data Flow

```
f1_model_ready.csv                     outputs/
     │                              ┌───────┴───────┐
     ▼                              │  ratings/     │  history/
F1SkillPipeline.run()               │  driver_ratings.csv     driver_rating_history.csv
  ├─ _load_and_prepare()            │  constructor_ratings.csv
  ├─ for each raceId:               └───────────────┘
  │    RaceProcessor.process_race()
  │      ├─ build teams
  │      ├─ TrueSkillEvaluator.update_skills()
  │      └─ env.apply_*_posteriors()
  ├─ _export_results()
  └─ _log_leaderboard()
```

---

## Key Design Decisions

### Constructor rating sharing (cloning + averaging)

A constructor fields two cars per race.  The `trueskill` library (v0.4.5)
returns per‑team posteriors rather than per‑player marginals when the same
`Rating` object appears on multiple teams.  To work around this:

1. The constructor's current rating is **cloned** for each car before the match.
2. `trueskill.rate()` returns independent posteriors for each clone.
3. The `mu` and `sigma` of the constructor's clones are **averaged** to produce
   a single updated constructor rating.

Drivers are updated directly (one posterior per driver per race).

### Extensibility hook

Every `SkillEvaluator` method and `RaceProcessor.process_race()` accepts an
optional `context: Dict[str, Any]` parameter.  The baseline `TrueSkillEvaluator`
ignores it, but a future `PyroSkillEvaluator` can consume weather, tyre
compound, track temperature, etc. as model covariates.

### TrueSkill parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `mu` | 25.0 | Initial skill estimate for new entrants |
| `sigma` | 25/3 ≈ 8.33 | Initial uncertainty |
| `beta` | 25/6 ≈ 4.17 | Skill difference for ~80 % win probability |
| `tau` | 25/300 ≈ 0.083 | Per‑race dynamics (drift) factor |
| `draw_probability` | 0.0 | F1 does not have traditional draws |

---

## Output Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `outputs/ratings/driver_ratings.csv` | 77 | `driverId`, `driverName`, `mu`, `sigma` | Final posterior rankings (sorted by mu desc) |
| `outputs/ratings/constructor_ratings.csv` | 23 | `constructorId`, `constructorName`, `mu`, `sigma` | Final constructor rankings |
| `outputs/history/driver_rating_history.csv` | 15 921 | `raceId`, `year`, `date`, `driverId`, `driverName`, `mu`, `sigma` | Per‑race rating snapshots for trajectory plots |

---

## Results (2011–2024)

### Top 10 Drivers

| Rank | Driver | mu | sigma |
|------|--------|----|-------|
| 1 | Nico Rosberg | 35.55 | 0.78 |
| 2 | Max Verstappen | 34.42 | 0.76 |
| 3 | Lando Norris | 33.98 | 0.75 |
| 4 | Oscar Piastri | 32.75 | 0.98 |
| 5 | Charles Leclerc | 31.29 | 0.74 |
| 6 | Lewis Hamilton | 31.12 | 0.73 |
| 7 | Carlos Sainz | 30.32 | 0.72 |
| 8 | Sebastian Vettel | 29.99 | 0.73 |
| 9 | Oliver Bearman | 29.34 | 3.28 |
| 10 | Fernando Alonso | 29.29 | 0.72 |

### Top 10 Constructors

| Rank | Constructor | mu | sigma |
|------|-------------|----|-------|
| 1 | Red Bull | 33.39 | 0.74 |
| 2 | Mercedes | 32.80 | 0.73 |
| 3 | Ferrari | 30.63 | 0.73 |
| 4 | Racing Point | 27.80 | 1.05 |
| 5 | McLaren | 27.79 | 0.73 |
| 6 | Force India | 27.40 | 0.73 |
| 7 | Lotus F1 | 27.30 | 0.83 |
| 8 | AlphaTauri | 26.04 | 0.82 |
| 9 | RB F1 Team | 24.81 | 1.29 |
| 10 | Renault | 24.72 | 0.76 |

---

## Future Enhancements

- **Weather integration** — merge track‑side weather data via the `context`
  parameter and build a `PyroSkillEvaluator` with weather covariates.
- **Tyre compound features** — incorporate tyre strategy as a performance
  modifier.
- **Temporal decay** — higher `tau` or explicit time‑weighted updates
  to prioritise recent form.
- **DNF handling** — model non‑finishes via censored observations instead of
  treating them as last‑place finishes.
