# F1 Model Evaluation & Comparison

Chronological cross‑validation framework for benchmarking F1 skill‑rating
models.  Five models are evaluated head‑to‑head across 10 test seasons
(2015–2024) using seven predictive metrics.

## Quick Start

```bash
.venv/Scripts/python.exe -m evaluation.run_evaluation
```

Outputs go to `outputs/evaluation/`.

---

## Architecture

| File | Role |
|------|------|
| `run_evaluation.py` | Orchestrator — loads data, builds predictors, runs CV, generates report |
| `metrics.py` | Seven per‑race metrics implementing `y_true` vs `y_pred` comparison |
| `baselines.py` | Five `SkillPredictor` classes: Grid, Elo, PrevSeason, Random, TrueSkill |
| `validator.py` | `ChronologicalValidator` — splits by season, trains each model, evaluates |
| `reporter.py` | Figures (bar charts, fold‑consistency lines) + Markdown report |

---

## Models Evaluated

| Model | Skill = | Training |
|-------|---------|----------|
| **TrueSkill** | μ from Bayesian PGM at end of training window | Re‑reads rating history CSV |
| **Grid** | `−grid_position` (pole = highest) | None (directly from data) |
| **Elo** | Pairwise Elo rating, K=32, initial 1500 | Chronological pairwise updates |
| **PrevSeason** | Previous year's championship points | Loads `results.csv` points |
| **Random** | Uniform random | None (lower bound) |

---

## Metrics

| Metric | Range | Interpretation |
|--------|-------|---------------|
| `pairwise_accuracy` | [0, 1] | Fraction of driver pairs correctly ordered by skill |
| `top_1_accuracy` | {0, 1} | Is the highest‑rated driver the race winner? |
| `top_3_accuracy` | {0, 1} | Is the highest‑rated driver on the podium? |
| `top_5_accuracy` | {0, 1} | Is the highest‑rated driver in the top 5? |
| `spearman_rho` | [-1, 1] | Rank correlation between predicted and actual order |
| `mrr` | (0, 1] | Reciprocal rank of the actual winner |
| `mse_position` | [0, ∞) | Squared error of predicted rank vs actual position |

---

## Validation Strategy

**Chronological expanding‑window CV:**

```
Fold  1: train 2011–2014  →  test 2015
Fold  2: train 2011–2015  →  test 2016
...
Fold 10: train 2011–2023  →  test 2024
```

Each model is retrained from scratch on the training set for every fold.
Temporal leakage is impossible — the training window always ends before the
test season begins.

---

## Output Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Long‑form: model × fold × metric |
| `metrics_summary_pivoted.csv` | Mean ± std per model |
| `model_comparison_*.png` | Bar charts comparing models per metric |
| `fold_consistency_*.png` | Line plots showing metric stability over time |
| `validation_report.md` | Full Markdown report with tables |

---

## Key Results (2015–2024)

| Model | Pairwise Acc | Top-1 | Spearman | MSE |
|-------|-------------|-------|----------|-----|
| **Grid** | 0.730 | 0.422 | 0.569 | 29.1 |
| **TrueSkill** | 0.659 | 0.389 | 0.455 | 36.8 |
| **Elo** | 0.678 | 0.389 | 0.496 | 34.2 |
| **PrevSeason** | 0.669 | 0.543 | 0.517 | 31.5 |
| **Random** | 0.506 | 0.032 | 0.012 | 66.9 |

### Key Insights

- **Grid position is the strongest single predictor** of race outcome
  (pairwise accuracy 0.730).  This is expected — qualifying pace strongly
  correlates with race pace, and grid position directly reflects car+driver
  performance at that circuit.
- **TrueSkill's advantage is not in point predictions alone** but in the
  full Bayesian posterior: μ and σ allow risk estimation, uncertainty
  quantification, and probabilistic decision‑making that point‑estimate
  models cannot provide.
- The next stage (Pyro backend with weather covariates) can use **grid
  position as a covariate** to close the gap with the Grid baseline while
  retaining full posterior uncertainty.
- **Elo and PrevSeason** are simple, competitive baselines — but neither
  provides uncertainty estimates.
