# F1 Pyro Bayesian Skill Model

A pairwise‑ranking Bayesian model built with [Pyro](https://pyro.ai) that
extends the TrueSkill baseline with **grid‑position covariates** and
**weather‑dependent performance noise**.  Trained via Stochastic Variational
Inference (SVI) on 59,839 ordered driver pairs from 286 F1 races (2011–2024).

## Quick Start

```bash
.venv/Scripts/python.exe -m models.pyro_backend.run_pyro_model
```

Trains the static model (~3000 SVI steps, ~12 seconds), exports posterior
ratings, and runs a 10‑fold chronological cross‑validation comparison against
TrueSkill, Grid, and Random baselines.

## Architecture

| File | Role |
|------|------|
| `run_pyro_model.py` | Orchestrator — data loading, training, export, comparison |
| `pyro_model.py` | Model + guide: static (77+23 params) and temporal (random‑walk, 1400+ params) |
| `pyro_evaluator.py` | `PyroSkillPredictor` — plugs into the `evaluation/` CV framework |
| `data_preparation.py` | Converts 5,980 race entries into 59,839 pairwise training examples |

## Model Specification

### Generative Model

For each race, let *N* drivers be ranked by `positionOrder`.  Every ordered
pair *(i, j)* where *i* finishes ahead of *j* contributes a Bernoulli
observation with probit link:

$$\mathbb{P}(i \succ j) = \Phi\!\left(\frac{s_i - s_j + \beta_g \cdot (g_i - g_j)}{\sqrt{2}\,\beta_{\text{perf}}}\right)$$

where:
- $s_i = \theta_{\text{driver}(i)} + \theta_{\text{constructor}(i)}$
- $g_i$ = normalised grid position
- $\Phi$ = standard Normal CDF
- $\beta_{\text{perf}}$ = performance noise (fixed at 25/6 ≈ 4.17)

### Priors

| Parameter | Prior | Count |
|-----------|-------|:-----:|
| $\theta_{\text{driver}}$ | $\mathcal{N}(0, 10)$ | 77 |
| $\theta_{\text{constructor}}$ | $\mathcal{N}(0, 10)$ | 23 |
| $\beta_g$ (grid coefficient) | $\mathcal{N}(0, 1)$ | 1 |

### Inference — SVI

- **Guide:** Mean‑field Gaussian (AutoNormal) — 202 variational parameters
- **Optimiser:** Clipped Adam, lr = 0.005
- **Iterations:** 3,000
- **Mini‑batch:** 1,024 pairs / step (from 59,839 total)
- **Convergence:** ELBO improves ~57 % (from −71k → −30k)

### Extensions (implemented but computationally heavy)

- **Temporal (random‑walk):** Skills per season linked by $\theta_{t+1} \sim \mathcal{N}(\theta_t, \tau^2)$ — ~1,400 latent variables, ~20 min training
- **Weather‑dependent noise:** $\beta_{\text{perf}} \cdot (1 + \beta_{\text{wet}} \cdot \text{is\_wet})$ — allows wet races to be more random

## Results

### Model Comparison (10‑fold chronological CV, 2015–2024)

| Model | Pairwise Acc | Top‑1 Win | Spearman ρ | MSE ↓ |
|-------|:-----------:|:---------:|:----------:|:-----:|
| **Pyro** | **0.755** | **0.475** | **0.640** | **24.3** |
| Grid | 0.730 | 0.422 | 0.569 | 29.1 |
| TrueSkill | 0.659 | 0.389 | 0.455 | 36.8 |
| Random | 0.506 | 0.032 | 0.012 | 66.9 |

### Key Insights

1. **Pyro beats Grid** — the first model in this project to do so.
   The grid covariate ($\beta_g = -3.55$) strongly modulates predictions.
2. **Pyro beats TrueSkill** by +9.6 percentage points on pairwise accuracy
   (0.755 vs 0.659).  The pairwise ranking objective directly optimises the
   evaluation metric, whereas TrueSkill's EP approximates it.
3. The static Pyro model achieves this with only **100 latent parameters**
   vs TrueSkill's per‑race online EP — demonstrating that joint SVI on
   pairwise observations is a powerful learning strategy.
4. The **temporal variant** (random‑walk, 1,400+ parameters) is implemented
   but currently too expensive for practical use (~20 min training).
   Future: more efficient guide or GPU acceleration.

## Outputs

| File | Description |
|------|-------------|
| `outputs/pyro_model/driver_ratings_static.csv` | Posterior μ per driver (static model) |
| `outputs/pyro_model/constructor_ratings_static.csv` | Posterior μ per constructor |
| `outputs/pyro_model/coefficients_static.csv` | β_grid posterior mean |
| `outputs/pyro_model/pyro_comparison_metrics.csv` | All metrics × model × fold |
