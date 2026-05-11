# F1 Driver-Constructor Skill Separation — Bayesian PGM

Probabilistic graphical model for inferring latent F1 driver skill and constructor
performance from race finishing orders (2011–2024). Built with Pyro, PyTorch, and Pandas.

---

## Project Overview

This project decomposes F1 race results into two separate latent dimensions:

- **Driver skill** `s_d` — how much of a result is the driver (independent of the car)
- **Constructor performance** `c_k` — how much is the car (independent of the driver)

Because every result is the product of both, **skill separation** relies on two signals:
teammates getting different results in the same car, and drivers switching teams
across seasons.

Three models of increasing complexity share the same likelihood (Plackett-Luce
ranking) and the same identifiability constraint (sum-to-zero on constructors):

| Model | Latent Structure | Covariates |
|-------|-----------------|------------|
| **Baseline** | Static driver + constructor skills | None |
| **Extended** | AR(1) temporal skills + circuit effects | Global weather |
| **Full** | AR(1) temporal skills + circuit effects | Weather, pit stops, DNF reliability, wet-weather driver interactions |

---

## Project Structure

```
├── pyproject.toml                     # Dependencies (uv)
├── SPEC.md                            # Full model specification
├── CLAUDE.md                          # Project rulebook (read-only)
│
├── data_preprocessing/                # Shared data pipeline
│   ├── build_f1_model_data.py         # Build clean dataset from raw CSVs
│   ├── f1_model_ready.csv             # Clean dataset (5,980 rows)
│   └── f1_enriched.csv                # + weather/engineered features
│
├── models/pgm_backend/                # PGM implementation
│   ├── data_preparation.py            # Load + encode data as tensors
│   ├── likelihood.py                  # Plackett-Luce log-probability
│   ├── model_baseline.py              # Model 1 — static skills
│   ├── model_extended.py              # Model 2 — temporal + weather
│   ├── model_full.py                  # Model 3 — reliability + interactions
│   ├── inference.py                   # SVI training + NUTS (Model 1 only)
│   ├── posterior.py                   # Posterior extraction
│   ├── run_pgm.py                     # Orchestrator: train all, export CSVs + plots
│   └── tests/                         # Unit tests + synthetic recovery
│
├── outputs/pgm_model/                 # Generated results
│   ├── baseline_posterior.csv         # Model 1 posterior estimates
│   ├── extended_posterior.csv         # Model 2 posterior estimates
│   ├── full_posterior.csv             # Model 3 posterior estimates
│   ├── nuts_vs_svi_comparison.csv     # NUTS validation (Model 1)
│   └── plots/                         # 10 diagnostic + results plots
│
├── tasks/                             # Implementation planning
│   ├── todo.md                        # Master task checklist
│   ├── plan.md                        # Detailed implementation spec
│   └── report_notes.md                # Design decisions for report
│
├── F1_PGM_Evaluation.ipynb            # Results notebook
├── archive/                           # Old prototype (TrueSkill + pairwise probit)
└── data/                              # Raw Kaggle F1 dataset (not tracked)
```

---

## Quick Start

**Prerequisites:** Python >= 3.13, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd MBML-TrueSkill-Formula-1
uv sync
source .venv/bin/activate
```

### Run the pipeline

```bash
# 1. Build the clean dataset
python data_preprocessing/build_f1_model_data.py

# 2. Run the full PGM pipeline (all 3 models)
uv run python -m models.pgm_backend.run_pgm
# → outputs/pgm_model/ (CSVs + 10 plots)

# 3. Run tests
uv run python -m pytest models/pgm_backend/tests/ -v
```

Training takes ~5 minutes on CPU. All models use SVI (stochastic variational
inference). Model 1 also supports NUTS (Hamiltonian Monte Carlo) for validation.

### Explore results

Open `F1_PGM_Evaluation.ipynb` in Jupyter for posterior summaries, driver rankings,
and cross-model comparisons.

---

## Model Architecture

All three models use the **Plackett-Luce likelihood** — the exact probability
of an observed race ordering under latent performance scores:

$$P(\pi \mid p) = \prod_{i=1}^{N} \frac{\exp(p_{\pi(i)})}{\sum_{j \geq i} \exp(p_{\pi(j)})}$$

where the latent performance of driver $d$ in constructor $k$ is
$p_{d,k} = s_d + c_k$ (plus optional covariates).

**Identifiability** is enforced via a sum-to-zero reparameterisation on
constructor skills: $c_{\text{raw}}$ has shape $(K{-}1)$, and
$c = [c_{\text{raw}},\; -\Sigma\,c_{\text{raw}}]$ ensures $\sum_k c_k = 0$.

Mechanical DNFs (engine failures, collisions) are excluded from the ranking
likelihood in Models 1–2. Model 3 adds a Bernoulli reliability term.

### Model 1 — Baseline (Static)

- Static driver skills $s_d \sim \mathcal{N}(0, \sigma_s^2)$
- Static constructor performance $c_k$ with sum-to-zero
- Plackett-Luce likelihood, no covariates
- Inference: SVI + NUTS validation

### Model 2 — Extended (Temporal + Weather)

- AR(1) temporal skills: $s_{d,y}$ evolves across seasons
- Vectorised cumsum of per-year innovations (no recursive loops)
- Circuit random effects: $u_{\text{circuit}} \sim \mathcal{N}(0, \sigma_u^2)$
- Global weather coefficient: wet races shift all performance scores
- Inference: SVI only

### Model 3 — Full

- Everything in Model 2, plus:
- Wet-weather driver interactions $\delta_d$ — some drivers excel in rain
- Pit-stop normalised count $\beta_\pi$ — more stops = worse finishing position
- Bernoulli reliability term for driver-fault DNFs
- Inference: SVI only

---

## Key Findings

- **Driver vs constructor separation works.** Hamilton and Verstappen rank top
  in driver skill; Mercedes and Red Bull dominate constructor performance with
  distinct temporal trajectories peaking in their respective eras.
- **Wet-weather specialists emerge naturally.** The $\delta_d$ parameter
  identifies drivers who outperform their baseline skill in rain — 5 of the
  top 6 are Formula 1 world champions.
- **NUTS validates SVI.** R-hat < 1.05 for all parameters in Model 1,
  confirming SVI variational posteriors are well-calibrated.
- **Constructor sum-to-zero is critical.** Without it, driver and constructor
  skills trade off and become unidentifiable.

See `outputs/pgm_model/plots/` and `F1_PGM_Evaluation.ipynb` for all results.

---

## Old Prototype

The original pipeline (TrueSkill baseline + pairwise probit Pyro model + EDA +
evaluation + weather enrichment) was moved to `archive/`. See that directory
for the earlier work. The archive preserves the full git history of those files.

---

## Dependencies

```
numpy, pandas         — data handling
pyro-ppl, torch       — PGM inference (SVI + NUTS)
matplotlib, seaborn   — visualisation
```

Managed by [uv](https://docs.astral.sh/uv/) via `pyproject.toml`.

---

## License

This project uses the [Kaggle Formula 1 dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
(via Ergast API). Weather data is from [Open-Meteo](https://open-meteo.com/)
(free, no API key).
