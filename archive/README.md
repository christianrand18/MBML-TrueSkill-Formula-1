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

## Mathematical Foundation

The model is an instance of a **Bayesian Probabilistic Graphical Model (PGM)**
built on the [TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
ranking system (Herbrich et al., 2007).  It treats every Formula 1 race as a
**free‑for‑all match** among 20–24 competitors, where each entry is a
two‑element team combining a driver's skill and a constructor's (car's) skill.

### 1. Prior

Every driver and constructor enters the system with an identical Gaussian prior
over their latent skill:

$$\theta_{\text{new}} \;\sim\; \mathcal{N}\!\bigl(\mu_0 = 25,\; \sigma_0^2 = (25/3)^2 \approx 69.44\bigr)$$

- **μ₀ = 25** — the **prior mean**.  This is the system's best guess before
  observing any data.  A value of 25 places a new entrant near the middle of the
  skill scale (roughly 0–50 after convergence on 2011–2024 data).
- **σ₀² = (25/3)² ≈ 69.44** — the **prior variance** (σ₀ ≈ 8.33).  This is a
  wide prior: ±2σ covers [8.3, 41.7], expressing high initial uncertainty.
  Every new driver could be anywhere from a backmarker to a champion.

The prior is stored as a `trueskill.Rating(mu, sigma)` object per entity.

### 2. Forward Model (generative story)

For a given race with *N* competitors the model assumes:

#### 2a. Skill dynamics (drift between races)

Between two consecutive races, a player's skill is allowed to drift slightly
(analogous to form fluctuations, injury, car development):

$$\theta^{(t+1)} \;\big|\; \theta^{(t)} \;\sim\; \mathcal{N}\!\bigl(\theta^{(t)},\; \tau^2\bigr)$$

where **τ = 25/300 ≈ 0.083** controls how much skill changes race‑to‑race.
A small τ means skills evolve slowly; a larger τ would allow sharp form swings.

#### 2b. Team performance

Each F1 car is modelled as a **team** of two independent skill contributors:
the driver and the constructor.  The combined **performance** of car *j* is the
sum of the two skills plus Gaussian noise:

$$t_j = \underbrace{\theta_{\text{driver}(j)}}_{\text{driver skill}} \;+\; \underbrace{\theta_{\text{constructor}(j)}}_{\text{car skill}} \;+\; \varepsilon_j, \qquad \varepsilon_j \;\sim\; \mathcal{N}(0,\; \beta^2)$$

where **β = 25/6 ≈ 4.17** sets the performance noise scale.  A skill gap of
*β* between two identical cars translates to roughly an 80 % win probability
for the stronger team.

This additive structure means:
- A great driver in a poor car (high θ_driver + low θ_constructor) can still
  achieve mediocre performance — and vice versa.
- Both driver and constructor ratings are *jointly* inferred from the same race
  outcomes via the factor graph.

#### 2c. Ranking likelihood

The race result is the **ordering** of the latent performance variables
`t₁, …, t_N`:

$$\mathbb{P}\bigl(\text{ranks} \;\big|\; t_1, \dots, t_N\bigr) \;=\; \mathbf{1}\!\left[t_{\text{1st}} > t_{\text{2nd}} > \dots > t_{\text{last}}\right]$$

Teams with the same `positionOrder` are considered **tied** and share equal
rank values — the likelihood uses `≥` comparisons among tied entries.

There is **no draw probability** in our model (`draw_probability = 0.0`)
because F1 position sharing is extremely rare (official ties are decided by
countback).

### 3. Posterior Inference (Expectation Propagation)

The **posterior distribution** over all skills after observing the race ranking
is the exact Bayesian update:

$$\underbrace{p(\theta_1, \dots, \theta_M \mid \text{ranks})}_{\text{posterior}} \;\propto\; \underbrace{\prod_{i=1}^{M} p(\theta_i)}_{\text{drift‑corrected prior}} \;\times\; \underbrace{p(\text{ranks} \mid t_1, \dots, t_N)}_{\text{ranking likelihood}}$$

The ranking likelihood couples all `tⱼ` via a discontinuous indicator function
(a non‑Gaussian, combinatorial constraint).  The exact posterior is therefore
intractable for *N* ≥ 3.

TrueSkill approximates the posterior via **Expectation Propagation (EP)**
— a moment‑matching algorithm that operates on the factor graph:

```
  θ_driver_1 ──┬── t₁ ──┐
               │         │
  θ_cons_X  ──┘         │
                         ├── ranking factor ── observed ranks
  θ_driver_2 ──┬── t₂ ──┤
               │         │
  θ_cons_Y  ──┘         │
                         │
        ...              │
                         │
  θ_driver_N ──┬── t_N ──┘
               │
  θ_cons_Z  ──┘
```

**EP iterates** among the pairwise comparison factors between every ordered
pair of teams, replacing each non‑Gaussian message with the Gaussian that
moment‑matches it.  After convergence, each skill variable `θᵢ` has an
**approximate Gaussian posterior**:

$$\theta_i \;\sim\; \mathcal{N}\bigl(\mu_i^{\text{post}},\; (\sigma_i^{\text{post}})^2\bigr)$$

This `(μ_post, σ_post)` pair is what `trueskill.Rating` stores and what appears
in the output files.

### 4. Chronological Update (the pipeline)

The pipeline processes races in strict chronological order (`date` ascending):

```
Race 1:  prior (25, 8.33)  ──[EP]──►  posterior 1  ──[τ drift]──►  prior for Race 2
Race 2:  prior 2            ──[EP]──►  posterior 2  ──[τ drift]──►  prior for Race 3
  ...
Race 286: prior 286         ──[EP]──►  posterior 286  (final ratings)
```

At each step the uncertainty `σᵢ` **decreases** (we learn about the player) and
may increase slightly via τ‑drift (form changes).  An entity that last raced in
2012 and never reappeared will keep its final posterior frozen — the system does
not retroactively update historical ratings.

### 5. Worked Example: One Race

Consider a simplified 2‑car race: Hamilton (Mercedes) vs Verstappen (Red Bull).

| Team | Driver skill | Constructor skill | Performance |
|------|-------------|------------------|-------------|
| Car A | θ_Ham ~ N(28, 5²) | θ_Merc ~ N(30, 3²) | t_A ~ N(28+30, 5²+3²+4.17²) = N(58, 7.35²) |
| Car B | θ_Ver ~ N(27, 6²) | θ_RB ~ N(31, 4²) | t_B ~ N(27+31, 6²+4²+4.17²) = N(58, 8.33²) |

If Hamilton wins (observed: t_A > t_B), TrueSkill's EP updates move the
posteriors:

- θ_Ham: μ increases, σ decreases (driver performed well)
- θ_Merc: μ increases slightly, σ decreases (car contributed to win)
- θ_Ver: μ decreases, σ decreases (lost, but was close — small penalty)
- θ_RB: μ decreases slightly, σ decreases

The magnitude of the update scales with the **surprise** relative to prior
expectations and the **certainty** (1/σ²) of the players involved.

### 6. Interpreting mu and sigma

- **μ (mu)** — the expected skill.  Higher = stronger driver/constructor.
  A driver with μ = 30 is expected to outperform one with μ = 27 by a
  significant margin (~3β ≈ one std‑dev of performance noise).
- **σ (sigma)** — the posterior uncertainty.  Lower = more evidence.
  - σ ≈ 8.3 → no data yet (prior only).
  - σ ≈ 2–4 → a few races observed.
  - σ ≈ 0.7–0.8 → well‑established veteran with 100+ races.

A **95 % credible interval** for skill is approximately `μ ± 1.96σ`.

### References

- Herbrich, R., Minka, T., & Graepel, T. (2007).  *TrueSkill™: A Bayesian Skill
  Rating System*.  Advances in Neural Information Processing Systems 19.
- Minka, T. (2001).  *Expectation Propagation for approximate Bayesian
  inference*.  UAI 2001.

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
