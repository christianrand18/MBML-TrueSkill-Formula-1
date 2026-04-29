# F1 Skill Separation — PGM Implementation Spec

**Version:** 1.0  
**Status:** Approved  
**Deadline:** 2026-05-15

---

## 1. Objective

Build a family of three probabilistic graphical models that infer **latent driver skill**
and **latent constructor performance** from observed F1 race finishing orders (2011–2024).
The key scientific goal is *skill separation*: because a driver and car always appear
together, separation relies on teammates having different results in the same car, and
drivers switching teams across seasons.

All three models share the same likelihood (Plackett-Luce ranking) and the same
identifiability constraint (sum-to-zero on constructors). They differ in complexity
of the latent structure and covariates.

---

## 2. Three Model Tiers

### 2.1 Model 1 — Baseline (Static)

**Latent variables:**
- `s_d` — driver skill, one scalar per driver, time-invariant. Shape `(D,)`.
- `c_k` — constructor performance, one scalar per constructor, time-invariant. Shape `(K,)`.

**Generative process:**

```
# Priors
s_d ~ Normal(0, sigma_s)   for each driver d        # (D,)
c_k ~ Normal(0, sigma_c)   for each constructor k   # (K,)

# Constraint: sum(c_k) = 0  [applied as reparameterisation — see §5]

# Per race r, per driver d in race r:
p_{d,r} = s_d + c_{k(d,r)} + eps,   eps ~ Normal(0, beta^2)

# Likelihood: Plackett-Luce over observed finishing order
P(observed order | p) = ∏_i  exp(p_i) / sum_{j >= i} exp(p_j)
```

**Hyperparameters (fixed constants, not learned):**
- `sigma_s = 1.0`
- `sigma_c = 1.0`
- `beta = 0.5` (performance noise)

**Inference:** Both SVI (mean-field guide) and MCMC (NUTS) must be implemented and
compared for this model. If SVI and NUTS disagree substantially on posterior means
(> 0.3 in standardised units), the guide or model has a bug.

---

### 2.2 Model 2 — Extended (Temporal + Circuit + Weather)

Extends Model 1 with temporal dynamics, circuit effects, and a global weather term.

**Additional latent variables:**

| Variable | Shape | Description |
|---|---|---|
| `s_{d,t}` | `(T, D)` | Driver skill per season (AR(1) random walk) |
| `c_{k,t}` | `(T, K)` | Constructor performance per season (AR(1)) |
| `e_circ` | `(C,)` | Circuit-specific latent effect |
| `beta_w` | scalar | Global wet-weather performance coefficient |

**Generative process:**

```
# Temporal dynamics (AR(1))
s_{d,0} ~ Normal(0, sigma_s)
s_{d,t} ~ Normal(s_{d,t-1}, gamma_s)   for t = 1..T-1

c_{k,0} ~ Normal(0, sigma_c)
c_{k,t} ~ Normal(c_{k,t-1}, gamma_c)   for t = 1..T-1

# Constraint: sum(c_{k,t}) = 0 for each t  [see §5]

# Circuit effects
e_c ~ Normal(0, sigma_e)               for each circuit c   # (C,)

# Weather coefficient (latent)
beta_w ~ Normal(0, 0.5)

# Per race r (in season t(r), on circuit circ(r)):
p_{d,r} = s_{d, t(r)}
         + c_{k(d,r), t(r)}
         + e_{circ(r)}
         + beta_w * w_r          # w_r is OBSERVED (0=dry, 1=wet)
         + eps,  eps ~ Normal(0, beta^2)
```

**New hyperparameters (fixed):**
- `gamma_s = 0.3` (AR(1) innovation std for drivers)
- `gamma_c = 0.5` (AR(1) innovation std for constructors; larger because regulation changes)
- `sigma_e = 0.5` (circuit effect prior std)

**Covariates (observed, no prior):**
- `w_r` — binary wet indicator from `is_wet` column in `f1_enriched.csv`

**Inference:** SVI only (NUTS is computationally prohibitive for `(D*T + K*T)` latents).

---

### 2.3 Model 3 — Full (Extended + Wet-Weather Skill + Pit Stops)

Extends Model 2 with driver-specific wet-weather skill (an interaction term) and pit-stop
execution as a covariate.

**Additional latent variables:**

| Variable | Shape | Description |
|---|---|---|
| `delta_d` | `(D,)` | Driver wet-weather skill modifier |
| `beta_pi` | scalar | Pit-stop time coefficient |

**Generative process (adds to Model 2's process):**

```
# Wet-weather driver skill (latent)
delta_d ~ Normal(0, sigma_delta)    for each driver d   # (D,)

# Pit-stop coefficient (latent)
beta_pi ~ Normal(0, 0.5)

# Per race r:
p_{d,r} = s_{d, t(r)}
         + c_{k(d,r), t(r)}
         + e_{circ(r)}
         + beta_w  * w_r
         + delta_d * w_r           # INTERACTION — not additive, must be a product
         + beta_pi * pi_{d,r}      # pi_{d,r} is OBSERVED pit time (normalised)
         + eps,  eps ~ Normal(0, beta^2)
```

**Critical:** `delta_d * w_r` is a multiplicative interaction. Do NOT add them separately.

**New hyperparameters (fixed):**
- `sigma_delta = 0.5` (wet-weather skill prior std)

**Covariates (observed, no prior):**
- `pi_{d,r}` — normalised mean pit-stop duration from `total_pit_duration_ms` column.
  Normalise per season (subtract season mean, divide by season std) before entering model.
- `w_r` — same as Model 2.

**Inference:** SVI only.

---

## 3. Shared Likelihood: Plackett-Luce

All three models use the same ranking likelihood. For a race with `N` drivers, where
`π` is the observed finishing order (π(1) = winner, π(N) = last):

```
log P(π | p) = sum_{i=1}^{N}  [ p_{π(i)} - log( sum_{j=i}^{N} exp(p_{π(j)}) ) ]
```

**Implementation note:** Use `torch.logsumexp` for numerical stability. Do NOT
implement this as a product of softmaxes applied to raw positions.

**DNF rows are included in the ranking (not filtered out)** — see §4 for ordering rules.

---

## 4. Data Preprocessing

### 4.1 Input Files

- `data_preprocessing/f1_model_ready.csv` — 5,980 rows × 13 columns (base dataset).
- `data_preprocessing/f1_enriched.csv` — 5,980 rows × 26 columns (adds weather).

Key columns: `raceId, year, driverId, constructorId, positionOrder, statusId,
total_pit_duration_ms, is_wet`.

### 4.2 DNF Classification

`statusId` is available for all rows. Classify each row as one of three categories:

| Category | statusId values | Treatment |
|---|---|---|
| **Finished** | 1, 11–20 (lapped cars still completed race) | Include in Plackett-Luce ranking |
| **Mechanical DNF** | 5 (Engine), 6 (Gearbox), 7 (Transmission), 8 (Clutch), 9 (Hydraulics), 10 (Electrical), 18 (Fuel system), 19 (Front wing), 21 (Fuel pressure), 22 (Exhaust), 26 (Oil leak), 28 (Fuel pump), 29 (Overheating), 31 (Brakes), 36 (Wheel), 40 (Suspension), 41 (Driveshaft), 43 (Differential), 44 (Puncture), 54 (Power unit), 61 (Oil pipe), 65 (Oil pressure), 66 (Wheel nut), 67 (Water pressure), 72 (Oil), 75 (Hydraulics), 82 (Throttle), 104 (Injection), 107 (Alternator), 108 (Brake duct), 130 (Power loss), 131 (ERS) | **Excluded from ranking** (Models 1 & 2). Bernoulli reliability term (Model 3 only) — see §4.3 |
| **Driver-fault DNF** | 3 (Accident), 4 (Collision), 20 (Spun off), 23 (Retired driver decision), 25 (Handling) and all other unlisted statusIds | Append after Finished entries, ranked by `positionOrder`; both driver and constructor penalised |

**Race ordering (Models 1 & 2):** `[Finished by positionOrder] → [Driver-fault DNFs by positionOrder]`  
Mechanical DNFs are dropped from the race entirely for the ranking step.

### 4.3 Mechanical DNF Handling

**Why not include mechanical DNFs in the ranking:**  
Including a mechanical DNF as "last place" via `p = c_k` creates an asymmetric bias:
a high-quality constructor (large `c_k`) suffers a larger Plackett-Luce penalty for its
reliability failure than a low-quality constructor, because the gradient penalises the
gap between observed position (last) and expected position (high, because `c_k` is large).
This is directionally wrong — reliability failures should not hurt fast teams more than slow ones.

**Models 1 & 2 — exclude from ranking:**  
Drop all mechanical DNF rows before building the race-level Plackett-Luce likelihood.
The constructor receives no signal from mechanical DNFs in these models.

**Model 3 — add Bernoulli reliability term:**  
Mechanical DNFs *do* carry constructor signal: better constructors fail mechanically less
often. Model 3 adds a separate observation equation for this signal:

```python
# is_mech: BoolTensor (N_entries,) — 1 if mechanical DNF, 0 otherwise (OBSERVED)
alpha_rel = pyro.sample("alpha_rel", dist.Normal(0.0, 1.0))  # baseline DNF rate intercept
mech_prob = torch.sigmoid(-alpha_rel - c[cons_idx])           # higher c_k → lower failure prob
pyro.factor("reliability", dist.Bernoulli(mech_prob).log_prob(is_mech.float()))
```

`alpha_rel` absorbs the baseline mechanical DNF rate (≈ 17% in this dataset; at convergence
`alpha_rel` ≈ 1.6). `c_k` then adjusts each constructor's reliability relative to that
baseline. This correctly penalises *low-quality* constructors more for mechanical failures
and gives *high-quality* constructors a positive reliability signal.

**Note for report:** In Models 1 & 2, `c_k` captures constructor *pace* only. In Model 3,
`c_k` captures a combination of pace and reliability. This is documented as a deliberate
modelling choice: overall constructor quality encompasses both dimensions.

### 4.4 Constructor Rebranding

Verified ID mapping from year-by-year constructor activity in `f1_model_ready.csv`:

| Canonical ID | Merge these IDs | Real-world team | Years active |
|---|---|---|---|
| `10` (Force India) | `211` (Racing Point), `117` (Aston Martin) | Silverstone factory | 2011–2024 |
| `4` (Renault) | `214` (Alpine) | Enstone factory | 2011, 2016–2024 |
| `5` (Toro Rosso) | `213` (AlphaTauri), `215` (Racing Bulls) | Faenza factory | 2011–2024 |
| `15` (Sauber/Kick Sauber) | `51` (Alfa Romeo) | Hinwil factory | 2011–2024 |

The small backmarker teams (164 Marussia/Virgin, 166 Caterham, 205 HRT, 206/207/208
their successors, 209 Manor) all exited F1 by 2016. Treat each as its own constructor
(no merge needed since they have no continuity).

**Implementation:** Build a `CONSTRUCTOR_REMAP` dict in the preprocessing module and
apply it before creating integer indices:

```python
CONSTRUCTOR_REMAP = {
    211: 10,   # Racing Point → Force India
    117: 10,   # Aston Martin → Force India
    214: 4,    # Alpine → Renault
    213: 5,    # AlphaTauri → Toro Rosso
    215: 5,    # Racing Bulls → Toro Rosso
    51:  15,   # Alfa Romeo → Sauber
}
```

### 4.5 Index Tensors

Mechanical DNF filtering happens **inside the preprocessor**, not inside the model.
`N_entries` refers only to finishers and driver-fault DNFs. Models 1 & 2 never see
a mechanical DNF row.

The preprocessing module must produce:

```python
# Models 1, 2, and 3 — ranking entries only (NO mechanical DNFs)
driver_idx    # LongTensor (N_entries,)  → index into s_d
cons_idx      # LongTensor (N_entries,)  → index into c_k
season_idx    # LongTensor (N_entries,)  → 0-based season index
circuit_idx   # LongTensor (N_entries,)  → index into e_circ
race_idx      # LongTensor (N_entries,)  → which race (for grouping entries into races)
pit_norm      # FloatTensor (N_entries,) → normalised pit-stop duration
wet           # FloatTensor (N_races,)   → binary wet indicator per race
race_order    # List[LongTensor]         → for each race r, indices of entries in finishing order
race_lengths  # LongTensor (N_races,)   → number of ranking entries per race

# Model 3 only — over ALL original rows (N_all > N_entries)
is_mech       # BoolTensor (N_all,)     → True if mechanical DNF (feeds Bernoulli reliability term)
cons_idx_all  # LongTensor (N_all,)     → constructor index for all original rows
```

`is_mech` and the ranking entries are mutually exclusive — no row appears in both.
`mech_mask` does not exist; there is no mask to apply inside any model.

**Run a shape assertion after building each tensor.** Silent shape bugs are the most
common failure mode.

### 4.6 Season Indexing

Map `year` to a zero-based integer index:
```python
seasons = sorted(df['year'].unique())       # [2011, 2012, ..., 2024]
season_map = {y: i for i, y in enumerate(seasons)}  # {2011: 0, ..., 2024: 13}
```

### 4.7 Pit-Stop Normalisation

Normalise `total_pit_duration_ms` per season:
```python
df['pit_norm'] = df.groupby('year')['total_pit_duration_ms'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)
```

Entries with no pit-stop data (if any) should be filled with 0.0 (the seasonal mean).

---

## 5. Identifiability Constraint

**Sum-to-zero on constructors.** Applied as a reparameterisation, not a post-hoc
normalisation:

```python
# In the model — baseline (static) example:
c_raw = pyro.sample("c_raw", dist.Normal(0, sigma_c).expand([K-1]).to_event(1))
c = torch.cat([c_raw, -c_raw.sum(keepdim=True)])   # last constructor constrained
```

For the temporal variant:
```python
# At each season t:
c_raw_t = ...                    # shape (K-1,)
c_t = torch.cat([c_raw_t, -c_raw_t.sum(keepdim=True)])   # shape (K,)
```

This must be present in all three models. The guide samples `c_raw` (or `c_raw_t`),
not `c` directly.

---

## 6. Inference

### 6.1 SVI (all three models)

- Optimizer: `ClippedAdam`, `lr=0.01`, `clip_norm=10.0`
- Guide: mean-field (independent Normals for all latent scalars and vectors)
- Loss: `Trace_ELBO(num_particles=1)`
- Steps: 3000 minimum; 5000 for Models 2 and 3
- Subsample races (not entries) if memory is tight — but maintain valid Plackett-Luce
  groupings (never split a race across batches)

### 6.2 MCMC / NUTS (Model 1 only)

- Sampler: `pyro.infer.MCMC` with `pyro.infer.NUTS` kernel
- Chains: 1 (2 if time allows)
- Warmup: 500, samples: 500
- Check R-hat < 1.05 for all latents; log a warning if violated
- Compare NUTS posterior means to SVI posterior means; report discrepancy in outputs

### 6.3 No Discrete Latents

None of the three models use discrete latent variables. DNF classification is done in
preprocessing (observed), not as a latent mixture. This means `Trace_ELBO` is sufficient
and `TraceEnum_ELBO` is not needed.

---

## 7. Verification: Artificial Data Recovery

Before running on real data, each model must pass a synthetic recovery test:

1. Fix ground-truth values, e.g.:
   ```python
   true_s = {hamilton: 2.5, verstappen: 2.3, bottas: 1.5, ...}
   true_c = {mercedes: 3.0, red_bull: 2.8, williams: -1.0, ...}
   ```
2. Use ancestral sampling to generate synthetic finishing orders for 50 races
3. Run inference on the synthetic data
4. Assert: posterior mean for each driver/constructor is within ±0.5 of ground truth

This test lives in `models/pgm_backend/tests/test_synthetic_recovery.py`.

---

## 8. Project Structure

```
models/
└── pgm_backend/
    ├── __init__.py
    ├── data_preparation.py    # All preprocessing: index tensors, DNF classification,
    │                          # constructor remapping, normalisation
    ├── likelihood.py          # PlackettLuceLikelihood — standalone, testable
    ├── model_baseline.py      # Model 1: static skills, SVI + NUTS
    ├── model_extended.py      # Model 2: temporal AR(1), circuit, weather
    ├── model_full.py          # Model 3: Model 2 + wet-weather skill + pit stops
    ├── inference.py           # train_svi(), run_nuts() shared helpers
    ├── posterior.py           # extract_posterior(), posterior_summary() for all models
    ├── run_pgm.py             # Orchestrator: train all 3, export results, compare
    └── tests/
        ├── test_likelihood.py          # Unit test PlackettLuce log-prob
        └── test_synthetic_recovery.py  # Recovery test for all 3 models
```

**Output directory:** `outputs/pgm_model/`
- `baseline_posterior.csv` — driver and constructor posterior means + stds
- `extended_posterior.csv`
- `full_posterior.csv`
- `nuts_vs_svi_comparison.csv` — Model 1 only
- `elbo_curves.png` — training curves for all 3 SVI runs

---

## 9. Acceptance Criteria

### 9.1 Correctness
- [ ] Plackett-Luce log-prob is correct on a hand-verifiable 3-driver example
- [ ] Sum-to-zero constraint holds: `c_k.sum() ≈ 0` after inference (tolerance 1e-4)
- [ ] ELBO is monotonically non-increasing over training (allow transient fluctuations < 5%)
- [ ] Synthetic recovery test passes for all three models

### 9.2 Posterior Sanity
- [ ] Top-5 drivers by posterior mean `s_d` include Hamilton and Verstappen
- [ ] Constructor ranking broadly matches: Mercedes dominant 2014–2021 era, Red Bull 2022–2024
- [ ] Posterior std for drivers with ≥ 50 races is smaller than for drivers with < 10 races
- [ ] Model 3 assigns higher `delta_d` to known wet-weather specialists (e.g., Alonso, Senna-era if applicable)

### 9.3 Inference
- [ ] SVI trains in < 5 minutes on CPU for Model 1
- [ ] NUTS completes in < 30 minutes on CPU for Model 1 (500 warmup + 500 samples)
- [ ] R-hat < 1.05 for all Model 1 NUTS latents
- [ ] SVI vs NUTS posterior mean discrepancy < 0.3 (standardised) for Model 1 drivers

### 9.4 Code Quality
- [ ] Every index tensor has an assertion checking its shape and dtype
- [ ] No `pyro.sample` with a non-delta distribution on observed variables (`w_r`, `pi_{d,r}`)
- [ ] `mech_mask` correctly zeros driver contribution for mechanical DNF entries
- [ ] Constructor rebranding mapping is documented with a comment explaining the real-world team

---

## 10. Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Ranking likelihood | Plackett-Luce | Proper ranking likelihood; pairwise probit throws away ordering information |
| Grid position | Excluded | Blocks information path from skill to result |
| DNF mechanical (Models 1 & 2) | Excluded from ranking entirely | Masking approach (`p = c_k`) asymmetrically penalises good constructors — a Mercedes DNF would hurt Mercedes more than a Haas DNF hurts Haas, which is wrong |
| DNF mechanical (Model 3) | Bernoulli reliability term with intercept `sigmoid(-alpha_rel - c_k)` | Captures reliability signal correctly: better constructors fail less often; `alpha_rel` absorbs baseline rate |
| Constructor rebranding | Merge to single ID | AR(1) continuity requires same entity across seasons |
| Identifiability | Sum-to-zero on constructors | Symmetric, no arbitrary reference team |
| Inference | SVI + NUTS on Model 1, SVI only on 2 & 3 | NUTS cost scales poorly with latent dimension |
| Grid position as covariate | Not included | Scientific goal is pure skill separation from race outcome |
