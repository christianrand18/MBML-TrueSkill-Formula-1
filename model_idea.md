# F1 Skill Separation — PGM Model Specification

This document is the **ground-truth specification** for the F1 probabilistic graphical
model. Use it to audit whether an existing implementation matches the intended design.

---

## 1. Problem Statement

Given observed race finishing orders, infer:

- **Driver skill** `s_d` — latent, per driver, independent of car
- **Constructor performance** `c_k` — latent, per constructor, independent of driver

The key identification challenge: a driver and their car always appear together, so
skill separation relies on (a) teammates having different results in the same car,
and (b) drivers switching teams across seasons.

---

## 2. Baseline Model

### 2.1 Generative Process

```
For each driver d = 1..D:
    s_d ~ Normal(mu_s, sigma_s^2)              # driver skill

For each constructor k = 1..K:
    c_k ~ Normal(mu_c, sigma_c^2)              # car performance

For each race r = 1..R:
    For each driver d in D_r (drivers in race r):
        p_{d,r} = s_d + c_{k(d,r)} + eps      # combined performance
        eps ~ Normal(0, beta^2)
        y_{d,r} ~ RankLikelihood({p_{d,r}})   # observed finishing order
```

### 2.2 Variables and Shapes

| Variable | Type | Shape | Description |
|---|---|---|---|
| `mu_s` | hyperparameter (const) | scalar | prior mean on driver skill |
| `sigma_s` | hyperparameter (const) | scalar | prior std on driver skill |
| `mu_c` | hyperparameter (const) | scalar | prior mean on constructor performance |
| `sigma_c` | hyperparameter (const) | scalar | prior std on constructor performance |
| `beta` | hyperparameter (const) | scalar | performance noise std |
| `s_d` | **latent** | `(D,)` | driver skills |
| `c_k` | **latent** | `(K,)` | constructor performances |
| `p_{d,r}` | **latent** (deterministic given s,c,eps) | `(D_r,)` per race | race performances |
| `y_{d,r}` | **observed** | `(D_r,)` per race | finishing positions (1 = winner) |

### 2.3 Prior Choices to Check

- Priors on `s_d` and `c_k` should be **centred** (e.g. `mu_s = mu_c = 0`) or have an
  **identifiability constraint** — without one, you can add a constant to all `s_d` and
  subtract it from all `c_k` and get equal likelihood. Common fix: pin one constructor
  (e.g. `c_0 = 0`) or use a sum-to-zero constraint.
- `sigma_s`, `sigma_c`, `beta` should be **positive** — check they use `HalfNormal`,
  `Exponential`, or similar, not unconstrained `Normal`.

### 2.4 Likelihood

The finishing order is a **ranking likelihood over performances**.  Two valid approaches:

**Option A — Plackett-Luce / TrueSkill-style:**  
The probability of the observed ordering equals the probability that performances sort
into that order. In Pyro this can be implemented as a sequence of categorical draws
(Plackett-Luce) or approximated via the TrueSkill factor.

**Option B — Gaussian approximation:**  
Model the observed position directly as `y_{d,r} ~ Normal(rank(p_{d,r}), noise)`.
This is an approximation but tractable. Check whether this is what is implemented and
flag it clearly as an approximation.

**What to flag as wrong:**  
- Treating position as a simple regression target without accounting for the
  ordering constraint (e.g. predicting position 1.3 independently for two drivers).
- Including `grid_position` as an observed node that sits *between* skill and result
  in the graph — this blocks the information path (see Section 5).

---

## 3. Extended Model

### 3.1 Additional Variables

On top of the baseline, the extended model adds:

| Variable | Type | Shape | Description |
|---|---|---|---|
| `gamma_s` | hyperparameter (const) | scalar | AR(1) innovation std for driver skill |
| `gamma_c` | hyperparameter (const) | scalar | AR(1) innovation std for constructor perf |
| `mu_e`, `sigma_e` | hyperparameter (const) | scalar | circuit effect prior |
| `mu_delta`, `sigma_delta` | hyperparameter (const) | scalar | wet-weather skill prior |
| `s_{d,t}` | **latent** | `(D, T)` | driver skill per season |
| `c_{k,t}` | **latent** | `(K, T)` | constructor performance per season |
| `e_c` | **latent** | `(C,)` | circuit-specific latent effect |
| `delta_d` | **latent** | `(D,)` | driver wet-weather skill |
| `beta_w` | **latent or const** | scalar | global weather coefficient |
| `beta_pi` | **latent or const** | scalar | pit-stop time coefficient |
| `w_r` | **observed** | `(R,)` | weather indicator (0=dry, 1=wet) |
| `pi_{d,r}` | **observed** | varies | mean pit-stop duration per driver per race |

### 3.2 Temporal Dynamics (AR(1))

```
# Initialisation (first season t=0):
s_{d,0} ~ Normal(mu_s, sigma_s^2)
c_{k,0} ~ Normal(mu_c, sigma_c^2)

# Subsequent seasons t = 1..T-1:
s_{d,t} ~ Normal(s_{d,t-1}, gamma_s^2)
c_{k,t} ~ Normal(c_{k,t-1}, gamma_c^2)
```

**What to check:**
- Are seasons indexed correctly? Season year should map to a zero-based index `t`.
- Is `gamma_s` given a sensible prior? Too large → skills jump wildly each season.
  Too small → model cannot track genuine team upgrades (e.g. Red Bull 2022).
- The dynamics must be inside the driver/constructor plates — not shared globally.

### 3.3 Extended Performance Equation

```
p_{d,r} = s_{d,t(r)}          # driver skill in season of race r
         + c_{k(d,r), t(r)}   # constructor performance in season of race r
         + e_{circ(r)}        # circuit effect
         + beta_w * w_r       # global weather effect
         + delta_d * w_r      # driver-specific wet-weather skill (interaction!)
         + beta_pi * pi_{d,r} # pit-stop execution
         + eps,   eps ~ Normal(0, beta^2)
```

**The interaction term `delta_d * w_r` is critical** — it is *not* just two additive
terms. Check that the code multiplies `delta_d` by `w_r` rather than adding them
independently.

### 3.4 Weather Data Integration

- `w_r` is **observed** — no prior should be placed on it.
- If using a continuous weather variable (e.g. rainfall in mm), check that it is
  normalised / standardised before entering the model.
- `beta_w` may be latent (with a prior) or a fixed constant — either is valid, but
  document the choice.

---

## 4. Key Implementation Checks (Pyro-specific)

### 4.1 Plate Structure

```python
# Expected Pyro plate structure — baseline
with pyro.plate("drivers", D):
    s = pyro.sample("s", dist.Normal(mu_s, sigma_s))          # (D,)

with pyro.plate("constructors", K):
    c = pyro.sample("c", dist.Normal(mu_c, sigma_c))          # (K,)

for r in range(R):                                             # or vectorised
    with pyro.plate(f"drivers_in_race_{r}", len(D_r)):
        p = s[driver_idx[r]] + c[constructor_idx[r]] + ...
        # likelihood over finishing order
```

Check that plate names are **unique per race** if using a loop, or that the
vectorised version correctly indexes into `s` and `c` using driver/constructor index
tensors.

### 4.2 Index Tensors

The model requires index arrays mapping each (driver, race) observation to the
correct latent variable. Verify:

```python
driver_idx[d, r]       # index into s — shape (D_r,) per race or (N_obs,) globally
constructor_idx[d, r]  # index into c — shape matches driver_idx
circuit_idx[r]         # index into e_c — shape (R,)
season_idx[r]          # index into time axis t — shape (R,)
```

Missing or incorrect index tensors are the most common source of silent bugs.
Run a **shape check** on every `pyro.sample` and every tensor operation.

### 4.3 Identifiability Constraint

Without a constraint, the model is **unidentified**: adding scalar `alpha` to all
`s_d` and subtracting `alpha` from all `c_k` leaves the likelihood unchanged.

Check that **at least one** of the following is present:
- [ ] One constructor's performance is pinned: `c[0] = 0` (delta distribution)
- [ ] A sum-to-zero constraint: `c = c_raw - c_raw.mean()`
- [ ] Priors are informative enough that the posterior is practically identified
      (weakest option — flag this if it is the only mechanism)

### 4.4 Discrete Latent Variables

If DNF (Did Not Finish) is modelled as a latent outcome (e.g. a mixture):
- Discrete latents require `TraceEnum_ELBO` in Pyro, not standard `Trace_ELBO`.
- Check the guide uses `pyro.sample` with `infer={"enumerate": "parallel"}`.
- Flag if this is missing — it will silently produce wrong gradients.

### 4.5 DNF Handling

Check how retirements are handled. Look for one of:
- [ ] DNF races are **filtered out** entirely — simplest, but loses data
- [ ] DNF is treated as finishing last — **incorrect**, biases constructor estimates
      (mechanical DNFs are car quality signal, not driver quality signal)
- [ ] `status` codes are used to distinguish driver-fault vs mechanical DNFs —
      **best practice**

### 4.6 Pit Stop Covariate

`pi_{d,r}` is **observed** — check no prior is placed on it. It should enter the
model only as a covariate (multiplied by `beta_pi`). If it has a `pyro.sample`
statement with a non-delta distribution it is being treated as latent, which is wrong.

---

## 5. Common Mistakes to Flag

### 5.1 Blocking Variable (Critical)

**Grid position** (`grid` column in `results.csv`) sits on the causal path:

```
Driver Skill  ──►  Qualifying  ──►  Grid Position  ──►  Race Result
Constructor   ──►                                   ──►
```

If grid position is **always observed** and placed as an intermediate node in the
graph, it **blocks** the path from skill to result. The driver-skill and race-result
sub-models become conditionally independent — there is no point fitting them jointly.

Ask your groupmate: *Is grid position in the model? If so, what role does it play?*

Acceptable uses of grid position:
- As an **additive covariate** in the performance equation (equivalent to: "given
  skill, starting position also matters")
- As a **separate observation equation** (model qualifying and race jointly with
  shared latent skill)

Not acceptable:
- As an intermediate observed node that skill flows *through* to reach race result.

### 5.2 Priors on Observed Variables

Check that no `pyro.sample` with a non-trivial distribution is placed on:
- `w_r` (weather — fully observed)
- `pi_{d,r}` (pit times — fully observed)
- `y_{d,r}` (finishing positions — this IS the observed data, it should be the
  likelihood, not a prior)

### 5.3 Non-Stationarity Without Dynamics

If the model uses a single `s_d` per driver across *all seasons* (no temporal
dynamics), it will average out genuine performance changes. This is acceptable for
the **baseline** but should be flagged as a limitation, especially for:
- Constructors with major regulation changes (2014 hybrid era, 2022 ground effect)
- Drivers with clear career trajectories

### 5.4 Posterior Predictive Check

Ask: *Has the model been validated with ancestral sampling?*

The workflow should be:
1. Fix known skill values (e.g. `s_Hamilton = 2.0`, `s_Bottas = 1.5`, `c_Mercedes = 3.0`)
2. Sample synthetic race outcomes from the model
3. Run inference on those synthetic outcomes
4. Verify posterior means recover the ground truth values

If this check has not been done, the inference may be broken without anyone noticing.

---

## 6. Dataset Preprocessing Checks

| Check | Expected |
|---|---|
| Driver IDs are consistent across seasons | Use `driverId` from `drivers.csv`, not name strings |
| Constructor IDs handle rebrandings | Renault/Alpine, Force India/Racing Point/Aston Martin should be checked — are they one constructor or separate? |
| DNF rows handled | `results.csv` rows where `positionOrder` is missing or `statusId` indicates retirement |
| Season indexing | Years mapped to 0-based integer index for temporal dynamics |
| Pit stop aggregation | Multiple stops per race should be aggregated (e.g. mean or total duration) before entering as `pi_{d,r}` |
| Weather data join | External weather joined on `raceId` + date — verify no off-by-one on race dates |
| Minimum race threshold | Drivers with very few races have poorly identified skills — consider filtering or using stronger priors |

---

## 7. Sanity Checks on Posterior

Once inference runs, verify the following **before** trusting results:

- [ ] Top-ranked drivers by posterior mean `s_d` broadly match expert consensus
      (Hamilton, Verstappen, Alonso should be near the top)
- [ ] Constructor ranking `c_k` by posterior mean broadly matches historical
      competitiveness (Mercedes 2014–2021, Red Bull 2022–2024)
- [ ] Posterior standard deviations `sigma_{s_d}` are **smaller** for drivers with
      many races and **larger** for rookies (uncertainty should decrease with data)
- [ ] The model assigns higher wet-weather skill `delta_d` to known rain specialists
      if the extended model is used
- [ ] ELBO loss decreases monotonically during VI training (or R-hat < 1.01 for MCMC)

---

## 8. Files to Request from Groupmate

To do a full review, ask for:

```
model.py          # Pyro model and guide definitions
data_loader.py    # Preprocessing — index tensors, DNF handling
train.py          # Inference loop, ELBO, hyperparameters used
notebooks/        # Any EDA or posterior analysis
```