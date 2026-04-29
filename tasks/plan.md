# F1 PGM Implementation Plan

**Based on:** SPEC.md v1.0  
**Target directory:** `models/pgm_backend/`  
**Stack:** Python 3.13, Pyro 1.9+, PyTorch 2.11+, Pandas 3+

---

## Dependency Graph

```
data_preparation.py
        │
        ├──────────────┐
        ▼              ▼
likelihood.py     inference.py
        │              │
        └──────┬────────┘
               ▼
      model_baseline.py  ←── Task 3 (SVI) + Task 4 (NUTS)
               │
               ▼
      model_extended.py  ←── Task 6
               │
               ▼
       model_full.py     ←── Task 7
               │
        ┌──────┴──────┐
        ▼             ▼
  posterior.py    run_pgm.py  ←── Task 8
```

**Tests** (Task 5) depend on: `likelihood.py` + `model_baseline.py`

---

## Task Breakdown

### Task 1 — Data Preparation (`data_preparation.py`)

**What:** Load `f1_enriched.csv`, apply constructor rebranding, classify DNFs, and
emit all index tensors needed by every model.

**Inputs:** `data_preprocessing/f1_enriched.csv`

**Outputs:** `F1RankingDataset` dataclass with:

Ranking entries (N_entries = finishers + driver-fault DNFs only, mechanical DNFs excluded):
- `driver_idx`, `cons_idx`, `season_idx`, `circuit_idx` — LongTensors `(N_entries,)`
- `race_idx` — LongTensor `(N_entries,)` mapping each entry to its race
- `pit_norm` — FloatTensor `(N_entries,)` normalised pit-stop time
- `wet` — FloatTensor `(N_races,)` binary wet indicator
- `race_order` — LongTensor `(N_entries,)` within-race finishing position (0=winner)
- `race_lengths` — LongTensor `(N_races,)` entries per race
- Integer counts: `n_drivers`, `n_constructors`, `n_seasons`, `n_circuits`, `n_races`
- Reverse maps: `driver_map`, `constructor_map` (int_id → int_idx and back)

Model 3 reliability fields (N_all = all original rows including mechanical DNFs):
- `is_mech` — BoolTensor `(N_all,)` True if mechanical DNF
- `cons_idx_all` — LongTensor `(N_all,)` constructor index for all rows

**There is no `mech_mask` field.** Mechanical DNF filtering is done in the preprocessor,
not inside any model. No model ever applies a mask to zero out entries.

**Key implementation notes:**
1. Apply `CONSTRUCTOR_REMAP` dict BEFORE building integer indices
2. Classify DNF category using `statusId` (see SPEC §4.2):
   - `MECHANICAL_STATUS_IDS` = frozenset({5,6,7,8,9,10,18,19,21,22,26,28,29,31,36,
     40,41,43,44,54,61,65,66,67,72,75,82,104,107,108,130,131})
   - Everything else (statusId not in {1,11..20} and not mechanical) = driver-fault
3. Filter out mechanical DNFs BEFORE building index tensors. N_entries only contains
   finishers and driver-fault DNFs. Build `race_order` by sorting each remaining race:
   - Group: Finished (statusId 1, 11-20) ranked by positionOrder
   - Then: driver-fault DNFs ranked by positionOrder
4. Separately, over ALL original rows (before filtering), emit:
   - `is_mech`: BoolTensor `(N_all,)` True if mechanical DNF
   - `cons_idx_all`: LongTensor `(N_all,)` constructor index for all rows
   These two fields are used only by Model 3's Bernoulli reliability term.
5. Normalise pit times per season (`(x - mean) / (std + 1e-8)`)
6. Assert shape and dtype of EVERY tensor before returning

**Acceptance criteria:**
- `dataset.n_races == 286`
- `dataset.n_drivers` in [17, 25] per season (spot check)
- `dataset.is_mech.sum() / len(dataset.is_mech)` ≈ 0.17 (mechanical DNF rate over all original rows)
- No NaN or -inf in any tensor
- Constructor 10 absorbs IDs 211 and 117 (assert in tests)

---

### Task 2 — Plackett-Luce Likelihood (`likelihood.py`)

**What:** A standalone, testable function that computes the Plackett-Luce log-probability
of an observed finishing order given performance scores.

**API:**
```python
def plackett_luce_log_prob(
    performances: torch.Tensor,   # (N_total,) — in finishing order within each race
    race_lengths: torch.Tensor,   # (R,)       — entries per race
) -> torch.Tensor:                # scalar
```

**Algorithm (padded approach for vectorisation):**
1. Scatter performances into a `(R, max_N)` padded tensor, padding positions = `-inf`
2. For each position `i` (column), compute:
   `log_P_i = perf[:, i] - logsumexp(perf[:, i:], dim=1, masked by race_length)`
3. Mask out positions beyond each race's length; sum remaining terms

**Acceptance criteria (hand-verifiable):**
- 3 drivers, performances `[2.0, 1.0, 0.0]`, correct ordering `[0, 1, 2]`:
  - Expected: `log(e²/(e²+e¹+e⁰)) + log(e¹/(e¹+e⁰)) + log(e⁰/e⁰)`
  - ≈ `-0.4076 + (-0.3133) + 0.0` ≈ `-0.7209`
  - `plackett_luce_log_prob(tensor([2., 1., 0.]), tensor([3]))` should equal this
- Worst possible ordering for same performances returns lower log-prob than correct
- Log-prob ≤ 0 always (it is a probability)
- `race_lengths=[3,3]` with two races returns sum of two individual race log-probs

---

### Task 2b — Prior Predictive Check

**What:** Before fitting any model, verify that the chosen priors produce plausible
F1 race outcomes via ancestral sampling. This is a precondition for trusting that
`sigma_s = 1.0` and `sigma_c = 1.0` are sensible weakly-informative priors, not
accidentally degenerate ones.

**Procedure:**
1. Sample 10 sets of prior parameters:
   ```python
   s = torch.randn(10, D)          # D = 20 representative drivers
   c_raw = torch.randn(10, K-1)    # K = 10 representative constructors
   c = torch.cat([c_raw, -c_raw.sum(-1, keepdim=True)], dim=-1)
   ```
2. For each sample, construct performances for a 20-driver race:
   `p = s[draw, driver_idx] + c[draw, cons_idx]`
3. Draw a synthetic finishing order by sampling from `PlackettLuce(softmax(p))`
4. Record: what fraction of races does the prior-fastest driver win across all 10 draws?
5. Record: what is the typical performance gap between P1 and P20 (top minus bottom)?

**Expected range:**
- Prior-fastest driver wins 30–60% of races. Below 20% → priors too flat (near-random
  outcomes). Above 80% → priors too sharp (nearly deterministic — prior already
  "knows" who wins).
- P1–P20 performance gap: roughly 2–4 units (2× sigma_s + 2× sigma_c at ±1σ).

**If check fails:**
- Too deterministic (win rate > 80%): reduce `sigma_s` or `sigma_c` to 0.5 and
  re-run; document the sensitivity in the report.
- Too random (win rate < 20%): increase to 1.5 and re-run.
- Document the chosen values and this check in `report_notes.md` §9.

**Output:** A short printed table — mean win rate, P1–P20 gap — logged to stdout.
No CSV needed. This is a manual sanity check, not a persistent artifact.

**Lives in:** `models/pgm_backend/tests/test_prior_predictive.py` — a single
pytest function that runs the check and asserts `0.20 <= win_rate <= 0.80`.

**Acceptance criteria:**
- Prior win rate for the strongest driver is between 20% and 80% across 100 draws
- P1–P20 performance gap is between 1.0 and 5.0 units
- Test passes with `python -m pytest models/pgm_backend/tests/test_prior_predictive.py -v`

---

### Task 3 — Model 1, SVI Path (`model_baseline.py` + `inference.py`)

**What:** Static skill model with proper Plackett-Luce likelihood and sum-to-zero
constructor constraint, trained with SVI. This is the first end-to-end model.

**Model (in `model_baseline.py`):**
```python
class BaselineModel:
    def model(self, dataset): ...
    def guide(self, dataset): ...
```

**Model internals:**
1. `s = pyro.sample("s", dist.Normal(0, sigma_s).expand([D]).to_event(1))`
2. Sum-to-zero constructor:
   ```python
   c_raw = pyro.sample("c_raw", dist.Normal(0, sigma_c).expand([K-1]).to_event(1))
   c = torch.cat([c_raw, -c_raw.sum(keepdim=True)])
   ```
3. Performance:
   ```python
   p = s[driver_idx] + c[cons_idx]          # (N_entries,)
   ```
   No masking needed — mechanical DNFs are not present in `driver_idx` or `cons_idx`.
4. Likelihood: `plackett_luce_log_prob(p, race_lengths)` called as a `pyro.factor`

**Guide:** Mean-field — `s` and `c_raw` each get `loc` + `scale` variational params.

**`inference.py`:**
```python
def train_svi(model, dataset, n_steps, lr, log_every) -> list[float]:
    """Returns ELBO loss history."""

def extract_svi_posterior(model) -> dict[str, torch.Tensor]:
    """Returns {'s_loc', 's_scale', 'c_loc', 'c_scale', ...}"""
```

**Acceptance criteria:**
- ELBO decreases over 3000 steps (final loss < initial loss)
- `c.sum()` ≈ 0 (tolerance 1e-4) after convergence
- Hamilton (`driverId=1`) in top-5 drivers by posterior mean `s_loc`
- SVI runs in < 5 minutes on CPU

---

### CHECKPOINT A — Model 1 SVI verified

At this point: data pipeline is correct, likelihood is unit-tested, and one full
model trains and produces plausible results. Do NOT proceed to Task 4 until
this checkpoint passes.

---

### Task 4 — Model 1, NUTS Path (`inference.py` addition)

**What:** Add NUTS inference to Model 1 and produce a SVI vs NUTS comparison.

**API addition to `inference.py`:**
```python
def run_nuts(model, dataset, num_warmup, num_samples) -> pyro.infer.MCMC:
    """Returns fitted MCMC object."""

def compare_svi_nuts(svi_posterior, mcmc_samples) -> pd.DataFrame:
    """Returns per-driver discrepancy table."""
```

**Implementation notes:**
- Use `pyro.infer.MCMC(pyro.infer.NUTS(model.model), ...)` 
- Pass `dataset` as the observed data argument
- Extract R-hat from `mcmc.diagnostics()`
- Log a `WARNING` for any R-hat ≥ 1.05
- Discrepancy metric: `|svi_mean_d - nuts_mean_d| / nuts_std_d` per driver

**Acceptance criteria:**
- R-hat < 1.05 for ≥ 90% of latents (log warning for outliers, do not crash)
- Max discrepancy between SVI and NUTS posterior means < 0.5 (standardised)
- NUTS completes in < 30 minutes on CPU (500 warmup + 500 samples)
- Comparison CSV written to `outputs/pgm_model/nuts_vs_svi_comparison.csv`

---

### Task 5 — Synthetic Recovery Tests (`tests/`)

**What:** Prove inference is working by generating data from known parameters and
recovering them. This must pass before building Models 2 and 3.

**File:** `models/pgm_backend/tests/test_synthetic_recovery.py`

**Test structure:**
```python
def test_baseline_recovery():
    # 1. Fix 5 drivers, 3 constructors, 50 synthetic races
    true_s = torch.tensor([2.5, 2.0, 1.5, 0.0, -1.5])
    true_c = torch.tensor([2.0, 0.5, -2.5])  # sum-to-zero
    # 2. Ancestral sample: for each race, draw eps, compute p, sort
    # 3. Build a minimal F1RankingDataset from synthetic data
    # 4. Run SVI (3000 steps)
    # 5. Assert: |posterior_mean - true_value| < 0.8 for all latents
    #    (looser than 0.5 because 50 races is small; SPEC §7 uses 0.5 for real data)
```

**Also include:**
- `test_likelihood_log_prob()` — the 3-driver hand check from Task 2 spec
- `test_constructor_sum_to_zero()` — post-inference assertion

**Acceptance criteria:**
- All tests pass with `python -m pytest models/pgm_backend/tests/ -v`
- Recovery within ±0.8 for all 5 drivers and 3 constructors (sign of ranking must match)

---

### CHECKPOINT B — Inference proven correct

Tasks 1–5 complete. Likelihood tested. Model 1 (SVI + NUTS) runs. Synthetic data
recovered. Do NOT build Models 2 or 3 until this checkpoint passes.

---

### Task 6 — Model 2, Extended (`model_extended.py`)

**What:** Temporal AR(1) skills, circuit effects, global weather coefficient.

**New latents vs Model 1:**
- `s` shape changes: `(D,)` static → `(D, T)` temporal via AR(1)
- `c_raw` shape: `(K-1,)` static → `(K-1, T)` temporal
- `e_circ`: `(C,)` circuit effects, `Normal(0, sigma_e)`
- `beta_w`: scalar, `Normal(0, 0.5)`

**AR(1) implementation (vectorised, no entity loop):**
```python
# Driver skills
s0 = pyro.sample("s0", dist.Normal(0, sigma_s).expand([D]).to_event(1))  # (D,)
s_innovations = pyro.sample(
    "s_innov", dist.Normal(0, gamma_s).expand([T-1, D]).to_event(2)
)  # (T-1, D)
s = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + s_innovations.cumsum(0)], dim=0)  # (T, D)
```
Do the same for constructors (with sum-to-zero applied at each time step).

**Performance equation:**
```python
p = (s[season_idx, driver_idx]
     + c[season_idx, cons_idx]
     + e_circ[circuit_idx]
     + beta_w * wet[race_idx])
```

**Acceptance criteria:**
- ELBO decreases over 5000 steps
- Seasonal `s[t, hamilton_idx]` shows increasing skill 2014–2020, declining trend after 2021
- Seasonal `c[t, mercedes_idx]` peaks 2014–2020
- SVI runs in < 15 minutes on CPU
- `c[:, :].sum(dim=1)` ≈ 0 for all T seasons (sum-to-zero holds temporally)

---

### Task 7 — Model 3, Full (`model_full.py`)

**What:** Extends Model 2 with driver wet-weather skill interaction and pit-stop covariate.

**New latents:**
- `delta_d`: `(D,)`, `Normal(0, sigma_delta=0.5)` — driver wet-weather modifier
- `beta_pi`: scalar, `Normal(0, 0.5)` — pit-stop coefficient

**Performance equation (addition to Model 2):**
```python
p = (s[season_idx, driver_idx]
     + c[season_idx, cons_idx]
     + e_circ[circuit_idx]
     + beta_w * wet[race_idx]
     + delta_d[driver_idx] * wet[race_idx]   # INTERACTION: multiply, not add
     + beta_pi * pit_norm)
```

**Acceptance criteria:**
- ELBO decreases over 5000 steps
- `delta_d` posterior is non-trivial (std > 0.1, not collapsed to prior)
- `beta_pi` posterior mean is negative (faster pit stops → better performance)
- SVI runs in < 20 minutes on CPU
- At least one known wet-weather specialist (Alonso=driverId 4, Webber=driverId 13,
  or similar) ranks in top-5 by `delta_d` posterior mean

---

### CHECKPOINT C — All three models trained and sane

Tasks 6 and 7 complete. Full SPEC §9.2 sanity checks pass for all three models.

---

### Task 8 — Posterior Extraction + Orchestrator (`posterior.py` + `run_pgm.py`)

**What:** A unified posterior extractor and a single entry point that trains all three
models, exports CSVs, and prints a comparison table.

**`posterior.py`:**
```python
def extract_posterior(model_name: str, param_store, maps: dict) -> pd.DataFrame:
    """Returns tidy DataFrame: entity_type, entity_id, entity_name, season (or 'all'),
       mu, sigma."""
```

**`run_pgm.py` flow:**
1. Build `F1RankingDataset` (call `data_preparation.py`)
2. Train Model 1 (SVI + NUTS), write `outputs/pgm_model/baseline_posterior.csv`
   and `nuts_vs_svi_comparison.csv`
3. Train Model 2, write `extended_posterior.csv`
4. Train Model 3, write `full_posterior.csv`
5. Plot ELBO curves for all three SVI runs → `elbo_curves.png`
6. Print top-10 driver and constructor table to stdout for all three models

**Acceptance criteria:**
- `python -m models.pgm_backend.run_pgm` runs end-to-end without errors
- All 4 output CSVs exist after run
- `elbo_curves.png` shows three monotonically decreasing curves
- Runtime < 45 minutes total on CPU

---

## Phase Summary

| Phase | Tasks | Gate |
|---|---|---|
| **Phase 1 — Foundation** | 1, 2, 2b | Data + likelihood verified; priors confirmed plausible |
| **Phase 2 — Baseline** | 3, 4, 5 | CHECKPOINT A + B: Model 1 fully proven |
| **Phase 3 — Extensions** | 6, 7 | CHECKPOINT C: Models 2 & 3 sane |
| **Phase 4 — Polish** | 8 | Full pipeline end-to-end |

---

## Key Engineering Risks

| Risk | Mitigation |
|---|---|
| AR(1) cumsum approach introduces gradient issues | Use `cumsum(0)` on innovations, not recursive sampling — avoids nested `pyro.sample` |
| `plackett_luce_log_prob` called as `pyro.factor` with wrong shape | Always return a scalar; verify with a 1-race test before multi-race |
| Sum-to-zero in guide must match model reparameterisation | Guide samples `c_raw` with shape `(K-1,)` — never `c` directly |
| Mechanical DNF Bernoulli (Model 3) needs intercept | Use `sigmoid(-alpha_rel - c_k)`, not `sigmoid(-c_k)` — baseline DNF rate ≈17%, so `alpha_rel ≈ 1.6` at convergence |
| NUTS memory explosion with temporal model | NUTS only on Model 1 (D+K-1 ≈ 85 latents); Models 2&3 are SVI only |
| `wet[race_idx]` shape mismatch | `wet` is `(R,)`, `race_idx` is `(N_total,)` — indexing gives `(N_total,)` ✓ |
