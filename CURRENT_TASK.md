# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T4  
**Task name:** Model 1 Baseline — NUTS path + SVI vs NUTS comparison (`inference.py` extension)

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## Context from T3

### Dataset counts (unchanged)
- `n_drivers` = 77, `n_constructors` = 17, `n_races` = 286
- `N_entries` (ranking) = 5457
- Constructor map has 17 entries (indices 0–16); index 6 maps to constructorId 10 (Force India)

### Model 1 SVI is trained and sane
- `BaselineModel` lives in `models/pgm_backend/model_baseline.py`
- `train_svi()` and `extract_svi_posterior()` live in `models/pgm_backend/inference.py`
- SVI param store keys after training: `"s_loc"` (77,), `"s_scale"` (77,), `"c_loc"` (16,), `"c_scale"` (16,)
- `extract_svi_posterior()` returns `c_loc` as (17,) with the derived K-th entry appended
- Sum-to-zero is exact: `c_loc.sum() == 0.0`
- Hamilton (`driverId=1`) ranks #4–5 with `s_loc ≈ 0.72`
- Top driver IDs by `s_loc`: [857, 846, 832, 830, 1, 20, 18, 844, 8, 4]
- No NaN/inf in any posterior parameter
- SVI runtime on CPU (M1 Pro) is < 2 minutes for 3000 steps
- `BaselineModel.model()` takes exactly `(driver_idx, cons_idx, race_lengths)` — no other arguments

---

## What to Build

Extend `models/pgm_backend/inference.py` with:
1. A NUTS training function
2. A comparison function that produces a CSV

No other file should be modified.

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/inference.py` | Modify (add two functions) |

Do not touch any other file.

---

## Full Implementation Spec

### API to add

```python
def run_nuts(model, dataset, num_warmup=500, num_samples=500):
    """
    Run NUTS on Model 1.

    Args:
        model:       BaselineModel instance
        dataset:     F1RankingDataset instance
        num_warmup:  HMC warmup iterations
        num_samples: HMC post-warmup samples

    Returns:
        mcmc: pyro.infer.MCMC object (already run)
    """
    ...


def compare_svi_nuts(svi_posterior, mcmc, dataset) -> pd.DataFrame:
    """
    Build per-driver and per-constructor discrepancy table.

    Args:
        svi_posterior: dict returned by extract_svi_posterior()
        mcmc:          fitted MCMC object from run_nuts()
        dataset:       F1RankingDataset instance (for reverse maps)

    Returns:
        pd.DataFrame with columns:
            - entity_type : 'driver' or 'constructor'
            - entity_id   : original F1 ID (driverId or constructorId)
            - entity_name : original F1 name (from dataset.driver_map / constructor_map if available, else empty)
            - svi_mean    : SVI posterior mean
            - nuts_mean   : NUTS posterior mean
            - nuts_std    : NUTS posterior std
            - discrepancy : |svi_mean - nuts_mean| / nuts_std  (standardised)
            - r_hat       : NUTS R-hat (if available, else NaN)
    """
    ...
```

### 1. `run_nuts`

Implementation requirements:
- Use `pyro.infer.NUTS(model.model)`
- Use `pyro.infer.MCMC(kernel, num_samples=num_samples, warmup_steps=num_warmup, ...)`
- Run `mcmc.run(driver_idx=dataset.driver_idx, cons_idx=dataset.cons_idx, race_lengths=dataset.race_lengths)`
- **Clear the Pyro param store before running NUTS** to avoid name collisions with SVI (`pyro.clear_param_store()`)
- Return the `mcmc` object (already executed)

### 2. `compare_svi_nuts`

Implementation requirements:
- Extract NUTS samples via `mcmc.get_samples()` — this returns a dict with keys `"s"` and `"c_raw"`
- Derive full constructor posterior from `c_raw` samples:
  ```python
  c_samples = torch.cat([
      c_raw_samples,
      -c_raw_samples.sum(dim=-1, keepdim=True)
  ], dim=-1)  # shape: (num_samples, K)
  ```
  Note: `dim=-1` on the sum because `c_raw_samples` has shape `(num_samples, K-1)`.
- Compute per-latent NUTS mean and std:
  - `s_nuts_mean = s_samples.mean(0)`  # (D,)
  - `s_nuts_std  = s_samples.std(0)`   # (D,)
  - `c_nuts_mean = c_samples.mean(0)`  # (K,)
  - `c_nuts_std  = c_samples.std(0)`   # (K,)
- Compute discrepancy for drivers:
  ```python
  s_svi = svi_posterior["s_loc"]  # (D,)
  s_disc = torch.abs(s_svi - s_nuts_mean) / s_nuts_std
  ```
- Compute discrepancy for constructors similarly using `svi_posterior["c_loc"]` and `c_nuts_mean`/`c_nuts_std`
- Extract R-hat from `mcmc.diagnostics()`:
  ```python
  diag = mcmc.diagnostics()
  s_rhat = diag["s"]["r_hat"]          # array shape (D,)
  c_raw_rhat = diag["c_raw"]["r_hat"]  # array shape (K-1,)
  # The K-th constructor has no direct R-hat because it is derived, not sampled.
  # Set its R-hat to NaN.
  ```
- Build a tidy DataFrame with one row per latent entity:
  - Drivers: 77 rows
  - Constructors: 17 rows
  - Columns exactly as listed in the docstring above
  - `entity_id` for drivers should be `dataset.driver_map[idx]` (reverse mapping from index to original driverId)
  - `entity_id` for constructors should be `dataset.constructor_map[idx]` (reverse mapping from index to original constructorId)
  - `entity_name` can be empty string if no name field is available in the dataset (there is no name field in F1RankingDataset — leave it as "")
- **Create the output directory if it does not exist:** `outputs/pgm_model/`
- **Write CSV:** `outputs/pgm_model/nuts_vs_svi_comparison.csv`
- Return the DataFrame

### 3. R-hat warnings

Inside `compare_svi_nuts`, after extracting R-hat:
```python
import logging
logger = logging.getLogger(__name__)
all_rhats = ...  # concatenate s_rhat and c_raw_rhat (exclude the derived K-th constructor)
bad = all_rhats[all_rhats >= 1.05]
if len(bad) > 0:
    logger.warning(f"R-hat >= 1.05 for {len(bad)} / {len(all_rhats)} latents")
```

---

## Verification Commands

```bash
# 1. Run SVI + NUTS end-to-end and write comparison CSV
uv run python -c "
import torch, pandas as pd, os
import pyro
from models.pgm_backend.data_preparation import load_dataset
from models.pgm_backend.model_baseline import BaselineModel
from models.pgm_backend.inference import train_svi, extract_svi_posterior, run_nuts, compare_svi_nuts

ds = load_dataset()
model = BaselineModel(ds.n_drivers, ds.n_constructors)

# SVI path
print('--- Training SVI ---')
losses = train_svi(model, ds, n_steps=3000, lr=0.01, log_every=500)
svi_post = extract_svi_posterior(model)

# NUTS path
print('\n--- Running NUTS ---')
pyro.clear_param_store()
mcmc = run_nuts(model, ds, num_warmup=500, num_samples=500)

# Comparison
print('\n--- Comparing ---')
df = compare_svi_nuts(svi_post, mcmc, ds)
print(df.head())
print(f'CSV written: {os.path.exists(\"outputs/pgm_model/nuts_vs_svi_comparison.csv\")}')

# Acceptance checks
s_disc = df[df.entity_type == 'driver']['discrepancy']
c_disc = df[df.entity_type == 'constructor']['discrepancy']
print(f'Max driver discrepancy:   {s_disc.max():.4f}')
print(f'Max constructor discrepancy: {c_disc.max():.4f}')

# R-hat check
rhats = df[df.entity_type == 'driver']['r_hat'].dropna()
rhats = pd.concat([rhats, df[df.entity_type == 'constructor']['r_hat'].dropna()])
good = (rhats < 1.05).mean()
print(f'R-hat < 1.05 fraction: {good:.2%}')
print(f'Max R-hat: {rhats.max():.4f}')
"
```

---

## Acceptance Criteria

- [ ] `run_nuts` executes without error and returns a fitted `MCMC` object
- [ ] `compare_svi_nuts` returns a DataFrame with exactly the columns listed in the spec
- [ ] CSV is written to `outputs/pgm_model/nuts_vs_svi_comparison.csv`
- [ ] R-hat < 1.05 for ≥ 90% of sampled latents (drivers + K-1 constructors; the derived K-th constructor is excluded from this check)
- [ ] Max standardised discrepancy (`|svi_mean - nuts_mean| / nuts_std`) < 0.5 across all drivers and constructors
- [ ] NUTS completes in < 30 minutes on CPU (500 warmup + 500 samples)
- [ ] No NaN or inf in any posterior statistic

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Report initial/final SVI loss values
- Report NUTS runtime
- Report max discrepancy (driver and constructor separately)
- Report R-hat fraction < 1.05 and max R-hat
- Report actual top-5 driver IDs by SVI mean vs NUTS mean
- Note any deviations from the spec

**Step 2 — Append to `tasks/report_notes.md`** only if you made a non-obvious decision (e.g. had to adjust NUTS parameters, encountered divergences, etc.).

**Step 3 — Report back:**
1. Full output of the verification command
2. Whether all acceptance criteria pass
3. Actual max discrepancy and R-hat statistics

Then stop. Do not implement T5.
