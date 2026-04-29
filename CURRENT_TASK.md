# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T5  
**Task name:** Synthetic Recovery Tests (`tests/test_synthetic_recovery.py` + `test_likelihood.py`)

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## Context from T4

### NUTS vs SVI comparison is complete and sane
- `inference.py` has `run_nuts()` and `compare_svi_nuts()` — both work correctly
- NUTS converges well: 100% R-hat < 1.05, max R-hat = 1.0134
- Mean-field SVI shows expected bias vs NUTS (constructors more affected than drivers)
- This is a known VI limitation, not a bug — do not attempt to "fix" it

### Model 1 SVI architecture (unchanged from T3)
- `BaselineModel` in `model_baseline.py`
- `train_svi()` and `extract_svi_posterior()` in `inference.py`
- SVI uses `Trace_ELBO`, Adam optimiser, mean-field guide
- Param store keys after training: `"s_loc"`, `"s_scale"`, `"c_loc"`, `"c_scale"`
- `extract_svi_posterior()` returns `c_loc` as full `(K,)` with derived K-th entry
- Sum-to-zero is exact: `c_loc.sum() == 0.0`

### Dataset counts (unchanged)
- `n_drivers` = 77, `n_constructors` = 17, `n_races` = 286
- `N_entries` (ranking) = 5457
- Constructor map has 17 entries (indices 0–16)

---

## What to Build

Create two test files in `models/pgm_backend/tests/`:
1. `test_likelihood.py` — unit tests for the Plackett-Luce likelihood
2. `test_synthetic_recovery.py` — end-to-end synthetic data recovery test

No other file should be modified.

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/tests/test_likelihood.py` | Create |
| `models/pgm_backend/tests/test_synthetic_recovery.py` | Create |

Do not touch any other file.

---

## Full Implementation Spec

### File 1: `test_likelihood.py`

Two test functions:

```python
def test_plackett_luce_hand_check():
    """
    3 drivers, performances [2.0, 1.0, 0.0], race_lengths [3].
    Expected log-prob ≈ -0.7209 (tolerance 1e-3).
    """
    ...

def test_plackett_luce_log_prob_nonpos():
    """
    Log-prob must be ≤ 0 for any valid input.
    Test with 20 random races of random lengths.
    """
    ...
```

Implementation requirements:
- Import `plackett_luce_log_prob` from `models.pgm_backend.likelihood`
- For `test_plackett_luce_hand_check`:
  - `performances = torch.tensor([2.0, 1.0, 0.0])`
  - `race_lengths = torch.tensor([3])`
  - Expected value: `math.log(math.exp(2) / (math.exp(2) + math.exp(1) + 1)) + math.log(math.exp(1) / (math.exp(1) + 1)) + 0.0`
  - Assert `abs(actual - expected) < 1e-3`
- For `test_plackett_luce_log_prob_nonpos`:
  - Set a fixed seed (`torch.manual_seed(42)`) for determinism
  - Generate 20 races with lengths uniformly in [2, 10] and performances from `Normal(0, 1)`
  - Assert all returned log-probs ≤ 0

---

### File 2: `test_synthetic_recovery.py`

One main test function:

```python
def test_baseline_recovery():
    """
    Generate synthetic races from known parameters, fit Model 1 with SVI,
    and assert posterior means recover true values within ±0.8.
    """
    ...
```

**Step-by-step spec:**

1. **Fix ground-truth parameters:**
   ```python
   true_s = torch.tensor([2.5, 2.0, 1.5, 0.0, -1.5])   # 5 drivers
   true_c = torch.tensor([2.0, 0.5, -2.5])              # 3 constructors, sum-to-zero
   N_RACES = 50
   ```

2. **Generate synthetic races:**
   - For each race `r` in `range(N_RACES)`:
     - Randomly assign each of the 5 drivers to one of the 3 constructors (uniform)
     - Compute performance: `p = true_s + true_c[cons_assignment]` (no noise — the Plackett-Luce likelihood itself provides stochasticity)
     - Draw a synthetic finishing order by sequential PL sampling:
       ```python
       remaining = list(range(5))
       order = []
       for _ in range(5):
           scores = p[torch.tensor(remaining)]
           probs = torch.softmax(scores, dim=0)
           pick = torch.multinomial(probs, 1).item()
           order.append(remaining[pick])
           remaining.pop(pick)
       ```
     - Record `driver_idx`, `cons_idx`, and `race_length = 5`
   - Use a fixed random seed (`torch.manual_seed(123)`) for determinism

3. **Build a minimal dataset:**
   - You need an object that has the same attributes as `F1RankingDataset` that `BaselineModel.model()` and `train_svi()` expect:
     - `driver_idx`: LongTensor `(N_ENTRIES,)` — driver index per finishing position
     - `cons_idx`: LongTensor `(N_ENTRIES,)` — constructor index per finishing position
     - `race_lengths`: LongTensor `(N_RACES,)` — all equal to 5
   - The simplest way: create a lightweight class or `types.SimpleNamespace` with these three attributes
   - `N_ENTRIES = 5 * 50 = 250`

4. **Run SVI:**
   ```python
   from models.pgm_backend.model_baseline import BaselineModel
   from models.pgm_backend.inference import train_svi, extract_svi_posterior

   model = BaselineModel(n_drivers=5, n_constructors=3)
   losses = train_svi(model, dataset, n_steps=3000, lr=0.01, log_every=500)
   posterior = extract_svi_posterior(model)
   ```

5. **Assertions:**
   - `len(losses) == 3000`
   - `losses[-1] < losses[0]` (ELBO decreased)
   - For each driver `d` in 0..4: `abs(posterior["s_loc"][d] - true_s[d]) < 0.8`
   - For each constructor `k` in 0..2: `abs(posterior["c_loc"][k] - true_c[k]) < 0.8`
   - `posterior["c_loc"].sum()` ≈ 0 (tolerance 1e-4)
   - The sign of the ranking must match: `torch.argsort(posterior["s_loc"])` should equal `torch.argsort(true_s)` (or reversed, since lower = worse)

6. **Print a summary:**
   ```python
   print("\nSynthetic recovery results:")
   for d in range(5):
       print(f"  Driver {d}: true={true_s[d]:.2f}, inferred={posterior['s_loc'][d]:.2f}")
   for k in range(3):
       print(f"  Constructor {k}: true={true_c[k]:.2f}, inferred={posterior['c_loc'][k]:.2f}")
   ```

---

## Verification Commands

```bash
# 1. Run all tests
uv run python -m pytest models/pgm_backend/tests/ -v

# 2. Run just the new files
uv run python -m pytest models/pgm_backend/tests/test_likelihood.py -v
uv run python -m pytest models/pgm_backend/tests/test_synthetic_recovery.py -v
```

---

## Acceptance Criteria

- [ ] `test_likelihood.py` passes: hand-check ≈ -0.7209 and all log-probs ≤ 0
- [ ] `test_synthetic_recovery.py` passes: all driver and constructor errors < 0.8
- [ ] Ranking signs match (argsort of posterior means matches argsort of true values)
- [ ] `c_loc.sum()` ≈ 0 within 1e-4
- [ ] SVI ELBO decreases over 3000 steps
- [ ] All tests in `models/pgm_backend/tests/` pass together (`pytest -v`)
- [ ] No NaN or inf in any posterior parameter
- [ ] Tests run in < 5 minutes total on CPU

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Report actual inferred vs true values for all 5 drivers and 3 constructors
- Report max absolute error
- Report whether ranking signs match
- Report ELBO initial/final values
- Note any deviations from the spec

**Step 2 — Append to `tasks/report_notes.md`** only if you made a non-obvious decision (e.g., had to add noise to performances, changed the number of races, etc.).

**Step 3 — Report back:**
1. Full output of `pytest models/pgm_backend/tests/ -v`
2. Whether all acceptance criteria pass
3. Actual max error and whether ranking signs match

Then stop. Do not implement T6.
