# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T2b  
**Task name:** Prior Predictive Check (`tests/test_prior_predictive.py`)

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## Context from T1 and T2

### Actual dataset counts (from T1)
- `n_drivers` = 77, `n_constructors` = 17

### Likelihood function (from T2)
```python
from models.pgm_backend.likelihood import plackett_luce_log_prob
# signature: (performances: (N,), race_lengths: (R,) LongTensor) -> scalar
```

T2b does **not** use these directly — it is a standalone prior predictive check.

---

## What to Build

Create one file:

**`models/pgm_backend/tests/test_prior_predictive.py`**

A single pytest function that verifies the chosen priors (`sigma_s = 1.0`, `sigma_c = 1.0`) produce plausible F1 race outcomes before any inference is run.

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/tests/test_prior_predictive.py` | Create |

Do not touch any other file.

---

## Full Implementation Spec

### Purpose

Confirm that `sigma_s = 1.0` and `sigma_c = 1.0` are weakly informative — not so flat that outcomes are near-random, not so sharp that one driver wins almost every race.

### Setup

Use **D = 20 representative drivers** and **K = 10 representative constructors** with a 20-driver race. Run **N_DRAWS = 100** independent prior draws.

```python
import torch
import pytest

D = 20      # representative drivers
K = 10      # representative constructors
N_DRAWS = 100
SIGMA_S = 1.0
SIGMA_C = 1.0
```

### Algorithm

For each of the 100 draws:

1. **Sample prior parameters:**
   ```python
   s = torch.randn(D) * SIGMA_S              # (D,)
   c_raw = torch.randn(K - 1) * SIGMA_C      # (K-1,)
   c = torch.cat([c_raw, -c_raw.sum(keepdim=True)])  # (K,), sum-to-zero
   ```

2. **Assign each driver to a constructor** (fixed round-robin across draws):
   ```python
   cons_assignment = torch.arange(D) % K     # (D,) — stable, not resampled
   ```

3. **Compute performance scores:**
   ```python
   p = s + c[cons_assignment]  # (D,)
   ```

4. **Identify the prior-fastest driver** (highest p):
   ```python
   fastest_driver = p.argmax()
   ```

5. **Sample a finishing order from Plackett-Luce** (no torch.distributions.PlackettLuce needed — use the sequential sampling procedure):
   ```python
   # PL draw: repeatedly sample winner from softmax of remaining scores
   remaining = list(range(D))
   winner = None
   order = []
   for _ in range(D):
       scores = p[torch.tensor(remaining)]
       probs = torch.softmax(scores, dim=0)
       pick = torch.multinomial(probs, 1).item()
       order.append(remaining[pick])
       remaining.pop(pick)
   winner = order[0]
   ```

6. **Record whether the fastest driver won:**
   ```python
   fastest_won = (winner == fastest_driver.item())
   ```

### Assertions

```python
win_rate = sum(fastest_won_list) / N_DRAWS

# P1-P20 gap: compute across all draws, take the mean
# gap_draw = p.max() - p.min()  (before drawing the order)
mean_gap = sum(gaps) / N_DRAWS

assert 0.20 <= win_rate <= 0.80, (
    f"Prior win rate {win_rate:.2f} outside [0.20, 0.80] — "
    "priors may be too flat or too sharp"
)
assert 1.0 <= mean_gap <= 5.0, (
    f"Mean P1-P20 performance gap {mean_gap:.2f} outside [1.0, 5.0]"
)
```

Print a summary table before asserting:
```
Prior predictive check (100 draws, 20 drivers, 10 constructors):
  Prior-fastest driver win rate: 0.XX
  Mean P1-P20 performance gap:   X.XX
```

### Full function skeleton

```python
def test_prior_predictive():
    torch.manual_seed(42)

    D, K = 20, 10
    N_DRAWS = 100
    SIGMA_S, SIGMA_C = 1.0, 1.0

    cons_assignment = torch.arange(D) % K  # fixed round-robin

    fastest_won_list = []
    gaps = []

    for _ in range(N_DRAWS):
        s = torch.randn(D) * SIGMA_S
        c_raw = torch.randn(K - 1) * SIGMA_C
        c = torch.cat([c_raw, -c_raw.sum(keepdim=True)])
        p = s + c[cons_assignment]

        fastest_driver = p.argmax().item()
        gaps.append((p.max() - p.min()).item())

        # PL sequential draw
        remaining = list(range(D))
        order = []
        for _ in range(D):
            scores = p[torch.tensor(remaining)]
            probs = torch.softmax(scores, dim=0)
            pick = torch.multinomial(probs, 1).item()
            order.append(remaining[pick])
            remaining.pop(pick)

        fastest_won_list.append(order[0] == fastest_driver)

    win_rate = sum(fastest_won_list) / N_DRAWS
    mean_gap = sum(gaps) / N_DRAWS

    print(f"\nPrior predictive check ({N_DRAWS} draws, {D} drivers, {K} constructors):")
    print(f"  Prior-fastest driver win rate: {win_rate:.2f}")
    print(f"  Mean P1-P20 performance gap:   {mean_gap:.2f}")

    assert 0.20 <= win_rate <= 0.80, ...
    assert 1.0 <= mean_gap <= 5.0, ...
```

---

## Verification Commands

```bash
# Run the prior predictive test
uv run python -m pytest models/pgm_backend/tests/test_prior_predictive.py -v -s
```

The `-s` flag ensures the print output is visible.

---

## Acceptance Criteria

- [ ] Prior-fastest driver win rate is between 0.20 and 0.80 (over 100 draws, seed=42)
- [ ] Mean P1–P20 performance gap is between 1.0 and 5.0
- [ ] Test passes with `pytest models/pgm_backend/tests/test_prior_predictive.py -v`
- [ ] Summary table is printed to stdout

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Report the actual win rate and mean gap values
- Note whether the test passed or required any adjustment to the bounds
- Note if sigma values needed to change (they should not — this is just a verification)

**Step 2 — Append to `tasks/report_notes.md`** (this IS a non-obvious decision for the report):

```markdown
## T2b — Prior predictive check: sigma_s=1.0, sigma_c=1.0 confirmed plausible

**Decision:** sigma_s = 1.0, sigma_c = 1.0 are retained as prior scales.

**Reasoning:** Prior predictive check (100 draws, 20 drivers, 10 constructors) showed
prior-fastest driver win rate = [actual value] and mean P1–P20 gap = [actual value],
both within acceptable bounds [0.20, 0.80] and [1.0, 5.0].

**For the report:** Priors are weakly informative — they allow the data to dominate
inference without being nearly uniform (which would make the model unidentifiable).
```

**Step 3 — Report back:**
1. Full output of the verification command (include the printed summary table)
2. Whether both assertions passed
3. The actual win rate and gap values

Then stop. Do not implement T3.
