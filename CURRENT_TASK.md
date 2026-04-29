# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T2  
**Task name:** Plackett-Luce Likelihood (`likelihood.py`)

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## Context from T1

The dataset is built and verified. Actual values:

| Field | Value |
|---|---|
| n_races | 286 |
| n_drivers | 77 |
| n_constructors | 17 (after remap) |
| N_entries | 5457 (ranking rows — no mechanical DNFs) |
| N_all | 5980 (all original rows) |
| race_lengths range | [12, 24] |

Key facts T2 must know:
- `wet` is per-race shape `(286,)`. To get per-entry values: `wet[race_idx]`. **T2 does not use this — it is noted for T3.**
- `race_lengths` is a LongTensor `(286,)` where each value is the count of ranking entries for that race. `race_lengths.sum() == 5457`.
- `race_order` is 0-indexed within each race (0 = winner). The performances tensor passed to `plackett_luce_log_prob` must already be sorted so that position 0 in each race block is the winner. This is guaranteed by how `data_preparation.py` builds `race_order` (sort by `positionOrder` ascending within each race).

**T2 is a pure function — it does not import `data_preparation.py`.** It only depends on PyTorch.

---

## What to Build

Create `models/pgm_backend/likelihood.py` with a single exported function:

```python
def plackett_luce_log_prob(
    performances: torch.Tensor,   # (N_total,) — in finishing order, winner first
    race_lengths: torch.Tensor,   # (R,) LongTensor — entries per race
) -> torch.Tensor:                # scalar
```

This function computes the Plackett-Luce log-probability of observing a set of race
finishing orders given latent performance scores.

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/likelihood.py` | Create |

Do not touch any other file.

---

## Full Implementation Spec

### What Plackett-Luce computes

For a single race with R drivers and performances `[p_0, p_1, ..., p_{R-1}]` in
finishing order (index 0 = winner):

```
log P = Σ_{i=0}^{R-1} [ p_i - log Σ_{j=i}^{R-1} exp(p_j) ]
      = Σ_{i=0}^{R-1} [ p_i - logsumexp(p_i, ..., p_{R-1}) ]
```

The last term always contributes 0 (`p_{R-1} - logsumexp([p_{R-1}]) = 0`) and can
be included or excluded — including it is fine.

### Algorithm: padded matrix approach

This function handles multiple races of different lengths in one vectorised pass.

**Step 1 — Reshape into padded matrix:**

Scatter the flat `(N_total,)` performances into a `(R, max_N)` matrix where
`max_N = race_lengths.max()`. Positions beyond each race's length are filled with `-inf`.

```python
max_N = race_lengths.max().item()
padded = torch.full((R, max_N), float('-inf'))
# fill row r with performances[offset : offset + race_lengths[r]]
```

Use a cumulative-sum offset to locate each race's block in the flat tensor.

**Step 2 — Compute per-position log-prob contributions:**

For each column `i` in `[0, max_N)`:
```python
# logsumexp over columns i..max_N-1 (padding -inf is correctly ignored by logsumexp)
tail_lse = torch.logsumexp(padded[:, i:], dim=1)   # (R,)
log_p_i  = padded[:, i] - tail_lse                  # (R,)
```

**Step 3 — Mask and sum:**

Build a validity mask: position `i` in race `r` is valid iff `i < race_lengths[r]`.

```python
# valid[r, i] = True iff i < race_lengths[r]
i_range = torch.arange(max_N, device=performances.device)
valid = i_range.unsqueeze(0) < race_lengths.unsqueeze(1)  # (R, max_N)
```

Where `valid` is False, `log_p_i` will be `nan` (from `-inf - (-inf)`). Zero these
out before summing:

```python
log_p_matrix[:, i] = torch.where(valid[:, i], log_p_i, torch.zeros_like(log_p_i))
```

Return `log_p_matrix[valid].sum()` or equivalently the masked sum. The result is a
scalar.

### Numerical correctness notes

- `torch.logsumexp` handles `-inf` entries correctly (ignores them), so no special
  treatment of padding is needed in the logsumexp call itself.
- The only dangerous case is `padded[:, i] - tail_lse` when both are `-inf` (i.e.,
  column `i` is entirely padding). Use `torch.where(valid, ...)` to mask these to 0.
- The function must return a scalar (0-dim tensor), not a `(1,)` tensor.

### Hand-verifiable example

Three drivers with performances `[2.0, 1.0, 0.0]` in correct order:

```
term 0: 2.0 - logsumexp([2.0, 1.0, 0.0]) = 2.0 - log(e² + e¹ + e⁰)
                                           = 2.0 - log(11.107) ≈ 2.0 - 2.4076 = -0.4076
term 1: 1.0 - logsumexp([1.0, 0.0])       = 1.0 - log(e¹ + e⁰)
                                           = 1.0 - log(3.718) ≈ 1.0 - 1.3133 = -0.3133
term 2: 0.0 - logsumexp([0.0])             = 0.0 - 0.0 = 0.0

total ≈ -0.7209
```

---

## Verification Commands

Run these after implementation. Report all output.

```bash
# 1. Hand check — must print approx -0.7209
uv run python -c "
import torch
from models.pgm_backend.likelihood import plackett_luce_log_prob

# 3-driver single race
lp = plackett_luce_log_prob(torch.tensor([2., 1., 0.]), torch.tensor([3]))
print('3-driver log-prob:', lp.item())
assert abs(lp.item() - (-0.7209)) < 1e-3, f'Expected -0.7209, got {lp.item()}'
print('Hand check: PASSED')
"

# 2. Ordering check — correct order must beat reversed order
uv run python -c "
import torch
from models.pgm_backend.likelihood import plackett_luce_log_prob

correct  = plackett_luce_log_prob(torch.tensor([2., 1., 0.]), torch.tensor([3]))
reversed_ = plackett_luce_log_prob(torch.tensor([0., 1., 2.]), torch.tensor([3]))
print('Correct order log-prob:', correct.item())
print('Reversed order log-prob:', reversed_.item())
assert correct > reversed_, 'Correct ordering must have higher log-prob'
print('Ordering check: PASSED')
"

# 3. Two-race additivity check
uv run python -c "
import torch
from models.pgm_backend.likelihood import plackett_luce_log_prob

perfs_r1 = torch.tensor([2., 1., 0.])
perfs_r2 = torch.tensor([1., 0., -1.])
joint = plackett_luce_log_prob(torch.cat([perfs_r1, perfs_r2]), torch.tensor([3, 3]))
r1_only = plackett_luce_log_prob(perfs_r1, torch.tensor([3]))
r2_only = plackett_luce_log_prob(perfs_r2, torch.tensor([3]))
print('Joint:', joint.item())
print('R1 + R2:', (r1_only + r2_only).item())
assert abs(joint.item() - (r1_only + r2_only).item()) < 1e-5, 'Additivity failed'
print('Two-race additivity: PASSED')
"

# 4. Non-positivity check
uv run python -c "
import torch
from models.pgm_backend.likelihood import plackett_luce_log_prob

for _ in range(20):
    n = torch.randint(2, 10, (1,)).item()
    perfs = torch.randn(n)
    lp = plackett_luce_log_prob(perfs, torch.tensor([n]))
    assert lp.item() <= 1e-6, f'Log-prob > 0: {lp.item()}'
print('Non-positivity check: PASSED (20 random races)')
"

# 5. Mixed race lengths
uv run python -c "
import torch
from models.pgm_backend.likelihood import plackett_luce_log_prob

perfs = torch.tensor([3., 2., 1., 0.,   # race 1: 4 drivers
                       1., 0.])           # race 2: 2 drivers
lp = plackett_luce_log_prob(perfs, torch.tensor([4, 2]))
print('Mixed race lengths log-prob:', lp.item())
assert lp.item() <= 0
assert not torch.isnan(lp)
print('Mixed race lengths: PASSED')
"
```

---

## Acceptance Criteria

- [ ] `plackett_luce_log_prob(tensor([2., 1., 0.]), tensor([3]))` ≈ -0.7209 (tolerance 1e-3)
- [ ] Correct ordering returns strictly higher log-prob than reversed ordering
- [ ] Two-race joint log-prob equals sum of individual log-probs (additivity, tolerance 1e-5)
- [ ] Log-prob ≤ 0 for 20 random inputs
- [ ] Mixed race lengths (e.g., `[4, 2]`) returns a finite scalar, no NaN
- [ ] All five verification commands run without error

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Confirm which verification commands passed
- Note any deviations from the algorithm spec (e.g. if you used a different vectorisation)
- Note the actual value of the 3-driver hand check

**Step 2 — Report back:**
1. The full output of all five verification commands
2. Any deviations from the spec you made, and why
3. Whether all acceptance criteria pass (yes/no per criterion)

Then stop. Do not implement T3.
