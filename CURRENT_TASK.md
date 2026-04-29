# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T1  
**Task name:** Data Preparation — F1RankingDataset

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## What to Build

Create `models/pgm_backend/data_preparation.py` that loads `data_preprocessing/f1_enriched.csv`
and emits a verified `F1RankingDataset` dataclass used by all three models.

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/__init__.py` | Create (empty) if it does not exist |
| `models/pgm_backend/data_preparation.py` | Create |

Do not touch any other file.

---

## Full Implementation Spec

See `tasks/plan.md` → **Task 1 — Data Preparation** for the complete spec.

Key points reproduced here for convenience:

**Output dataclass:**
```python
@dataclass
class F1RankingDataset:
    # Ranking entries only (N_entries = finishers + driver-fault DNFs, NO mechanical DNFs)
    driver_idx: torch.Tensor    # (N_entries,) LongTensor
    cons_idx: torch.Tensor      # (N_entries,) LongTensor
    season_idx: torch.Tensor    # (N_entries,) LongTensor
    circuit_idx: torch.Tensor   # (N_entries,) LongTensor
    race_idx: torch.Tensor      # (N_entries,) LongTensor

    # Covariates (ranking entries)
    pit_norm: torch.Tensor      # (N_entries,) FloatTensor — normalised per season
    wet: torch.Tensor           # (N_races,) FloatTensor — binary wet indicator

    # Ranking
    race_order: torch.Tensor    # (N_entries,) LongTensor — 0=winner within-race
    race_lengths: torch.Tensor  # (N_races,) LongTensor — entries per race

    # Model 3 reliability fields (N_all = all original rows including mechanical DNFs)
    is_mech: torch.Tensor       # (N_all,) BoolTensor — True if mechanical DNF
    cons_idx_all: torch.Tensor  # (N_all,) LongTensor — constructor index for all rows

    # Counts
    n_drivers: int
    n_constructors: int
    n_seasons: int
    n_circuits: int
    n_races: int

    # Reverse maps
    driver_map: dict   # int_idx → driverId
    constructor_map: dict  # int_idx → constructorId
```

**Constructor remapping (apply BEFORE building integer indices):**
```python
CONSTRUCTOR_REMAP = {
    211: 10,   # Racing Point → Force India
    117: 10,   # Aston Martin → Force India
    214: 4,    # Alpine → Renault
    213: 5,    # AlphaTauri → Toro Rosso
    215: 5,    # Racing Bulls → Toro Rosso
    51: 15,    # Alfa Romeo → Sauber
}
```

**DNF classification — use `statusId`:**
```python
MECHANICAL_STATUS_IDS = frozenset({
    5,6,7,8,9,10,18,19,21,22,26,28,29,31,36,
    40,41,43,44,54,61,65,66,67,72,75,82,104,107,108,130,131
})
FINISHED_STATUS_IDS = frozenset({1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20})
# driver-fault = everything else (not finished, not mechanical)
```

**Race ordering:**
1. Include: Finished (ranked by `positionOrder`) then driver-fault DNFs (ranked by `positionOrder`)
2. Exclude: mechanical DNFs from the Plackett-Luce entries (they are in `is_mech` only)
3. `race_order` = 0 for winner within each race

**Pit normalisation:** `(x - mean) / (std + 1e-8)` per season.

**Assertions (run before returning):**
- `dataset.n_races == 286`
- `dataset.is_mech.float().mean()` between 0.10 and 0.25 (≈ 17% mechanical DNF rate over all original rows)
- `dataset.race_lengths.sum() == dataset.driver_idx.shape[0]` (ranking entries tally)
- No `torch.isnan` or `torch.isinf` in any float tensor
- Every dtype correct (LongTensor for idx, FloatTensor for covariates, BoolTensor for `is_mech`)

---

## Verification Commands

Run these after implementation. Report all output.

```bash
# 1. Quick sanity check
uv run python -c "
from models.pgm_backend.data_preparation import load_dataset
ds = load_dataset()
print('n_races:', ds.n_races)
print('n_drivers:', ds.n_drivers)
print('n_constructors:', ds.n_constructors)
print('is_mech mean (all rows):', ds.is_mech.float().mean().item())
print('race_lengths sum:', ds.race_lengths.sum().item(), '(should equal N_entries)')
print('driver_idx shape:', ds.driver_idx.shape)
print('race_order max per race OK:', ds.race_order.max().item())
print('cons_idx_all shape:', ds.cons_idx_all.shape)
"

# 2. Constructor merge assertion
uv run python -c "
from models.pgm_backend.data_preparation import load_dataset, CONSTRUCTOR_REMAP
import pandas as pd
df = pd.read_csv('data_preprocessing/f1_enriched.csv')
# Old IDs 211 and 117 must not appear in the dataset's constructor index
ds = load_dataset()
print('constructor_map:', ds.constructor_map)
assert 211 not in ds.constructor_map.values(), 'ID 211 leaked through remap'
assert 117 not in ds.constructor_map.values(), 'ID 117 leaked through remap'
print('Constructor remap: OK')
"
```

---

## Acceptance Criteria

- [ ] `ds.n_races == 286`
- [ ] `ds.is_mech.float().mean()` ≈ 0.17 (within 0.10–0.25, over all original rows)
- [ ] No NaN or inf in any tensor
- [ ] Constructor IDs 211 and 117 absent from `constructor_map.values()`
- [ ] `ds.race_lengths.sum() == ds.driver_idx.shape[0]` (ranking entries tally)
- [ ] `ds.cons_idx_all.shape[0] > ds.driver_idx.shape[0]` (N_all > N_entries, because mechanical DNFs excluded from ranking)
- [ ] Both verification commands run without error

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Actual values: n_races, n_drivers, n_constructors, n_seasons, n_circuits, N_entries, N_all
- Any deviations from the spec and why
- Anything T2 must know about the actual implementation

**Step 2 — Report back:**
1. The full output of both verification commands
2. Any deviations from the spec you made, and why
3. Whether all acceptance criteria pass (yes/no per criterion)

Then stop. Do not implement T2.
