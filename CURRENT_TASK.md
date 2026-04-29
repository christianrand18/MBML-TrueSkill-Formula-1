# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T3  
**Task name:** Model 1 Baseline — SVI (`model_baseline.py` + `inference.py`)

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## Context from T1, T2, and T2b

### Actual dataset counts (from T1)
- `n_drivers` = 77, `n_constructors` = 17, `n_seasons` = 14, `n_circuits` = 35, `n_races` = 286
- `N_entries` (ranking) = 5457, `N_all` (original rows) = 5980
- `wet` is per-race (shape `(286,)`) — NOT per-entry
- Constructor map has 17 entries (indices 0–16)

### Likelihood function (from T2)
```python
from models.pgm_backend.likelihood import plackett_luce_log_prob
# signature: (performances: (N_total,), race_lengths: (R,) LongTensor) -> scalar Tensor
# performances MUST be sorted in finishing order within each race block
# (this is guaranteed by data_preparation.py)
```

### Prior check (from T2b)
- sigma_s = 1.0, sigma_c = 1.0 confirmed plausible
- Prior-fastest driver win rate: 0.39 (within [0.20, 0.80])
- These sigma values are to be used in the model

### F1RankingDataset fields (from T1)
```python
from models.pgm_backend.data_preparation import F1RankingDataset, load_dataset

# Key fields for Model 1:
# ds.driver_idx  : (N_entries,) LongTensor  — driver integer index 0..76
# ds.cons_idx    : (N_entries,) LongTensor  — constructor integer index 0..16
# ds.race_lengths: (N_races,)  LongTensor   — entries per race
```

---

## What to Build

Two files:

**`models/pgm_backend/model_baseline.py`** — the Pyro model and guide  
**`models/pgm_backend/inference.py`** — SVI trainer and posterior extractor

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/model_baseline.py` | Create |
| `models/pgm_backend/inference.py` | Create |

Do not touch any other file.

---

## Full Implementation Spec

### 1. `model_baseline.py`

A single class `BaselineModel` with two methods: `model()` and `guide()`.

```python
import pyro
import pyro.distributions as dist
import torch

from models.pgm_backend.likelihood import plackett_luce_log_prob

SIGMA_S = 1.0
SIGMA_C = 1.0


class BaselineModel:
    def __init__(self, n_drivers: int, n_constructors: int):
        self.D = n_drivers
        self.K = n_constructors

    def model(self, driver_idx, cons_idx, race_lengths):
        """
        Args:
            driver_idx  : (N_entries,) LongTensor
            cons_idx    : (N_entries,) LongTensor
            race_lengths: (N_races,)  LongTensor
        """
        D, K = self.D, self.K

        # Latent driver skills: (D,) independent Normals
        s = pyro.sample(
            "s",
            dist.Normal(0.0, SIGMA_S).expand([D]).to_event(1),
        )

        # Sum-to-zero constructor reparameterisation
        c_raw = pyro.sample(
            "c_raw",
            dist.Normal(0.0, SIGMA_C).expand([K - 1]).to_event(1),
        )
        c = torch.cat([c_raw, -c_raw.sum(dim=0, keepdim=True)])  # (K,)

        # Performance: s_d + c_k for each entry
        p = s[driver_idx] + c[cons_idx]  # (N_entries,)

        # Plackett-Luce likelihood via pyro.factor
        log_prob = plackett_luce_log_prob(p, race_lengths)
        pyro.factor("race_obs", log_prob)

    def guide(self, driver_idx, cons_idx, race_lengths):
        """Mean-field variational guide."""
        D, K = self.D, self.K

        # Variational parameters for driver skills
        s_loc = pyro.param("s_loc", torch.zeros(D))
        s_scale = pyro.param(
            "s_scale",
            torch.ones(D),
            constraint=dist.constraints.positive,
        )
        pyro.sample("s", dist.Normal(s_loc, s_scale).to_event(1))

        # Variational parameters for sum-to-zero constructors
        c_loc = pyro.param("c_loc", torch.zeros(K - 1))
        c_scale = pyro.param(
            "c_scale",
            torch.ones(K - 1),
            constraint=dist.constraints.positive,
        )
        pyro.sample("c_raw", dist.Normal(c_loc, c_scale).to_event(1))
```

**Key constraints:**

- The guide samples `c_raw`, NEVER `c` directly. This ensures the sum-to-zero constraint holds exactly.
- `c_raw.sum(dim=0, keepdim=True)` — the `dim=0` argument is required for PyTorch 2.11+. Do NOT use `.sum(keepdim=True)` without `dim`.
- `s_scale` and `c_scale` must be constrained positive via `dist.constraints.positive`.

### 2. `inference.py`

Two functions:

```python
import pyro
import pyro.optim
from pyro.infer import SVI, Trace_ELBO

from models.pgm_backend.model_baseline import BaselineModel


def train_svi(model, dataset, n_steps=3000, lr=0.01, log_every=100):
    """
    Train the baseline model with SVI.

    Args:
        model:    BaselineModel instance
        dataset:  F1RankingDataset instance
        n_steps:  number of optimisation steps
        lr:       learning rate
        log_every: print ELBO every N steps

    Returns:
        losses: list of negative ELBO values (one per step)
    """
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(n_steps):
        loss = svi.step(
            driver_idx=dataset.driver_idx,
            cons_idx=dataset.cons_idx,
            race_lengths=dataset.race_lengths,
        )
        losses.append(loss)
        if step % log_every == 0 or step == n_steps - 1:
            print(f"Step {step:5d}  ELBO loss: {loss:.2f}")

    return losses


def extract_svi_posterior(model) -> dict[str, torch.Tensor]:
    """
    Extract posterior parameter estimates from the trained guide.

    Returns dict with keys:
        "s_loc"    : (D,) driver skill posterior means
        "s_scale"  : (D,) driver skill posterior stds
        "c_loc"    : (K,) constructor posterior means (including derived c_K)
        "c_scale"  : (K-1,) constructor posterior stds
    """
    s_loc = pyro.param("s_loc").detach().clone()
    s_scale = pyro.param("s_scale").detach().clone()
    c_loc_raw = pyro.param("c_loc").detach().clone()
    c_scale = pyro.param("c_scale").detach().clone()

    # Derive full c_loc including the constrained K-th entry
    c_loc = torch.cat([c_loc_raw, -c_loc_raw.sum(dim=0, keepdim=True)])

    return {
        "s_loc": s_loc,
        "s_scale": s_scale,
        "c_loc": c_loc,
        "c_scale": c_scale,
    }
```

---

## Verification Commands

```bash
# Full end-to-end: load data, build model, train SVI, extract posterior
uv run python -c "
import torch
from models.pgm_backend.data_preparation import load_dataset
from models.pgm_backend.model_baseline import BaselineModel
from models.pgm_backend.inference import train_svi, extract_svi_posterior

ds = load_dataset()
print(f'Loaded: {ds.n_drivers} drivers, {ds.n_constructors} constructors, {ds.n_races} races')

model = BaselineModel(ds.n_drivers, ds.n_constructors)
losses = train_svi(model, ds, n_steps=3000, lr=0.01, log_every=500)

post = extract_svi_posterior(model)

# Sum-to-zero check
c_sum = post['c_loc'].sum().item()
print(f'c.sum() = {c_sum:.6f} (should be ~0)')

# Hamilton check (driverId=1)
s_sorted = post['s_loc'].sort(descending=True)
print(f'Top 5 drivers by s_loc: {s_sorted.values[:5].tolist()}')

# Map the top driver indices back to driverIds
driver_ids = [ds.driver_map[i] for i in s_sorted.indices[:10].tolist()]
print(f'Top 10 driver IDs: {driver_ids}')

# ELBO trend
print(f'Initial loss: {losses[0]:.2f}, Final loss: {losses[-1]:.2f}, Decreasing: {losses[-1] < losses[0]}')
"
```

---

## Acceptance Criteria

- [ ] ELBO decreases over 3000 steps (final loss < initial loss)
- [ ] `c.sum()` ≈ 0 (tolerance 1e-4) at convergence
- [ ] Hamilton (`driverId=1`) in top 5 drivers by posterior mean `s_loc`, OR if driverId=1 is absent from the dataset, at least the top 10 driver IDs appear reasonable (e.g. include known champion driver IDs like 4=Vettel, 20=Alonso, 822=Verstappen, 830=Leclerc)
- [ ] SVI runs in < 5 minutes on CPU
- [ ] No NaN or inf in posterior parameters

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Report actual loss values (initial, final)
- Report actual `c.sum()` value
- Report actual top-5 driver IDs and s_loc values
- Note training time
- Note any deviations from the spec

**Step 2 — Append to `tasks/report_notes.md`** only if you made a non-obvious decision.

**Step 3 — Report back:**
1. Full output of the verification command
2. Whether all acceptance criteria pass
3. The actual top-5 driver IDs and their s_loc values

Then stop. Do not implement T4.
