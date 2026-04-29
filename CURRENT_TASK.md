# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T6  
**Task name:** Model 2 Extended — AR(1) Temporal Skills, Circuit Effects, Weather

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## Context from T5

### Inference proven — checkpoint B cleared
- `test_likelihood.py` passes (hand-check and non-positivity)
- `test_synthetic_recovery.py` passes after centering driver skills for PL shift-invariance
- Constructors recovered to <0.15 error; ranking signs match perfectly
- `train_svi()` and `extract_svi_posterior()` work correctly for Model 1

### Dataset dimensions (unchanged from T1)
- `n_drivers = 77`, `n_constructors = 17`, `n_seasons = 14`, `n_circuits = 35`, `n_races = 286`
- `N_entries = 5457`
- `wet` shape: `(286,)` — index via `wet[race_idx]` to broadcast to entry level
- `season_idx`, `circuit_idx`, `race_idx` all have shape `(N_entries,)`

### Key indices for sanity checks
- Hamilton (`driverId=1`) → `driver_map[0]`, so index `0` in latent tensors
- Mercedes (`constructorId=131`) → `constructor_map[8]`, so index `8` in latent tensors
- Season indices map linearly to years: index 0 = 2011, …, index 13 = 2024
  - Hybrid era (2014–2021) corresponds to season indices 3–10

---

## What to Build

Create `models/pgm_backend/model_extended.py` with the ExtendedModel class, and generalise `train_svi` in `inference.py` so it can pass extra tensors to Model 2.

No other file should be modified.

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/model_extended.py` | Create |
| `models/pgm_backend/inference.py` | Modify (generalise `train_svi`, add `extract_svi_posterior_extended`) |

Do not touch any other file.

---

## Full Implementation Spec

### File 1: `model_extended.py`

Create a new file `models/pgm_backend/model_extended.py`:

```python
import pyro
import pyro.distributions as dist
import torch

from models.pgm_backend.likelihood import plackett_luce_log_prob

SIGMA_S = 1.0
SIGMA_C = 1.0
GAMMA_S = 0.3
GAMMA_C = 0.5
SIGMA_E = 0.5


class ExtendedModel:
    def __init__(self, n_drivers: int, n_constructors: int, n_seasons: int, n_circuits: int):
        self.D = n_drivers
        self.K = n_constructors
        self.T = n_seasons
        self.C = n_circuits

    def model(self, driver_idx, cons_idx, season_idx, circuit_idx, race_idx, wet, race_lengths):
        D, K, T, C = self.D, self.K, self.T, self.C

        # ---- Driver AR(1) skills ----
        s0 = pyro.sample(
            "s0",
            dist.Normal(0.0, SIGMA_S).expand([D]).to_event(1),
        )
        s_innov = pyro.sample(
            "s_innov",
            dist.Normal(0.0, GAMMA_S).expand([T - 1, D]).to_event(2),
        )
        s = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + s_innov.cumsum(0)], dim=0)  # (T, D)

        # ---- Constructor AR(1) skills (sum-to-zero per season) ----
        c0_raw = pyro.sample(
            "c0_raw",
            dist.Normal(0.0, SIGMA_C).expand([K - 1]).to_event(1),
        )
        c_innov = pyro.sample(
            "c_innov",
            dist.Normal(0.0, GAMMA_C).expand([T - 1, K - 1]).to_event(2),
        )
        c_raw = torch.cat([c0_raw.unsqueeze(0), c0_raw.unsqueeze(0) + c_innov.cumsum(0)], dim=0)  # (T, K-1)
        c = torch.cat([c_raw, -c_raw.sum(dim=1, keepdim=True)], dim=1)  # (T, K)

        # ---- Circuit effects ----
        e_circ = pyro.sample(
            "e_circ",
            dist.Normal(0.0, SIGMA_E).expand([C]).to_event(1),
        )

        # ---- Global weather coefficient ----
        beta_w = pyro.sample("beta_w", dist.Normal(0.0, 0.5))

        # ---- Performance ----
        p = (
            s[season_idx, driver_idx]
            + c[season_idx, cons_idx]
            + e_circ[circuit_idx]
            + beta_w * wet[race_idx]
        )

        log_prob = plackett_luce_log_prob(p, race_lengths)
        pyro.factor("race_obs", log_prob)

    def guide(self, driver_idx, cons_idx, season_idx, circuit_idx, race_idx, wet, race_lengths):
        D, K, T, C = self.D, self.K, self.T, self.C

        s0_loc = pyro.param("s0_loc", torch.zeros(D))
        s0_scale = pyro.param("s0_scale", torch.ones(D), constraint=dist.constraints.positive)
        pyro.sample("s0", dist.Normal(s0_loc, s0_scale).to_event(1))

        s_innov_loc = pyro.param("s_innov_loc", torch.zeros(T - 1, D))
        s_innov_scale = pyro.param(
            "s_innov_scale", torch.ones(T - 1, D), constraint=dist.constraints.positive
        )
        pyro.sample("s_innov", dist.Normal(s_innov_loc, s_innov_scale).to_event(2))

        c0_raw_loc = pyro.param("c0_raw_loc", torch.zeros(K - 1))
        c0_raw_scale = pyro.param(
            "c0_raw_scale", torch.ones(K - 1), constraint=dist.constraints.positive
        )
        pyro.sample("c0_raw", dist.Normal(c0_raw_loc, c0_raw_scale).to_event(1))

        c_innov_loc = pyro.param("c_innov_loc", torch.zeros(T - 1, K - 1))
        c_innov_scale = pyro.param(
            "c_innov_scale", torch.ones(T - 1, K - 1), constraint=dist.constraints.positive
        )
        pyro.sample("c_innov", dist.Normal(c_innov_loc, c_innov_scale).to_event(2))

        e_circ_loc = pyro.param("e_circ_loc", torch.zeros(C))
        e_circ_scale = pyro.param(
            "e_circ_scale", torch.ones(C), constraint=dist.constraints.positive
        )
        pyro.sample("e_circ", dist.Normal(e_circ_loc, e_circ_scale).to_event(1))

        beta_w_loc = pyro.param("beta_w_loc", torch.tensor(0.0))
        beta_w_scale = pyro.param(
            "beta_w_scale", torch.tensor(1.0), constraint=dist.constraints.positive
        )
        pyro.sample("beta_w", dist.Normal(beta_w_loc, beta_w_scale))
```

**Critical implementation notes:**
- The AR(1) MUST use `cumsum(0)` on innovations — no recursive `pyro.sample` loop (see CLAUDE.md Architecture Constraint #4).
- Sum-to-zero on constructors is applied **per season** (`dim=1` on `c_raw` which has shape `(T, K-1)`).
- `wet[race_idx]` broadcasts the per-race wet indicator to each entry.
- All `to_event` dimensions must match the sample shapes exactly (see code above).

---

### File 2: `inference.py` modifications

Make two changes to `models/pgm_backend/inference.py`:

**Change A — Generalise `train_svi`**

Replace the existing `train_svi` function with this backward-compatible version:

```python
def train_svi(model, dataset, n_steps=3000, lr=0.01, log_every=100, step_kwargs=None):
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

    if step_kwargs is None:
        step_kwargs = {
            "driver_idx": dataset.driver_idx,
            "cons_idx": dataset.cons_idx,
            "race_lengths": dataset.race_lengths,
        }

    losses = []
    for step in range(n_steps):
        loss = svi.step(**step_kwargs)
        losses.append(loss)
        if step % log_every == 0 or step == n_steps - 1:
            print(f"Step {step:5d}  ELBO loss: {loss:.2f}")

    return losses
```

When `step_kwargs` is not provided, behaviour is identical to the old Model-1-only path.

**Change B — Add `extract_svi_posterior_extended`**

Append this function to the end of `inference.py`:

```python
def extract_svi_posterior_extended(model) -> dict[str, torch.Tensor]:
    """Extract reconstructed temporal posterior means for Model 2."""
    T, D = model.T, model.D
    K = model.K

    s0_loc = pyro.param("s0_loc").detach().clone()
    s_innov_loc = pyro.param("s_innov_loc").detach().clone()
    s_loc = torch.cat([s0_loc.unsqueeze(0), s0_loc.unsqueeze(0) + s_innov_loc.cumsum(0)], dim=0)

    c0_raw_loc = pyro.param("c0_raw_loc").detach().clone()
    c_innov_loc = pyro.param("c_innov_loc").detach().clone()
    c_raw_loc = torch.cat([c0_raw_loc.unsqueeze(0), c0_raw_loc.unsqueeze(0) + c_innov_loc.cumsum(0)], dim=0)
    c_loc = torch.cat([c_raw_loc, -c_raw_loc.sum(dim=1, keepdim=True)], dim=1)

    e_circ_loc = pyro.param("e_circ_loc").detach().clone()
    beta_w_loc = pyro.param("beta_w_loc").detach().clone()

    return {
        "s_loc": s_loc,          # (T, D)
        "c_loc": c_loc,          # (T, K)
        "e_circ_loc": e_circ_loc,
        "beta_w_loc": beta_w_loc,
    }
```

---

## Verification Commands

Run this script after implementing:

```bash
uv run python -c "
import time
import torch
from models.pgm_backend.data_preparation import load_dataset
from models.pgm_backend.model_extended import ExtendedModel
from models.pgm_backend.inference import train_svi, extract_svi_posterior_extended

dataset = load_dataset()
model = ExtendedModel(
    n_drivers=dataset.n_drivers,
    n_constructors=dataset.n_constructors,
    n_seasons=dataset.n_seasons,
    n_circuits=dataset.n_circuits,
)

step_kwargs = {
    'driver_idx': dataset.driver_idx,
    'cons_idx': dataset.cons_idx,
    'season_idx': dataset.season_idx,
    'circuit_idx': dataset.circuit_idx,
    'race_idx': dataset.race_idx,
    'wet': dataset.wet,
    'race_lengths': dataset.race_lengths,
}

start = time.time()
losses = train_svi(model, dataset, n_steps=5000, lr=0.01, log_every=500, step_kwargs=step_kwargs)
elapsed = time.time() - start

posterior = extract_svi_posterior_extended(model)

assert len(losses) == 5000
assert losses[-1] < losses[0], f'ELBO did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}'
c_sum = posterior['c_loc'].sum(dim=1).abs().max()
assert c_sum < 1e-4, f'sum-to-zero violated: max abs season sum = {c_sum:.6f}'

print(f'\\nRuntime: {elapsed/60:.1f} min')
print(f'ELBO: {losses[0]:.2f} -> {losses[-1]:.2f}')
print(f'c sum-to-zero max abs: {c_sum:.6f}')
print(f'beta_w: {posterior[\"beta_w_loc\"].item():.4f}')
print(f'Hamilton trajectory (idx 0): {posterior[\"s_loc\"][:, 0].tolist()}')
print(f'Mercedes trajectory (idx 8): {posterior[\"c_loc\"][:, 8].tolist()}')
"
```

---

## Acceptance Criteria

- [ ] `model_extended.py` created with `ExtendedModel` class matching the spec above
- [ ] `inference.py` modified: `train_svi` accepts optional `step_kwargs`; `extract_svi_posterior_extended` added
- [ ] ELBO decreases over 5000 steps (final < initial)
- [ ] `c_loc.sum(dim=1).abs().max() < 1e-4` (sum-to-zero holds for every season)
- [ ] Hamilton (`driverId=1`, latent index 0) trajectory printed; visually shows higher skill during 2014–2021 hybrid era than 2011–2013 or 2022–2024
- [ ] Mercedes (`constructorId=131`, latent index 8) trajectory printed; visually peaks during 2014–2021 hybrid era
- [ ] `beta_w` posterior mean is a finite real number (not NaN/inf)
- [ ] SVI runtime < 15 minutes on CPU
- [ ] No NaN or inf in any posterior parameter
- [ ] `pytest models/pgm_backend/tests/ -v` still passes (Model 1 tests unaffected)

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Report actual ELBO initial/final values
- Report runtime in minutes
- Report max `c_loc` season-sum absolute value
- Report `beta_w` posterior mean
- Report whether Hamilton/Mercedes trajectories look sane
- Note any deviations from the spec

**Step 2 — Append to `tasks/report_notes.md`** only if you made a non-obvious decision (e.g., had to change hyperparameters, AR(1) pattern differed from spec, etc.).

**Step 3 — Report back:**
1. Full output of the verification script (ELBO curve, runtime, trajectories)
2. Whether all acceptance criteria pass
3. Any deviations or concerns

Then stop. Do not implement T7.
