# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T7  
**Task name:** Model 3 Full — Wet-Weather Skill + Pit Stops + Reliability

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## Context from T6

### Model 2 proven — checkpoint C in progress
- `model_extended.py` created with `ExtendedModel` class
- `train_svi()` generalized with `step_kwargs` — backward-compatible, no changes needed for Model 3
- `extract_svi_posterior_extended()` reconstructs `s_loc (T,D)` and `c_loc (T,K)` from innovations via `cumsum(0)`
- Pyro param store keys for temporal model: `s0_loc`, `s0_scale`, `s_innov_loc`, `s_innov_scale`, `c0_raw_loc`, `c0_raw_scale`, `c_innov_loc`, `c_innov_scale`, `e_circ_loc`, `e_circ_scale`, `beta_w_loc`, `beta_w_scale`
- Season indexing: season 0 = 2011, season 13 = 2024. Hybrid era = seasons 3–10
- Hamilton (`driverId=1`) → latent index `0`. Mercedes (`constructorId=131`) → latent index `8`
- Alonso (`driverId=4`) → latent index `4`. Webber (`driverId=13`) → latent index `13`
- `beta_w` posterior mean is close to zero (≈ −0.07), suggesting weather has weak marginal effect on PL ranking when treated as a global shift
- All 4 existing tests pass (`test_likelihood.py`, `test_prior_predictive.py`, `test_synthetic_recovery.py`)

### Dataset dimensions (unchanged from T1)
- `n_drivers = 77`, `n_constructors = 17`, `n_seasons = 14`, `n_circuits = 35`, `n_races = 286`
- `N_entries = 5457` (ranking entries), `N_all = 5980` (all original rows)
- `wet` shape: `(286,)` — index via `wet[race_idx]` to broadcast to entry level
- `pit_norm` shape: `(5457,)` — normalised pit-stop duration for ranking entries only
- `is_mech` shape: `(5980,)`, `cons_idx_all` shape: `(5980,)` — for Model 3 reliability term

---

## What to Build

Create `models/pgm_backend/model_full.py` with the FullModel class, add `season_idx_all` to `F1RankingDataset`, and add `extract_svi_posterior_full()` to `inference.py`.

No other file should be modified.

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/model_full.py` | Create |
| `models/pgm_backend/data_preparation.py` | Modify (add `season_idx_all` field to `F1RankingDataset`) |
| `models/pgm_backend/inference.py` | Modify (add `extract_svi_posterior_full`) |

Do not touch any other file.

---

## Full Implementation Spec

### File 1: `model_full.py`

Create a new file `models/pgm_backend/model_full.py`:

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
SIGMA_DELTA = 0.5


class FullModel:
    def __init__(self, n_drivers: int, n_constructors: int, n_seasons: int, n_circuits: int):
        self.D = n_drivers
        self.K = n_constructors
        self.T = n_seasons
        self.C = n_circuits

    def model(
        self,
        driver_idx,
        cons_idx,
        season_idx,
        circuit_idx,
        race_idx,
        wet,
        race_lengths,
        pit_norm,
        is_mech,
        cons_idx_all,
        season_idx_all,
    ):
        D, K, T, C = self.D, self.K, self.T, self.C

        # ---- Driver AR(1) skills ----
        s0 = pyro.sample("s0", dist.Normal(0.0, SIGMA_S).expand([D]).to_event(1))
        s_innov = pyro.sample(
            "s_innov", dist.Normal(0.0, GAMMA_S).expand([T - 1, D]).to_event(2)
        )
        s = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + s_innov.cumsum(0)], dim=0)  # (T, D)

        # ---- Constructor AR(1) skills (sum-to-zero per season) ----
        c0_raw = pyro.sample("c0_raw", dist.Normal(0.0, SIGMA_C).expand([K - 1]).to_event(1))
        c_innov = pyro.sample(
            "c_innov", dist.Normal(0.0, GAMMA_C).expand([T - 1, K - 1]).to_event(2)
        )
        c_raw = torch.cat([c0_raw.unsqueeze(0), c0_raw.unsqueeze(0) + c_innov.cumsum(0)], dim=0)  # (T, K-1)
        c = torch.cat([c_raw, -c_raw.sum(dim=1, keepdim=True)], dim=1)  # (T, K)

        # ---- Circuit effects ----
        e_circ = pyro.sample("e_circ", dist.Normal(0.0, SIGMA_E).expand([C]).to_event(1))

        # ---- Global weather coefficient ----
        beta_w = pyro.sample("beta_w", dist.Normal(0.0, 0.5))

        # ---- Driver wet-weather skill modifier ----
        delta_d = pyro.sample("delta_d", dist.Normal(0.0, SIGMA_DELTA).expand([D]).to_event(1))

        # ---- Pit-stop coefficient ----
        beta_pi = pyro.sample("beta_pi", dist.Normal(0.0, 0.5))

        # ---- Reliability intercept ----
        alpha_rel = pyro.sample("alpha_rel", dist.Normal(0.0, 1.0))

        # ---- Performance (ranking entries only) ----
        p = (
            s[season_idx, driver_idx]
            + c[season_idx, cons_idx]
            + e_circ[circuit_idx]
            + beta_w * wet[race_idx]
            + delta_d[driver_idx] * wet[race_idx]
            + beta_pi * pit_norm
        )

        log_prob = plackett_luce_log_prob(p, race_lengths)
        pyro.factor("race_obs", log_prob)

        # ---- Mechanical DNF reliability term (all rows) ----
        mech_prob = torch.sigmoid(-alpha_rel - c[season_idx_all, cons_idx_all])
        pyro.factor("reliability", dist.Bernoulli(mech_prob).log_prob(is_mech.float()))

    def guide(
        self,
        driver_idx,
        cons_idx,
        season_idx,
        circuit_idx,
        race_idx,
        wet,
        race_lengths,
        pit_norm,
        is_mech,
        cons_idx_all,
        season_idx_all,
    ):
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

        delta_d_loc = pyro.param("delta_d_loc", torch.zeros(D))
        delta_d_scale = pyro.param(
            "delta_d_scale", torch.ones(D), constraint=dist.constraints.positive
        )
        pyro.sample("delta_d", dist.Normal(delta_d_loc, delta_d_scale).to_event(1))

        beta_pi_loc = pyro.param("beta_pi_loc", torch.tensor(0.0))
        beta_pi_scale = pyro.param(
            "beta_pi_scale", torch.tensor(1.0), constraint=dist.constraints.positive
        )
        pyro.sample("beta_pi", dist.Normal(beta_pi_loc, beta_pi_scale))

        alpha_rel_loc = pyro.param("alpha_rel_loc", torch.tensor(0.0))
        alpha_rel_scale = pyro.param(
            "alpha_rel_scale", torch.tensor(1.0), constraint=dist.constraints.positive
        )
        pyro.sample("alpha_rel", dist.Normal(alpha_rel_loc, alpha_rel_scale))
```

**Critical implementation notes:**
- `delta_d[driver_idx] * wet[race_idx]` is a multiplicative **interaction** — do NOT add them separately.
- The reliability term uses `c[season_idx_all, cons_idx_all]` because constructor skills are temporal. `is_mech` has shape `(N_all,)` and includes all original rows (mechanical DNFs + everyone else).
- `mech_prob = sigmoid(-alpha_rel - c[...])` — higher constructor skill → lower mechanical failure probability.
- Sum-to-zero on constructors is applied **per season** (`dim=1` on `c_raw` which has shape `(T, K-1)`), exactly as in Model 2.
- AR(1) uses `cumsum(0)` on innovations — no recursive `pyro.sample` loop.

---

### File 2: `data_preparation.py` modifications

Add `season_idx_all` to the `F1RankingDataset` dataclass and populate it in `load_dataset`.

**Step 1 — Add the field to the dataclass:**

Insert this line after `cons_idx_all` in the `F1RankingDataset` dataclass:

```python
    season_idx_all: torch.Tensor     # (N_all,) LongTensor — season index for all original rows
```

**Step 2 — Build the tensor in `load_dataset`:**

After the line that builds `season_idx_tensor` (inside the sorted-ranking block), add this block immediately before the "# ---- 7. Pit normalisation" comment:

```python
    season_idx_all_tensor = torch.tensor(
        df["year"].map(season_lookup).values, dtype=torch.long
    )
```

**Step 3 — Pass it to the constructor:**

Add `season_idx_all=season_idx_all_tensor,` inside the `F1RankingDataset(...)` call.

**Step 4 — Add dtype assertion:**

Add this assertion in the assertion block at the bottom:

```python
    assert ds.season_idx_all.dtype == torch.long, "season_idx_all must be LongTensor"
```

---

### File 3: `inference.py` modifications

Append this function to the end of `models/pgm_backend/inference.py`:

```python
def extract_svi_posterior_full(model) -> dict[str, torch.Tensor]:
    """Extract reconstructed temporal posterior means for Model 3."""
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
    delta_d_loc = pyro.param("delta_d_loc").detach().clone()
    beta_pi_loc = pyro.param("beta_pi_loc").detach().clone()
    alpha_rel_loc = pyro.param("alpha_rel_loc").detach().clone()

    return {
        "s_loc": s_loc,               # (T, D)
        "c_loc": c_loc,               # (T, K)
        "e_circ_loc": e_circ_loc,     # (C,)
        "beta_w_loc": beta_w_loc,     # scalar
        "delta_d_loc": delta_d_loc,   # (D,)
        "beta_pi_loc": beta_pi_loc,   # scalar
        "alpha_rel_loc": alpha_rel_loc,  # scalar
    }
```

---

## Verification Commands

Run this script after implementing:

```bash
uv run python -c "
import time
import torch
import pyro
from models.pgm_backend.data_preparation import load_dataset
from models.pgm_backend.model_full import FullModel
from models.pgm_backend.inference import train_svi, extract_svi_posterior_full

dataset = load_dataset()
model = FullModel(
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
    'pit_norm': dataset.pit_norm,
    'is_mech': dataset.is_mech,
    'cons_idx_all': dataset.cons_idx_all,
    'season_idx_all': dataset.season_idx_all,
}

start = time.time()
losses = train_svi(model, dataset, n_steps=5000, lr=0.01, log_every=500, step_kwargs=step_kwargs)
elapsed = time.time() - start

posterior = extract_svi_posterior_full(model)

assert len(losses) == 5000
assert losses[-1] < losses[0], f'ELBO did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}'
c_sum = posterior['c_loc'].sum(dim=1).abs().max()
assert c_sum < 1e-4, f'sum-to-zero violated: max abs season sum = {c_sum:.6f}'

# delta_d non-trivial (not collapsed to prior)
delta_d_std = pyro.param('delta_d_scale').detach().mean().item()
assert delta_d_std > 0.1, f'delta_d collapsed to prior: mean scale = {delta_d_std:.4f}'

# beta_pi negative (faster pit stops -> better performance)
beta_pi_mean = posterior['beta_pi_loc'].item()
assert beta_pi_mean < 0, f'beta_pi not negative: {beta_pi_mean:.4f}'

# Known wet-weather specialist in top-5 by delta_d
delta_d_sorted = torch.argsort(posterior['delta_d_loc'], descending=True)
top5 = delta_d_sorted[:5].tolist()
assert 4 in top5 or 13 in top5, f'No known wet-weather specialist in top-5 delta_d: {top5}'

print(f'\nRuntime: {elapsed/60:.1f} min')
print(f'ELBO: {losses[0]:.2f} -> {losses[-1]:.2f}')
print(f'c sum-to-zero max abs: {c_sum:.6f}')
print(f'beta_pi: {beta_pi_mean:.4f}')
print(f'delta_d mean scale: {delta_d_std:.4f}')
print(f'Top 5 delta_d indices: {top5}')
print(f'Alonso (idx 4) delta_d: {posterior[\"delta_d_loc\"][4].item():.4f}')
print(f'Webber (idx 13) delta_d: {posterior[\"delta_d_loc\"][13].item():.4f}')
"
```

---

## Acceptance Criteria

- [ ] `model_full.py` created with `FullModel` class matching the spec above
- [ ] `data_preparation.py` modified: `F1RankingDataset` has `season_idx_all` field, populated and asserted
- [ ] `inference.py` modified: `extract_svi_posterior_full` added
- [ ] ELBO decreases over 5000 steps (final < initial)
- [ ] `c_loc.sum(dim=1).abs().max() < 1e-4` (sum-to-zero holds for every season)
- [ ] `delta_d` posterior is non-trivial (`delta_d_scale.mean() > 0.1`)
- [ ] `beta_pi` posterior mean is negative
- [ ] At least one known wet-weather specialist (Alonso=index 4, Webber=index 13) ranks in top-5 by `delta_d_loc`
- [ ] `alpha_rel` posterior mean is a finite real number
- [ ] SVI runtime < 20 minutes on CPU
- [ ] No NaN or inf in any posterior parameter
- [ ] `pytest models/pgm_backend/tests/ -v` still passes (Models 1 & 2 tests unaffected)

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Report actual ELBO initial/final values
- Report runtime in minutes
- Report max `c_loc` season-sum absolute value
- Report `beta_pi`, `delta_d_scale.mean()`, `alpha_rel` posterior means
- Report top-5 `delta_d` indices and whether Alonso/Webber appear
- Note any deviations from the spec

**Step 2 — Append to `tasks/report_notes.md`** only if you made a non-obvious decision (e.g., had to change hyperparameters, reliability term differed from spec, etc.).

**Step 3 — Report back:**
1. Full output of the verification script (ELBO curve, runtime, parameter summaries)
2. Whether all acceptance criteria pass
3. Any deviations or concerns

Then stop. Do not implement T8.
