# Current Task

**Status:** READY FOR HANDOFF  
**Task ID:** T8  
**Task name:** Posterior Extraction + Orchestrator + Plots

---

## Context

Read `CLAUDE.md` for project overview, tech stack, architecture constraints, and your role.  
This file contains everything you need to implement the current task. Do not look elsewhere.

---

## Context from T7

### All three models built and training successfully
- `model_baseline.py` — static skills, SVI + NUTS
- `model_extended.py` — AR(1) temporal skills, circuit effects, global weather
- `model_full.py` — adds `delta_d` wet-weather interaction, `beta_pi` pit-stop coefficient, `alpha_rel` reliability Bernoulli term

### Key posterior parameter store keys (after training each model)
**Model 1 (Baseline):**
- `s_loc` (77,), `s_scale` (77,), `c_loc` (16,), `c_scale` (16,)
- `extract_svi_posterior()` reconstructs full `c_loc` (17,) via `cat([c_loc_raw, -c_loc_raw.sum()])`

**Model 2 (Extended):**
- `s0_loc` (77,), `s0_scale` (77,), `s_innov_loc` (13,77), `s_innov_scale` (13,77)
- `c0_raw_loc` (16,), `c0_raw_scale` (16,), `c_innov_loc` (13,16), `c_innov_scale` (13,16)
- `e_circ_loc` (35,), `e_circ_scale` (35,), `beta_w_loc` scalar, `beta_w_scale` scalar
- `extract_svi_posterior_extended()` reconstructs `s_loc` (14,77) and `c_loc` (14,17) via cumsum

**Model 3 (Full):**
- All Model 2 keys, PLUS:
- `delta_d_loc` (77,), `delta_d_scale` (77,), `beta_pi_loc` scalar, `beta_pi_scale` scalar
- `alpha_rel_loc` scalar, `alpha_rel_scale` scalar
- `extract_svi_posterior_full()` reconstructs same as extended plus `delta_d_loc`, `beta_pi_loc`, `alpha_rel_loc`

### Dataset dimensions (unchanged from T1)
- `n_drivers = 77`, `n_constructors = 17`, `n_seasons = 14`, `n_circuits = 35`, `n_races = 286`
- `N_entries = 5457` (ranking entries), `N_all = 5980` (all original rows)
- `wet` shape: `(286,)` — index via `wet[race_idx]` to broadcast to entry level
- `pit_norm` shape: `(5457,)` — normalised pit-stop duration for ranking entries only
- `is_mech` shape: `(5980,)`, `cons_idx_all` shape: `(5980,)`, `season_idx_all` shape: `(5980,)`

### Important driver/constructor index mappings
- Hamilton (`driverId=1`) → latent index `0`
- Verstappen (`driverId=830`) → latent index `?` (look up in `dataset.driver_map`)
- Alonso (`driverId=4`) → latent index `4`
- Mercedes (`constructorId=131`) → latent index `8`
- Red Bull (`constructorId=9`) → latent index `?`
- Ferrari (`constructorId=6`) → latent index `?`

### T7 empirical findings to respect in plots
- `beta_pi` posterior mean is **positive** (~+0.25), NOT negative. Do NOT hard-code a negative expectation in plots or captions.
- Top wet-weather drivers by `delta_d` are [44, 2, 60, 15, 0] — Alonso (idx 4) is 6th, Webber (idx 13) is below average. Plot what the model actually found.
- `alpha_rel` ≈ 2.04 (finite, real)
- `beta_w` ≈ 0.03 (close to zero)

### Existing outputs
- `outputs/pgm_model/nuts_vs_svi_comparison.csv` already exists from T4 (94 rows, 8 columns)

---

## What to Build

Create two new files:
1. `models/pgm_backend/posterior.py` — unified posterior extractor that converts param-store tensors into tidy DataFrames
2. `models/pgm_backend/run_pgm.py` — orchestrator that trains all 3 models, exports 4 CSVs, and generates 10 plots

No other file should be modified.

---

## Target Files

| File | Action |
|---|---|
| `models/pgm_backend/posterior.py` | Create |
| `models/pgm_backend/run_pgm.py` | Create |

Do not touch any other file.

---

## Full Implementation Spec

### File 1: `posterior.py`

Create `models/pgm_backend/posterior.py` with a single public function:

```python
def extract_posterior(model_name: str, dataset) -> pd.DataFrame:
    """Return a tidy DataFrame of posterior means and stds for a trained model.

    Parameters
    ----------
    model_name : {"baseline", "extended", "full"}
        Which model's param store to read.
    dataset : F1RankingDataset
        Needed for driver_map, constructor_map, n_seasons, etc.

    Returns
    -------
    pd.DataFrame with columns:
        - entity_type : str  — "driver" or "constructor" or "global"
        - entity_id : int    — driverId or constructorId (or -1 for global params)
        - entity_name : str  — empty string (placeholder; keep empty)
        - season : int or str — 0-13 for temporal models, "all" for static
        - mu : float         — posterior mean
        - sigma : float      — posterior std (scale param from guide)
    """
```

**Model 1 (baseline) handling:**
- Read `s_loc`, `s_scale` from param store (shape `(D,)` each)
- Read `c_loc_raw`, `c_scale` (shape `(K-1,)`), reconstruct full `c_loc` via `cat([c_loc_raw, -c_loc_raw.sum(keepdim=True)])`
- For constructors, the full `c_scale` should be the same extension (append the last constructor's scale = same as the K-1 raw scale — or just set it to the mean of raw scales; either is fine for plotting)
- Emit one row per driver with `season="all"`
- Emit one row per constructor with `season="all"`

**Model 2 (extended) handling:**
- Use `extract_svi_posterior_extended()` logic (or call it) to reconstruct `s_loc` (14,77) and `c_loc` (14,17)
- For scales: `s0_scale` is the initial scale; `s_innov_scale` gives per-season innovation scales. For simplicity, use `s0_scale` as the scale for season 0, and for seasons 1+ use the cumulative RSS of innovations: `sqrt(s0_scale**2 + cumsum(s_innov_scale**2, dim=0))`. Or just use the raw `s_innov_scale` as a per-season scale approximation. **Simplicity first** — using `s0_scale` for season 0 and `s_innov_scale[t-1]` for season t is acceptable.
- Emit one row per (driver, season) and per (constructor, season). Season values 0–13.
- Also emit `e_circ` as rows with `entity_type="circuit"`, `entity_name=""`, `season="all"`
- Emit `beta_w` as a single row with `entity_type="global"`, `entity_id=-1`, `season="all"`

**Model 3 (full) handling:**
- Same as extended for `s`, `c`, `e_circ`, `beta_w`
- Add `delta_d` rows: `entity_type="driver"`, one row per driver, `season="all"`
- Add `beta_pi` row: `entity_type="global"`, `entity_id=-1`, `season="all"`
- Add `alpha_rel` row: `entity_type="global"`, `entity_id=-1`, `season="all"`

**Important:** Use `dataset.driver_map` and `dataset.constructor_map` to map integer indices back to original IDs. Do NOT hard-code IDs.

---

### File 2: `run_pgm.py`

Create `models/pgm_backend/run_pgm.py`. It must be runnable as:
```bash
uv run python -m models.pgm_backend.run_pgm
```

The module should:
1. Import everything needed
2. Have a `main()` function
3. End with:
```python
if __name__ == "__main__":
    main()
```

**`main()` flow:**

```python
def main():
    # 1. Load dataset
    dataset = load_dataset()

    # 2. Train Model 1 (SVI)
    print("=== Model 1: Baseline SVI ===")
    model1 = BaselineModel(n_drivers=dataset.n_drivers, n_constructors=dataset.n_constructors)
    losses1 = train_svi(model1, dataset, n_steps=3000, lr=0.01, log_every=500)
    posterior1 = extract_svi_posterior(model1)
    df1 = extract_posterior("baseline", dataset)
    df1.to_csv("outputs/pgm_model/baseline_posterior.csv", index=False)

    # 3. Train Model 1 (NUTS) — OPTIONAL but recommended
    # Only if runtime is acceptable. If you skip NUTS, still generate the scatter
    # from the existing nuts_vs_svi_comparison.csv.
    print("=== Model 1: NUTS ===")
    mcmc = run_nuts(model1, dataset, num_warmup=500, num_samples=500)
    compare_svi_nuts(posterior1, mcmc, dataset)

    # 4. Train Model 2
    print("=== Model 2: Extended SVI ===")
    model2 = ExtendedModel(
        n_drivers=dataset.n_drivers,
        n_constructors=dataset.n_constructors,
        n_seasons=dataset.n_seasons,
        n_circuits=dataset.n_circuits,
    )
    step_kwargs2 = {
        "driver_idx": dataset.driver_idx,
        "cons_idx": dataset.cons_idx,
        "season_idx": dataset.season_idx,
        "circuit_idx": dataset.circuit_idx,
        "race_idx": dataset.race_idx,
        "wet": dataset.wet,
        "race_lengths": dataset.race_lengths,
    }
    losses2 = train_svi(model2, dataset, n_steps=5000, lr=0.01, log_every=500, step_kwargs=step_kwargs2)
    posterior2 = extract_svi_posterior_extended(model2)
    df2 = extract_posterior("extended", dataset)
    df2.to_csv("outputs/pgm_model/extended_posterior.csv", index=False)

    # 5. Train Model 3
    print("=== Model 3: Full SVI ===")
    model3 = FullModel(
        n_drivers=dataset.n_drivers,
        n_constructors=dataset.n_constructors,
        n_seasons=dataset.n_seasons,
        n_circuits=dataset.n_circuits,
    )
    step_kwargs3 = {
        "driver_idx": dataset.driver_idx,
        "cons_idx": dataset.cons_idx,
        "season_idx": dataset.season_idx,
        "circuit_idx": dataset.circuit_idx,
        "race_idx": dataset.race_idx,
        "wet": dataset.wet,
        "race_lengths": dataset.race_lengths,
        "pit_norm": dataset.pit_norm,
        "is_mech": dataset.is_mech,
        "cons_idx_all": dataset.cons_idx_all,
        "season_idx_all": dataset.season_idx_all,
    }
    losses3 = train_svi(model3, dataset, n_steps=5000, lr=0.01, log_every=500, step_kwargs=step_kwargs3)
    posterior3 = extract_svi_posterior_full(model3)
    df3 = extract_posterior("full", dataset)
    df3.to_csv("outputs/pgm_model/full_posterior.csv", index=False)

    # 6. Generate all plots
    os.makedirs("outputs/pgm_model/plots", exist_ok=True)
    _plot_elbo_curves(losses1, losses2, losses3)
    _plot_temporal_drivers(posterior2, dataset)
    _plot_temporal_constructors(posterior2, dataset)
    _plot_wet_weather_specialists(posterior3, dataset)
    _plot_beta_pi(posterior3)
    _plot_cross_model_ranking(posterior1, posterior2, posterior3, dataset)
    _plot_uncertainty_vs_races(posterior1, dataset)
    _plot_prior_predictive()
    _plot_svi_vs_nuts()
    _plot_synthetic_recovery()

    # 7. Print summary tables
    print("\n=== Top 10 Drivers (Model 1) ===")
    ...
```

**Plot specifications (10 plots total):**

All plots save to `outputs/pgm_model/plots/` with dpi ≥ 150, figure size ~10×6 inches.

| # | Filename | Description | Data source |
|---|----------|-------------|-------------|
| 1 | `prior_predictive_win_rate.png` | Histogram of prior-fastest-driver win rates across 100 draws. Annotate 20–80% acceptance band with vertical dashed lines. | Re-run the prior predictive sampling logic from `test_prior_predictive.py` (or import and call it). |
| 2 | `svi_vs_nuts_scatter.png` | Scatter: SVI mean vs NUTS mean per driver. x=y reference line in black. Error bars = ±1 NUTS std. | `outputs/pgm_model/nuts_vs_svi_comparison.csv` (already exists) or regenerate from `compare_svi_nuts()`. |
| 3 | `synthetic_recovery.png` | Scatter: true value vs posterior mean for all 5 synthetic drivers + 3 constructors. x=y reference line. Shade ±0.8 tolerance band. | Re-run `test_baseline_recovery` logic capturing means, or read from an ad-hoc run in `run_pgm.py`. |
| 4 | `elbo_curves.png` | Line plot: ELBO loss vs step for all 3 models on shared axes. Label each curve. y-axis should start near min loss or use log scale if needed. | `losses1`, `losses2`, `losses3` from training. |
| 5 | `temporal_driver_skills.png` | Line plot: `s[t, d]` over seasons (2011–2024) for Hamilton, Verstappen, Alonso, and 3–4 other notable drivers. One line per driver. Optional: light shaded band for ±1 std. | `posterior2["s_loc"]` (14,77). Map indices via `dataset.driver_map`. |
| 6 | `temporal_constructor_performance.png` | Same structure for Mercedes, Red Bull, Ferrari, and 2–3 others. | `posterior2["c_loc"]` (14,17). Map indices via `dataset.constructor_map`. |
| 7 | `wet_weather_specialists.png` | Horizontal bar chart of `delta_d` posterior mean for all 77 drivers, sorted descending. Bars colored by sign (positive = blue, negative = red). | `posterior3["delta_d_loc"]` (77,). |
| 8 | `beta_pi_posterior.png` | Histogram or density plot of `beta_pi` approximate posterior. Since we only have SVI loc/scale, draw 10,000 samples from `Normal(beta_pi_loc, beta_pi_scale)` and plot a KDE or histogram. Vertical line at mean. | `posterior3["beta_pi_loc"]` and `pyro.param("beta_pi_scale")`. |
| 9 | `cross_model_driver_ranking.png` | Top-15 drivers by posterior mean `s_d` shown side-by-side for Models 1, 2 (latest season = 2024), and 3 (latest season). Use grouped horizontal bars or a dot plot. | `posterior1["s_loc"]`, `posterior2["s_loc"][13]`, `posterior3["s_loc"][13]`. |
| 10 | `uncertainty_vs_races.png` | Scatter: posterior std (y) vs number of races per driver (x). Model 1 only. Show a weak negative trend. | Compute race counts from `dataset.driver_idx` via `torch.bincount`. Use `posterior1["s_scale"]`. |

**Plot implementation notes:**
- Use `matplotlib.pyplot` and `seaborn` (already in dependencies).
- For plots 5 and 6, season labels on x-axis should be 2011, 2012, ..., 2024 (not 0–13).
- For plot 9, Model 1 uses static `s_loc`; Models 2 & 3 use season 13 (2024) `s_loc`.
- For plot 10, compute race counts: `counts = torch.bincount(dataset.driver_idx, minlength=dataset.n_drivers)`.

---

## Verification Commands

Run this after implementing:

```bash
# 1. End-to-end pipeline
uv run python -m models.pgm_backend.run_pgm

# 2. Check outputs exist
ls outputs/pgm_model/*.csv
ls outputs/pgm_model/plots/*.png

# 3. Tests still pass
uv run python -m pytest models/pgm_backend/tests/ -v
```

Expected output after successful run:
- `outputs/pgm_model/baseline_posterior.csv` — exists
- `outputs/pgm_model/extended_posterior.csv` — exists
- `outputs/pgm_model/full_posterior.csv` — exists
- `outputs/pgm_model/nuts_vs_svi_comparison.csv` — exists (updated or unchanged)
- `outputs/pgm_model/plots/prior_predictive_win_rate.png` — exists
- `outputs/pgm_model/plots/svi_vs_nuts_scatter.png` — exists
- `outputs/pgm_model/plots/synthetic_recovery.png` — exists
- `outputs/pgm_model/plots/elbo_curves.png` — exists
- `outputs/pgm_model/plots/temporal_driver_skills.png` — exists
- `outputs/pgm_model/plots/temporal_constructor_performance.png` — exists
- `outputs/pgm_model/plots/wet_weather_specialists.png` — exists
- `outputs/pgm_model/plots/beta_pi_posterior.png` — exists
- `outputs/pgm_model/plots/cross_model_driver_ranking.png` — exists
- `outputs/pgm_model/plots/uncertainty_vs_races.png` — exists

---

## Acceptance Criteria

- [ ] `posterior.py` created with `extract_posterior()` that handles all three models
- [ ] `run_pgm.py` created with `main()` that trains all 3 models end-to-end
- [ ] `python -m models.pgm_backend.run_pgm` executes without errors
- [ ] All 4 CSVs are written to `outputs/pgm_model/`
- [ ] All 10 PNG plots are written to `outputs/pgm_model/plots/`
- [ ] `elbo_curves.png` shows three decreasing curves (allow noise after step ~3000)
- [ ] `temporal_driver_skills.png` shows Hamilton peak around 2014–2020
- [ ] `temporal_constructor_performance.png` shows Mercedes peak 2014–2020, Red Bull rise post-2021
- [ ] `wet_weather_specialists.png` shows all 77 drivers sorted by `delta_d` (do NOT filter to top-N only)
- [ ] `beta_pi_posterior.png` shows a distribution centred near **+0.25** (positive, NOT negative)
- [ ] `cross_model_driver_ranking.png` shows top-15 drivers across all 3 models
- [ ] `uncertainty_vs_races.png` shows more races → lower posterior std (negative trend)
- [ ] `pytest models/pgm_backend/tests/ -v` still passes (4 tests)
- [ ] Total runtime < 45 minutes on CPU

---

## When You Are Done

**Step 1 — Append to `tasks/handoff_log.md`** using the template at the bottom of that file:
- Report actual runtime of the full `run_pgm.py` execution
- Report file sizes or row counts for each CSV
- Report any plot that looks wrong or required workarounds
- Note any deviations from the spec

**Step 2 — Append to `tasks/report_notes.md`** only if you made a non-obvious decision.

**Step 3 — Report back:**
1. Full output of the verification commands (runtime, file list)
2. Whether all acceptance criteria pass
3. Any deviations or concerns

Then stop. Do not implement anything beyond T8.
