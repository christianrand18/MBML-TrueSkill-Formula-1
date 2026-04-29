# Current Task

**Status:** ALL IMPLEMENTATION COMPLETE
**Task ID:** T8 (final task)
**Task name:** Posterior Extraction + Orchestrator + Plots

---

## Summary

All tasks T1–T8 have been implemented and verified. The full PGM pipeline is operational.

### Final review outcome for T8

- [x] `posterior.py` created with `extract_posterior()` — handles baseline, extended, full
- [x] `run_pgm.py` created with `main()` — trains all 3 models, exports CSVs, generates 10 plots
- [x] End-to-end pipeline executes without errors (runtime ~13 min on M1 Pro CPU)
- [x] All 4 CSVs written: `baseline_posterior.csv`, `extended_posterior.csv`, `full_posterior.csv`, `nuts_vs_svi_comparison.csv`
- [x] All 10 PNG plots written to `outputs/pgm_model/plots/`
- [x] `pytest models/pgm_backend/tests/ -v` passes (4/4 tests)
- [x] ELBO curves decrease; temporal plots show expected peaks; beta_pi centred at +0.26

### Deviation found and fixed during review

**Prior predictive plot bug:** `_plot_prior_predictive()` in `run_pgm.py` counted the prior-fastest driver as "winning" if they were picked at ANY position in the simulated race, not just P1. This produced a spurious win rate of 1.00. Fixed by tracking `pos` and only counting `pos == 0`. Corrected win rate = 0.29 (within 20–80% acceptance band).

---

## What Remains

- Report writing (deadline 2026-05-15)
- Any additional sensitivity analyses or posterior predictive checks desired for the report

No further implementation tasks remain in `tasks/plan.md`.
