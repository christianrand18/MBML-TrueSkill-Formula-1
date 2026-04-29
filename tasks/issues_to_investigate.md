# Issues to Investigate / Debug

Findings from a code and data audit of the pipeline against the report notes.
Each issue has been verified against the actual source code and data.

---

## Issue 1 — StatusIDs 18 & 19 in both MECHANICAL and FINISHED sets

**Severity:** Medium — code bug, small but real data misclassification

**What the code does:**
In `models/pgm_backend/data_preparation.py`, statusIDs 18 and 19 appear in **both**
`MECHANICAL_STATUS_IDS` and `FINISHED_STATUS_IDS`:

```python
MECHANICAL_STATUS_IDS = frozenset({5, 6, 7, 8, 9, 10, 18, 19, ...})
FINISHED_STATUS_IDS   = frozenset({1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20})
```

In Ergast, statusIDs 18 and 19 are "+8 Laps" and "+9 Laps" — drivers who finished
the race far behind the leader, NOT mechanical retirements.

**Evidence:** Querying the data shows 3 affected rows:
- raceId 841 (2011): driverId 10, positionOrder 15, num_pit_stops 1 (clearly finished)
- raceId 900 (2014): driverId 824, positionOrder 14, num_pit_stops 2 (clearly finished)
- raceId 917 (2014): driverId 154, positionOrder 17, num_pit_stops 3 (clearly finished)

**Consequence:** These 3 actual finishers are:
1. **Excluded from the Plackett-Luce ranking** (treated as mechanical DNFs)
2. **Counted as mechanical DNFs** in Model 3's Bernoulli reliability term (`is_mech=True`)

**Fix:** Remove 18 and 19 from `MECHANICAL_STATUS_IDS`. StatusIDs 11–20 are all
"+N Laps" finished statuses in Ergast and belong only in `FINISHED_STATUS_IDS`.

Note: `FINISHED_STATUS_IDS` is computed but never actually used in the ranking logic
(the mask is `~is_mechanical`). So the fix is simply removing 18 and 19 from
`MECHANICAL_STATUS_IDS`.

---

## Issue 2 — alpha_rel ≈ 1.6 in report notes is mathematically wrong

**Severity:** Low — documentation error only (no Model 3 outputs saved to verify)

**What the notes say (Section 4.3):**
> `alpha_rel` absorbs the baseline mechanical DNF rate (≈17% in this dataset;
> at convergence `alpha_rel ≈ 1.6`)

**Why this is wrong:**

The parameterisation is `mech_prob = sigmoid(-alpha_rel - c_k)`. At the field
average (`c_k ≈ 0`), `mech_prob = sigmoid(-alpha_rel)`. The actual mechanical DNF
rate in the dataset is **8.75%** (confirmed in report notes Section 14 and the T1
handoff log), not 17%.

```
sigmoid(-1.6) = 0.168  →  implies 16.8% DNF rate  (what notes claim)
sigmoid(-2.35) = 0.087 →  implies 8.7% DNF rate   (what data shows)
```

So the correct value for `alpha_rel` at convergence should be approximately **2.35**,
not 1.6. The 1.6 figure was written based on the spec's expected 17% rate before T1
measured the actual rate. Section 14 corrected the rate but Section 4.3 was never
updated.

**What to do:** Run Model 3 and check the actual posterior mean of `alpha_rel_loc`.
If it's near 1.6, there is a model bug (the reliability term is not being observed
correctly). If it's near 2.35, the notes just need updating.

---

## Issue 3 — pit_norm confound: 249 zero-duration entries distort beta_pi

**Severity:** High — likely the true cause of the beta_pi > 0 finding

**What the data shows:**

249 ranking entries (4.6% of all ranking entries) have `total_pit_duration_ms = 0`.
These are driver-fault DNFs who retired before making a pit stop.

```
0-pit entries at positions 19+:  191 / 249  = 76.7%
Mean pit_norm z-score (0-pit):  -0.971
Mean pit_norm z-score (nonzero): +0.046
Fraction of bottom-of-field (pos ≥19) with 0 pit time: 37%
```

**The confound:** Drivers with 0 pit time (early retirements) are systematically at
the bottom of the Plackett-Luce order. The model therefore learns:

```
pit_norm very negative → ranked last   →  higher pit_norm = better performance
```

This mechanical correlation has nothing to do with pit crew speed. It means
`beta_pi > 0` is driven partly by "drivers who pitted at all finished better than
drivers who didn't pit at all", not by "faster pit stops = better race result".

The report notes (Section T7) explain the positive sign as "top teams use longer
strategic stops". That explanation is likely secondary or wrong. The primary driver
is the zero-duration confound.

**What to check:** Re-run Model 3 with zero-duration entries imputed (e.g. replace
0 with the within-season median) or excluded from the pit covariate. If beta_pi
becomes negative or near zero, the confound explanation is confirmed.

---

## Issue 4 — 360 extreme pit outliers (>1M ms) corrupt the pit_norm z-scores

**Severity:** High — data quality issue affecting 6.6% of ranking entries

**What the data shows:**

360 ranking entries (6.6%) have `total_pit_duration_ms > 1,000,000 ms` (>16 min).
The maximum is **3,703,013 ms = 61.7 minutes** — physically impossible for pit work.

```
Outlier finishing positions: mean = 8.8, min = 1  (normal finishers, NOT backmarkers)
Outlier pit_norm z-scores:   mean = +2.55, max = +18.1
Normal  pit_norm z-scores:   mean = -0.14, max = +6.5
Clustered in years: 2020, 2021, 2022, 2023, 2024
```

**Most likely explanation:** `total_pit_duration_ms` in the enriched CSV appears to
measure the **elapsed time from first pit lane entry to last pit lane exit**, not
the cumulative active pit stop time. For a driver who pits on lap 10 and lap 50
of a 90-second-per-lap race, that span is 40 × 90s = 3,600 seconds = 3.6M ms,
matching the observed values.

If true, the variable does NOT measure pit crew execution speed at all — it
measures the **time window spanned by the strategy**. A driver who pits early and
late has a very large value. A one-stop driver who pits mid-race has a medium value.
An early-retirement (0 stops) has 0.

This completely changes what `beta_pi` captures. The "operational execution
covariate" framing in the report is based on a misunderstanding of what the
column represents.

**What to check:**
1. Check the data_preprocessing code that creates `total_pit_duration_ms` to
   confirm how it is calculated.
2. If it is span-based: the covariate is not measuring pit crew speed and should
   be reconsidered (or dropped from Model 3).
3. Check the actual pit data — does Ergast provide per-stop durations that could
   be summed correctly?

---

## Issue 5 — "AR(1)" in code comments and notes is actually a random walk (ρ = 1)

**Severity:** Low — report terminology issue, code is correctly implemented

**What the code does (both `model_extended.py` and `model_full.py`):**

```python
s = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + s_innov.cumsum(0)], dim=0)
```

This implements `s[t] = s[t-1] + innov[t-1]` — a **random walk** with no
mean-reversion. ρ = 1 exactly.

**What a general AR(1) would be:** `s[t] = ρ·s[t-1] + innov[t-1]` with ρ < 1,
which mean-reverts toward zero between seasons.

**Why this matters for the report:** A random walk (ρ=1) allows skills to drift
without bound over 14 seasons — a strong prior claim. A stationary AR(1) with
ρ ≈ 0.8–0.9 would be more conservative: large skill jumps are possible but the
model expects skills to partially revert between seasons.

The code is not wrong — the random walk is a valid choice — but the report should
not call it "AR(1)" without qualification. Calling it "random walk" or "AR(1) with
ρ=1 (unit root)" is more precise and avoids implying mean-reversion that doesn't
exist.

---

## Priority order for investigation

| # | Issue | Action needed |
|---|-------|---------------|
| 4 | Pit variable definition (span vs. sum?) | Audit `data_preprocessing/` code that builds `f1_enriched.csv` |
| 3 | Zero-duration pit confound | Re-run Model 3 with 0-pit entries excluded/imputed |
| 1 | StatusIDs 18/19 in MECHANICAL set | Fix: remove 18 and 19 from `MECHANICAL_STATUS_IDS` |
| 2 | alpha_rel ≈ 1.6 wrong in notes | Run Model 3, check posterior mean of `alpha_rel_loc` |
| 5 | "AR(1)" terminology | Report language fix only |
