# Issues to Investigate / Debug

Findings from a code and data audit of the pipeline against the report notes.
Each issue has been verified against the actual source code and data.

**Resolution date:** 2026-05-11 — see `bugfixes.md` for full details.

---

## Issue 1 — StatusIDs 18 & 19 in both MECHANICAL and FINISHED sets ✅ FIXED

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

**Resolution:** Removed 18 and 19 from `MECHANICAL_STATUS_IDS`. 3 rows now correctly
classified as finishers (in ranking) rather than mechanical DNFs (excluded). N_entries
increased from 5457 → 5520 (combined with Issue 6 fix).

---

## Issue 2 — alpha_rel ≈ 1.6 in report notes is mathematically wrong ✅ FIXED

**Severity:** Low — documentation error only

**What the notes said (Section 4.3):**
> `alpha_rel` absorbs the baseline mechanical DNF rate (≈17% in this dataset;
> at convergence `alpha_rel ≈ 1.6`)

**Why this was wrong:**

The parameterisation is `mech_prob = sigmoid(-alpha_rel - c_k)`. At the field
average (`c_k ≈ 0`), `mech_prob = sigmoid(-alpha_rel)`. The actual mechanical DNF
rate in the dataset is 7.7% (after reclassification), not 17%.

```
sigmoid(-1.6) = 0.168  →  implies 16.8% DNF rate  (what notes claimed)
sigmoid(-2.09) = 0.110 →  implies 11.0% DNF rate  (what model now estimates)
```

**Resolution:** Updated `report_notes.md` §4.3 to alpha_rel ≈ 2.09 and DNF rate ≈ 7.7%.
Updated `REPORT_OUTLINE.md` §5.3 to alpha_rel ≈ 2.09, sigmoid ≈ 11.0%.
After re-running pipeline: actual posterior alpha_rel = 2.09, confirming the fix.

---

## Issue 3 — pit_norm confound: zero-duration entries distort beta_pi ✅ FIXED

**Severity:** High — confirmed as the true cause of the beta_pi > 0 finding

**What the data showed:**

271 ranking entries (4.9% post-reclassification) have `total_pit_duration_ms = 0`.
These are driver-fault DNFs who retired before making a pit stop.

```
0-pit entries at positions 19+:  ~77%
Mean pit_norm z-score (0-pit):  negative (below mean)
Mean pit_norm z-score (nonzero): near zero
```

**The confound:** Drivers with 0 pit time (early retirements) are systematically at
the bottom of the Plackett-Luce order. The model therefore learned:
```
pit_norm very negative → ranked last → higher pit_norm = better performance
```

**Resolution:** Replaced naive z-scoring with robust procedure:
- Zero entries excluded from per-season mean/std computation
- Zero entries assigned `pit_norm = 0` (neutral pit contribution)
- Combined with Issue 4 fix (winsorisation of extreme values)

**Confirmation:** beta_pi flipped from **+0.26 to −0.0023** (±0.03). The previous
positive value was entirely a data artefact. Pit duration has zero detectable effect
on race performance after cleaning the data.

---

## Issue 4 — 382 extreme pit outliers (>1M ms) corrupt the pit_norm z-scores ✅ FIXED

**Severity:** High — data quality issue affecting 6.6% of ranking entries

**What the data showed:**

382 ranking entries (6.6%) have `total_pit_duration_ms > 1,000,000 ms` (>16 min).
The maximum is **3,703,013 ms = 61.7 minutes** — physically impossible for pit work.

```
Outlier finishing positions: mean = 9.3  (normal finishers, mid-field)
Outlier values cluster by race: 24 races affected
Per-stop avg in affected races: ~14 minutes (impossible)
```

**Most likely explanation:** The `milliseconds` column in the source `pit_stops` table
encodes race-timing values (lap time × lap number at pit entry) rather than actual
pit stop durations for these races. `build_f1_model_data.py` sums these values across
stops, producing a variable that reflects strategic positioning rather than crew
execution speed.

**Resolution:** Added winsorisation at 99th percentile within each season before
computing z-score mean/std. This prevents outlier races from inflating the standard
deviation and compressing the z-scores of normal entries. The raw source data issue
cannot be fixed without corrected pit_stops data; the covariate interpretation is
revised from "operational execution" to "data-limited exploratory covariate" in
`report_notes.md` §11.

---

## Issue 5 — "AR(1)" in code comments and notes is actually a random walk (ρ = 1) ⬜ WON'T FIX

**Severity:** Low — report terminology already correct

**What the code does (both `model_extended.py` and `model_full.py`):**

```python
s = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + s_innov.cumsum(0)], dim=0)
```

This implements `s[t] = s[t-1] + innov[t-1]` — a **random walk** with no
mean-reversion. ρ = 1 exactly.

**Resolution:** No code change needed. `REPORT_OUTLINE.md` already correctly uses
"AR(1) random walk" throughout. The random walk is a valid design choice. A
stationary AR(1) with ρ < 1 would be a different model with different properties;
the current implementation is intentional.

---

## Issue 6 — StatusId 130 "Collision damage" in MECHANICAL set ✅ FIXED

**Severity:** Medium — 60 entries misclassified

**What the data showed:** StatusId 130 ("Collision damage") appears in
`MECHANICAL_STATUS_IDS` with 60 occurrences. Collision damage is a driver-external
or racing-incident outcome, not a car-internal mechanical failure.

**Resolution:** Removed 130 from `MECHANICAL_STATUS_IDS`. 60 entries now correctly
classified as driver-fault DNFs (in ranking, not in Bernoulli reliability term).
Combined with Issue 1 fix, N_entries increased from 5457 → 5520 and is_mech mean
dropped from 0.0875 → 0.0769.

---

## Issue 7 — FINISHED_STATUS_IDS computed but never used ⬜ WON'T FIX

**Severity:** Low — dead code, zero impact

**Finding:** `FINISHED_STATUS_IDS` is defined at `data_preparation.py:26-28` but
never referenced. The ranking exclusion logic uses `~is_mechanical` instead.

**Resolution:** Not fixed. No impact on the pipeline. Low-priority cleanup.

---

## Issue 8 — Renault AR(1) continuity gap (2012–2015) ✅ DOCUMENTED

**Severity:** Low-Medium — affects constructor trajectory interpretation

**Finding:** constructorId 4 (Renault) has data in 2011 and 2016–2020, but not
2012–2015. During the gap, the Enstone factory ran as Lotus F1 Team under
different constructorIds (205, 206, 207). The AR(1) random walk bridges the gap
via the innovation prior — Renault's constructor trajectory during 2012–2015 is
a prior artefact, not a data-driven estimate.

**Resolution:** Documented in `report_notes.md` T11. The Lotus constructorIds are
deliberately kept separate (different legal entity, different technical leadership).
Constructor trajectories should be interpreted only over contiguous seasons.

---

## Priority order — final status

| # | Issue | Action | Status |
|---|-------|--------|--------|
| 4 | Pit variable: extreme outliers | Winsorise + reinterpret in report | ✅ |
| 3 | Zero-duration pit confound | Exclude zeros from mean/std, set pit_norm=0 | ✅ |
| 1 | StatusIDs 18/19 in MECHANICAL | Remove from MECHANICAL_STATUS_IDS | ✅ |
| 6 | StatusID 130 in MECHANICAL | Remove from MECHANICAL_STATUS_IDS | ✅ |
| 2 | alpha_rel ≈ 1.6 wrong in notes | Update to 2.09 | ✅ |
| 8 | Renault continuity gap | Documented in report_notes.md | ✅ |
| 5 | "AR(1)" terminology | Report already correct | ⬜ |
| 7 | FINISHED_STATUS_IDS unused | Dead code, no impact | ⬜ |
