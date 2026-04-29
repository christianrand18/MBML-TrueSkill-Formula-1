# Report Notes — Design Decisions and Modelling Considerations

These notes capture the reasoning behind each major design decision made during
the modelling discussion. They are intended to be drawn on directly when writing
the methods section and discussion of the report.

---

## 1. Ranking Likelihood: Plackett-Luce over Pairwise Probit

**Decision:** Use a proper Plackett-Luce ranking likelihood instead of the pairwise
probit (Thurstone-Mosteller) approximation used in the existing `pyro_backend`.

**Reasoning:**  
The pairwise probit approach decomposes a race with N finishers into N(N−1)/2
independent binary comparisons. This is an approximation that throws away information:
it treats "Hamilton beat Verstappen" and "Hamilton beat Bottas" as independent events,
when in reality both are consequences of the same latent performance draw. The
Plackett-Luce model is the exact likelihood for the observed ordering — the probability
of the full observed ordering equals the product of sequential softmax draws, which
correctly accounts for the joint nature of the race result.

Concretely:
```
P(π) = ∏_i  exp(p_{π(i)}) / Σ_{j≥i} exp(p_{π(j)})
```

This is a sequential elimination model: at each position, the winner is drawn with
probability proportional to their exponentiated performance, given the remaining
field. The pairwise probit recovers an approximation of this in the limit of many
pairwise comparisons, but loses the conditioning structure.

**For the report:** Emphasise that this is a core modelling fidelity choice, not just
an implementation detail. The pairwise approach was available as a baseline but
deliberately replaced.

---

## 2. Grid Position: Excluded as Blocking Variable

**Decision:** Grid position (qualifying result) is excluded from all three models.

**Reasoning:**  
Grid position sits on the causal path from skill to race result:

```
Driver Skill ──► Qualifying ──► Grid Position ──► Race Result
Constructor  ──►                               ──►
```

If grid position is always observed and included as an intermediate node, it blocks
the information flow from skill to race result. The model would decompose into two
independent sub-models:
- "Skill → Grid position" (qualifying model)
- "Grid position → Race result" (race execution model)

Estimating them jointly becomes pointless — no information passes between skill and
race result. Including grid position as an additive covariate is technically acceptable
(it does not block the path in the same way), but it smuggles in qualifying performance
and muddies what the skill latent actually means: a driver's posterior skill would
partially reflect their qualifying ability rather than their race-day ability.

Since the goal of the model is to infer race skill from race outcomes, the cleanest
design excludes grid position entirely. The existing `pyro_backend` model included
it as a covariate; this is noted as a limitation of that model in the results
comparison.

**For the report:** This is a good example of the "blocking variable" pitfall described
in the DTU course materials. Explicitly reference the causal graph and explain why
the exclusion is necessary for valid inference.

---

## 3. Identifiability: Sum-to-Zero Constraint on Constructors

**Decision:** Apply a sum-to-zero constraint on constructor performance parameters,
implemented as a reparameterisation with K−1 free parameters.

**Reasoning:**  
The performance equation is `p_{d,r} = s_d + c_k + ε`. Adding a scalar α to all
`s_d` and subtracting α from all `c_k` leaves the likelihood unchanged. The model
is therefore unidentified without a constraint — the posterior is a ridge rather
than a point (or a narrow region).

Two common fixes:
- Pin one constructor to zero (delta prior on `c_reference = 0`)
- Sum-to-zero: `Σ_k c_k = 0`

Pinning one constructor is arbitrary (which one?) and makes that team's choice
load-bearing for all other estimates. Sum-to-zero is symmetric and more natural:
constructor performances are measured relative to the field average, which is
the natural reference for a comparative study.

**Implementation:** Rather than enforcing the constraint post-hoc (which would
bias the guide), it is built into the parameterisation:
```python
c_raw ~ Normal(0, σ_c)^(K-1)          # K-1 free parameters
c = [c_raw, -c_raw.sum()]              # K-th constructor constrained
```
The guide samples `c_raw`, never `c` directly. This ensures the constraint holds
exactly throughout training, not just at convergence.

**For the report:** Mention that without this constraint, the ELBO surface has a
flat direction (the posterior mean of `s_d + c_k` is identified, but not the
individual components). The sum-to-zero fixes this. This is a standard technique
in additive ANOVA-style models.

---

## 4. DNF Handling: A Two-Stage Approach Across Model Tiers

**Decision:** Mechanical and driver-fault DNFs are treated differently, and the
treatment evolves across the three model tiers.

### 4.1 Why the Naive Masking Approach Fails

An initial design masked the driver skill contribution for mechanical DNFs:
```python
p = s[driver_idx] * mech_mask + c[cons_idx]
```
A mechanical DNF gives `p = c_k` (driver shielded, constructor still present).
The Plackett-Luce gradient then pushes `c_k` down because the entry is ranked last.

This is directionally correct — the constructor *should* be penalised for a
mechanical failure — but it creates an asymmetric bias: a high-quality constructor
(large `c_k`) suffers a larger gradient penalty for its reliability failure than a
low-quality constructor, because the Plackett-Luce gradient is proportional to how
surprising the last-place finish is given the latent performance. Concretely:

- Mercedes has a mechanical DNF: `p = c_Mercedes` is large → model is very surprised
  by last place → large downward gradient on `c_Mercedes`
- Haas has a mechanical DNF: `p = c_Haas` is small → model is less surprised → small
  downward gradient on `c_Haas`

This penalises fast constructors disproportionately for reliability failures. It is
directionally wrong.

### 4.2 Models 1 & 2: Exclude from Ranking

Mechanical DNFs are dropped entirely from the Plackett-Luce ranking. The constructor
receives no gradient signal from its mechanical failures. This means `c_k` captures
*pace only* — how fast the car is among those who finish. This is the cleanest
interpretation and avoids the asymmetric bias.

Driver-fault DNFs (accident, collision, spin) ARE included, ranked after all finishers.
Both the driver and constructor are penalised, which is appropriate: the driver caused
the retirement, and the constructor's car was part of the outcome.

### 4.3 Model 3: Bernoulli Reliability Term

Mechanical DNFs carry real information: better constructors fail mechanically less
often. Model 3 adds a separate observation equation to capture this:

```python
mech_prob = sigmoid(-alpha_rel - c[cons_idx])
factor("reliability", Bernoulli(mech_prob).log_prob(is_mech))
```

The parameterisation `sigmoid(-alpha_rel - c_k)` ensures:
- Higher `c_k` (better constructor) → lower mechanical DNF probability ✓
- `alpha_rel` absorbs the baseline mechanical DNF rate (≈17% in this dataset;
  at convergence `alpha_rel ≈ 1.6`)
- `c_k` adjusts reliability relative to the field average

This correctly penalises low-quality constructors more for their failures and gives
high-quality constructors a positive reliability signal.

**Implication for `c_k` interpretation in Model 3:** The latent `c_k` now captures
a combination of pace and reliability. These two dimensions of constructor quality
are positively correlated historically (dominant teams tend to be both fast and
reliable), so the conflation is unlikely to produce pathological estimates. However,
it is worth noting as a modelling assumption. A natural extension would introduce
a separate reliability latent `r_k`, leaving `c_k` as pure pace — but this is out
of scope for this project.

**For the report:** This is a rich methodological discussion. The progression from
"exclude" (Models 1 & 2) to "model explicitly" (Model 3) is a good narrative for
the model complexity ladder. The Bernoulli term is also a clean example of using
auxiliary observations to inform a shared latent variable.

---

## 5. Constructor Rebranding: Continuity over Branding Epochs

**Decision:** Rebranded constructors are treated as the same entity across their
branding epochs. IDs are merged into a canonical ID before indexing.

**Merges applied:**

| Canonical | Merged IDs | Real-world team |
|---|---|---|
| 10 (Force India) | 211 (Racing Point), 117 (Aston Martin) | Silverstone factory |
| 4 (Renault) | 214 (Alpine) | Enstone factory |
| 5 (Toro Rosso) | 213 (AlphaTauri), 215 (Racing Bulls) | Faenza factory |
| 15 (Sauber) | 51 (Alfa Romeo) | Hinwil factory |

**Reasoning:**  
The Ergast database assigns new `constructorId` values at each rebranding, even when
the physical team (factory, engineering staff, ownership) is unchanged. The AR(1)
temporal model requires that the same latent variable `c_{k,t}` propagates across
seasons. If Force India and Racing Point are treated as separate entities, the model
cannot learn that Racing Point 2019 inherited Force India 2018's car capability. The
AR(1) innovation links consecutive seasons of the same constructor; breaking continuity
at a rebrand severs this link without cause.

**Scope:** Small backmarker teams (Marussia, Caterham, HRT, Manor) that permanently
exited F1 are kept as separate entities since they have no continuity to preserve.

**ID mapping verified** from year-by-year activity patterns in the dataset, not
assumed from external documentation.

---

## 6. Temporal Dynamics: AR(1) Random Walk on Skills

**Decision:** Model 2 and 3 use a season-level AR(1) random walk for both driver
and constructor skills.

**Generative process:**
```
s_{d,0} ~ Normal(0, σ_s)
s_{d,t} ~ Normal(s_{d,t-1}, γ_s)   for t = 1..T-1
```
and symmetrically for `c_{k,t}`.

**Reasoning:**  
A single static skill per driver across 14 seasons averages out genuine performance
changes. This is demonstrably wrong for:
- Constructors: Mercedes dominated 2014–2021 under the hybrid era regulations,
  then Red Bull dominated 2022–2024 under ground-effect rules. A static `c_k` for
  Mercedes would average these two eras and fail to capture either.
- Drivers: Career trajectories are real. Alonso 2005–2006 (peak) vs Alonso 2014–2016
  (mid-field car) cannot be captured by a single skill value.

The AR(1) innovation variance γ controls how rapidly skills can change:
- Too small: model cannot track genuine step-changes (e.g. Red Bull 2022 regulation
  benefit)
- Too large: skills jump wildly season-to-season, destroying the sharing of
  information across seasons

**Implementation note:** The AR(1) is implemented using cumulative sums of
innovations rather than a recursive `pyro.sample` loop. This avoids creating
D×T or K×T individual sample sites (which would be ~77×14 = 1,078 driver sites
alone) and instead uses two vectorised samples:
```python
s0 ~ Normal(0, σ_s)^D                     # (D,)
s_innov ~ Normal(0, γ_s)^{(T-1) × D}     # (T-1, D)
s = cumsum([s0, s_innov], dim=time)        # (T, D)
```

**For the report:** This is a standard state-space model applied to sports rating.
TrueSkill uses a similar innovation model (the `tau` parameter in the Microsoft
implementation). Reference the AR(1) literature if space allows.

---

## 7. Wet-Weather Skill: Interaction, Not Additive

**Decision:** In Model 3, the driver wet-weather modifier `δ_d` enters the
performance equation as an interaction with the rain indicator `w_r`:

```
p_{d,r} = ... + β_w · w_r + δ_d · w_r + ε
```

not as two separate additive terms `+ β_w + δ_d`.

**Reasoning:**  
`β_w` is a global weather effect — on average, wet conditions change performance
scores by `β_w` for all drivers equally. `δ_d` is the *deviation* of driver d from
this average wet-weather performance. If both were additive without the `w_r`
interaction, `δ_d` would affect performance in ALL races, not just wet ones. The
interaction `δ_d · w_r` correctly activates the driver-specific wet modifier only
when the race is wet.

This is equivalent to a heterogeneous treatment effect: each driver has a different
response to the "wet weather treatment". The global `β_w` captures the average
treatment effect; `δ_d` captures individual deviations.

**For the report:** This is a good example of careful modelling of conditional
dependence. The distinction between additive and interaction terms is a common
source of model misspecification.

---

## 8. Inference: SVI + NUTS on Baseline, SVI Only on Extensions

**Decision:** Model 1 (Baseline) is trained with both SVI (mean-field guide) and
MCMC (NUTS). Models 2 and 3 are trained with SVI only.

**Reasoning for NUTS on Model 1:**  
The DTU course guidelines explicitly recommend testing multiple inference algorithms
as a sanity check. If NUTS and SVI disagree substantially, it indicates a problem
with the guide (e.g. it is too restrictive and cannot approximate the true posterior)
or the model (e.g. there is a parameterisation issue). Model 1 has a small latent
space (D + K − 1 ≈ 85 parameters for this dataset), making NUTS computationally
feasible.

**Reasoning for SVI only on Models 2 & 3:**  
The temporal models introduce D×T + K×T latents (≈77×14 + 19×14 ≈ 1,344 for
this dataset). NUTS cost scales poorly with dimension — the step size and number
of leapfrog steps must be tuned globally, and the sampler struggles with
high-dimensional geometry. SVI remains tractable because the mean-field guide
factorises across all latents.

**Comparison protocol:**  
The SVI vs NUTS comparison on Model 1 serves as validation: if R-hat < 1.05 for
all latents and the posterior means agree within 0.5 standard deviations, we can
trust that the SVI guide is adequately approximating the true posterior. This gives
confidence that SVI results on Models 2 & 3 are interpretable.

---

## 9. Prior Choices

**Skill priors:**
- `s_d ~ Normal(0, 1)` — centred at zero, unit scale. Skills are measured in
  performance units relative to the field mean. Unit scale means ±1 represents a
  typical spread of skill across the field.
- `c_k ~ Normal(0, 1)` — same rationale, with sum-to-zero applied.

**AR(1) innovation priors (hyperparameters, not learned):**
- `γ_s = 0.3`: driver skill can shift by ≈0.3 performance units per season.
- `γ_c = 0.5`: constructor performance can shift more per season (regulation changes
  are larger shocks than individual driver development).

**For the report:** All priors are weakly informative by design. Sensitivity analysis
(running the model with σ_s ∈ {0.5, 1.0, 2.0}) would strengthen the conclusions
but is left as an extension.

---

## 10. eps, beta, and the TrueSkill Connection

**Decision:** Drop `eps` from the generative process. The correct generative process is:

```
p_{d,r} = s_d + c_{k(d,r)}
y_{d,r} ~ PlackettLuce(softmax(p_{d,r}))
```

**Why sampling eps explicitly is wrong:**  
Adding a per-(driver, race) noise term `eps_{d,r} ~ Normal(0, β²)` as an explicit
latent variable introduces ~6,000 unidentified parameters. Given only the finishing
order, `s_d` and `eps_{d,r}` are perfectly collinear in their effect on the likelihood —
the posterior over `s_d` would collapse toward its prior while `eps` absorbs all signal.
This is a genuine identifiability bug.

**What eps actually does in the original TrueSkill:**  
In TrueSkill, the per-match performance draw `p_{d,r} ~ Normal(s_d + c_k, β²)` is
integrated out analytically via EP message passing. Beta controls the upset probability —
how often a weaker driver beats a stronger one due to race-day randomness. It never
appears as an explicit sample site in TrueSkill either. Dividing Plackett-Luce scores
by β before the softmax is mathematically equivalent to adding Normal(0, β²) noise and
marginalising (under a Gumbel noise assumption). The two formulations are the same model
written differently.

**Why beta as a fixed hyperparameter does nothing:**  
If β is hardcoded and not learned, it is just a global scale on all performance scores.
Since `s_d` and `c_k` are already unidentified up to scale (only their differences are
identified by the likelihood), a fixed β is redundant.

**For the report (methods section):**  
> "Race-day performance variance is implicitly captured by the Plackett-Luce stochasticity.
> A temperature parameter β could be introduced as a learnable scale on performances,
> equivalent to the additive Gaussian noise term in the original TrueSkill formulation —
> we fix β = 1 and leave joint estimation as a future extension."

---

## 11. Pit-Stop Time: Operational Execution Covariate

**Decision:** Pit-stop time (`π_{d,r}`) in Model 3 is described as an **operational
execution covariate**, not a driver variable or a shared variable.

**Framing:**  
Conditioning on pit-stop time allows `c_k` to be interpreted as pure constructor *pace*.
Operational execution — the speed and reliability of the pit crew — is separated from
the raw pace advantage of the car. This is a deliberate decomposition.

**Why this is defensible despite the mediator risk:**  
Pit-stop time does sit on a partial causal path:

```
Constructor quality ──► pit crew execution ──► race result
Constructor quality ──► car pace ──────────► race result
```

By conditioning on pit-stop time, the model attributes the execution pathway to a
separate coefficient (`β_π`) and lets `c_k` reflect only the pace pathway. This is
a valid decomposition *if* pit-stop execution and car pace are imperfectly correlated —
i.e. if some teams are fast but sloppy, or slow but precise. If they were perfectly
correlated, the separation would be ill-conditioned.

**Limitation to flag in the report discussion:**  
Dominant teams like Mercedes historically excelled at *both* pit-stop execution and
raw pace. Over 2014–2021, the two dimensions were strongly positively correlated across
the field. This collinearity limits how cleanly `c_k` and `β_π` can be separated for
those teams. The discussion should note this explicitly: the pace/execution decomposition
is most informative for teams with mismatches between the two dimensions.

---

## 12. Circuit Effects and Weather Confounding

**Limitation to flag in the report discussion:**  
Model 2 introduces a per-circuit latent effect `e_circ` and a global wet-weather
coefficient `β_w`. These two terms can partially confound at circuits with a
consistent wet-weather history. Spa-Francorchamps is the clearest example: it is
both one of the most technically demanding circuits on the calendar and one of the
wettest. A large negative `e_circ[spa]` could reflect genuine circuit difficulty,
persistent wet conditions, or both — the model cannot separate them without
additional structure.

This is not a correctness problem (the model converges), but it affects
interpretation: `β_w` is identified primarily from variance *across* races with
different weather, but if wet races cluster at specific circuits, some of that
signal gets absorbed by `e_circ`.

**Report sentence:**  
> "Circuit effects and global weather effects may partially confound at circuits
> with a consistent wet-weather history (e.g. Spa-Francorchamps). A
> circuit-specific wet-weather interaction term would fully disentangle them but
> is left as a future extension."

---

## 13. Model Progression Narrative

The three models form a deliberate complexity ladder for the report:

| Model | New elements vs. previous | What it adds scientifically |
|---|---|---|
| **Baseline** | — | Pure skill separation: can we separate driver from car at all? |
| **Extended** | AR(1) dynamics, circuit effects, global weather | Does skill change over time? Does rain affect all drivers equally? |
| **Full** | Wet-weather driver interaction, pit-stop covariate, reliability term | Do some drivers excel in the rain? Does pit-stop execution matter? |

Each tier answers a progressively refined question. The baseline is the proof of
concept; the full model is the scientifically richest claim. Posterior predictive
checks and the sanity criteria in SPEC §9.2 are the validation steps at each tier.

---

## 14. [T1] — Mechanical DNF Rate: Empirical Value Lower Than Expected

**Decision:** The assertion on `is_mech.float().mean()` was relaxed from [0.10, 0.25] to
[0.05, 0.25] because the actual mechanical DNF rate in the data is ~8.7%, not the ~17%
estimated in the spec.

**Reasoning:** The exact MECHANICAL_STATUS_IDS set specified in the task was used
without modification. Despite the set containing 33 status IDs, only 24 of them occur
in the dataset (the remaining 9 have zero occurrences). The total mechanical DNF count
is 523 out of 5980 total rows = 8.75%. The spec's expected rate of ~17% (approximately
double the actual value) overestimates the mechanical DNF prevalence in the F1 dataset.

**For the report:** The mechanical DNF rate in Formula 1 (2011–2024) is approximately
8.7% of all entries, not 17%. This affects the calibrati on of Model 3's reliability
term and should be reflected in any discussion of expected Bernoulli success probabilities.
The reliability baseline `α_rel` will converge to a value reflecting this lower rate.

---

## 15. [T2b] — Prior Predictive Check: sigma_s=1.0, sigma_c=1.0 partially pass

**Decision:** Prior predictive check (100 draws, seed=42, 20 drivers, 10 constructors)
with sigma_s=1.0, sigma_c=1.0.

**Results:**
- Prior-fastest driver win rate: 0.39 (within [0.20, 0.80] ✓)
- Mean P1–P20 performance gap: 6.26 (exceeds 5.0 upper bound ✗)

**Reasoning:** The win rate confirms priors are appropriately weakly informative for
predicting winners — the strongest driver wins ~39% of the time, comparable to Hamilton's
historical win rate. However, the performance gap between the best and worst driver
is wider than expected at 6.26 units (vs. the 5.0 upper bound). With sigma_s=1.0 and
sigma_c=1.0, the theoretical gap for ±2σ drivers at ±2σ constructors is ~4 × 1.0 + 4 × 1.0 ≈ 8.0,
and actual draws cluster around ~6.3. Reducing sigma_s, sigma_c, or both to ~0.75 would
bring the expected gap within [1.0, 5.0].

**For the report:** If sigmas are kept at 1.0, note that the priors are wide enough to
accommodate large performance gaps (plausible for F1, where backmarkers can be >6s/lap
slower). If sigmas are reduced to 0.75, note that this was calibrated via prior
predictive checking to keep the P1–P20 gap within a physically reasonable F1 range.
Either choice is defensible; the key is transparency about the prior predictive check
results.
