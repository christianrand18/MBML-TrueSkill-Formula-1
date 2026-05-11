# Report Outline — F1 Skill Separation via Bayesian PGM

**Format:** 6-page IEEE double-column  
**Deadline:** 2026-05-15  
**Figures:** 4 finalised (see `outputs/pgm_model/plots/report/`)  
**Plate diagram:** To be done in LaTeX (tikz / bayesnet package)

---

## Figure List

| # | File | Content |
|---|---|---|
| 1 | LaTeX (tikz) | Plate diagram — generative model, Model 3 (Full) |
| 2 | `fig2_constructor_trajectories.png` | Constructor performance over time with era annotations |
| 3 | `fig3_static_vs_temporal.png` | Static vs temporal driver skill comparison |
| 4 | `fig4_inference_validation.png` | (a) SVI vs NUTS scatter; (b) Synthetic recovery |
| 5 | `fig5_model3_scalars.png` | Posterior densities for β_w and β_π |

---

## Section 1 — Introduction (~0.4 pages)

**Goal:** Frame the scientific problem and state what this paper does.

**Content:**
- F1 is a team sport masquerading as an individual one. Every result conflates driver
  skill and car performance — they are never observed separately.
- The challenge: identify latent driver skill `s_d` and constructor performance `c_k`
  from observed finishing orders alone. Leverage comes from (a) teammates in the same
  car producing different results, and (b) drivers switching teams across seasons.
- TrueSkill (Herbrich et al., 2006) is the natural starting point: it was designed
  for exactly this kind of multi-player skill inference under uncertainty.
- This paper extends TrueSkill into a full Bayesian PGM implemented in Pyro, with
  three contributions:
  1. Replace the intractable Gaussian ranking likelihood with Plackett-Luce
  2. Add temporal AR(1) dynamics to track season-to-season skill evolution
  3. Extend to wet-weather interactions, pit-stop covariates, and reliability

---

## Section 2 — Background: From TrueSkill to Plackett-Luce (~0.5 pages)

**Goal:** Explain the TrueSkill generative model, why it can't be ported directly
to Pyro, and how Plackett-Luce solves this.

**Content:**

**TrueSkill generative model:**
```
s_d  ~ Normal(μ_d, σ_d²)          # latent skill
p_dr ~ Normal(s_d + c_k, β²)      # per-race performance draw
observed ranking = argsort(p)
```
Inference in TrueSkill uses Expectation Propagation (EP) to analytically integrate
out performance draws. EP is not available in Pyro — implementing it would require
sampling all ~6,000 per-race performance draws explicitly, making the latent space
intractably large and the posterior poorly identified.

**The Plackett-Luce solution:**
Replacing Gaussian performance noise with Gumbel-distributed noise yields the
Plackett-Luce likelihood in closed form:
```
log P(π | p) = Σ_i [ p_{π(i)} - log Σ_{j≥i} exp(p_{π(j)}) ]
```
This is the exact likelihood for the observed finishing order — computed via
`torch.logsumexp` for numerical stability. It is strictly more correct than
decomposing a race into N(N−1)/2 independent pairwise comparisons (the
Bradley-Terry / pairwise probit approach used in earlier work), which discards
the joint structure of the race.

**Key message:** Our model is TrueSkill's generative skeleton with a tractable
ranking likelihood — same probabilistic concept, principled implementation.

---

## Section 3 — Model (~1.5 pages)

**Goal:** Describe all three model tiers clearly. Emphasise WHY each design choice
was made and why each extension is scientifically motivated.

### 3.1 Design Choices

**Identifiability — sum-to-zero constraint:**  
The performance equation `p = s_d + c_k` is unidentified up to a global shift: adding
a constant to all `s_d` and subtracting it from all `c_k` leaves the likelihood
unchanged. We enforce `Σ_k c_k = 0` via reparameterisation: sample K−1 free
constructor scores `c_raw` and derive `c_K = −Σ c_raw`. This holds exactly
throughout training.

**Grid position excluded:**  
Qualifying performance is itself an expression of the latent variables we are
estimating — driver skill and constructor quality jointly determine how fast a
car laps in qualifying, and therefore where it starts on the grid. Grid position
is downstream of the skill signal, not independent of it. Conditioning on grid
position as a covariate would partial out this information, producing skill
estimates that reflect only race-execution ability (overtaking, tyre management,
strategy) rather than total driver quality. Since our goal is holistic skill
separation — capturing the full contribution of driver and car to race outcomes —
we exclude grid position and allow qualifying performance to contribute to the
skill signal naturally. A complete treatment would model qualifying and race
jointly in a two-stage model; we leave this as future work.

**DNF handling:**  
Including mechanical DNFs at last place in the Plackett-Luce ranking creates an
asymmetric bias: a high-performing constructor (large `c_k`) receives a larger
gradient penalty for a mechanical failure than a low-performing one, because the
model is more surprised by a last-place Mercedes than a last-place HRT. This is
directionally wrong. Models 1 and 2 therefore exclude mechanical DNFs from the
ranking entirely. Model 3 adds a separate Bernoulli reliability term.

**Constructor rebranding:**  
The Ergast database assigns new IDs at each rebranding (e.g. Force India → Racing
Point → Aston Martin). The AR(1) temporal model requires identity continuity across
seasons. Six remappings are applied before indexing.

### 3.2 Three-Tier Complexity Ladder

The three models share the same Plackett-Luce likelihood and sum-to-zero constraint.
They differ in the richness of the latent structure and the covariates included.
Each tier answers a strictly harder scientific question than the one before it.

**[TABLE — show full specification per model, not just deltas]**

| Model | Latent variables | Observed covariates | Scientific question |
|---|---|---|---|
| **1 — Baseline** | `s_d` (D scalars), `c_k` (K−1 scalars) — both static across all seasons | None | Can we separate driver from car at all? |
| **2 — Extended** | `s_{d,t}` (D×T, AR(1) random walk), `c_{k,t}` (K−1×T, AR(1)), `e_c` (C circuit effects), `β_w` (global scalar) | `w_r` — binary wet indicator per race | Do skills change over time? Does the car dominate in certain regulation eras? |
| **3 — Full** | All of Model 2, plus `δ_d` (D wet-skill scalars), `β_π` (global scalar), `α_rel` (reliability intercept) | All of Model 2, plus `π_{d,r}` — normalised pit-stop duration | Do some drivers excel specifically in the rain? Does pit-stop execution affect results beyond car pace? |

**Model 1 — Baseline (Static):**  
The simplest possible instantiation of the skill-separation problem. One scalar skill
per driver `s_d` and one scalar performance per constructor `c_k`, both fixed across
all 14 seasons. Performance equation:
```
p_{d,r} = s_d + c_{k(d,r)}
```
Priors: `s_d ~ N(0, 1)`, `c_raw ~ N(0, 1)^{K-1}`. This model serves as the
identifiability proof of concept — if inference recovers sensible driver and constructor
rankings from 286 races, the Plackett-Luce signal is sufficient for skill separation.
Its fundamental limitation is that it averages skill over 14 seasons, conflating
car-era dominance (e.g. Mercedes 2014–2021) with driver ability.

**Model 2 — Extended (Temporal):**  
Replaces the static skills with season-level AR(1) random walks, allowing skills to
evolve year by year:
```
s_{d,0} ~ N(0, σ_s)
s_{d,t} ~ N(s_{d,t-1}, γ_s)    for t = 1..T-1
```
and symmetrically for `c_{k,t}`. The innovation variance `γ_s = 0.3` (drivers) and
`γ_c = 0.5` (constructors — larger, reflecting that regulation changes can cause
step-changes in car performance overnight). Implemented via cumulative sums of
sampled innovation vectors rather than a recursive sample loop, keeping the latent
space fully vectorised. Two additional terms are added: circuit-specific effects
`e_c ~ N(0, σ_e)` absorb track-specific biases independent of car and driver, and
a global wet-weather coefficient `β_w ~ N(0, 0.5)` tests whether rain shifts all
drivers' performance equally. This is the key model tier: it can recover
regulation-era step-changes in constructor performance that the static model
completely misses.

**Model 3 — Full:**  
Extends Model 2 with three additions that each capture a distinct source of variance
in race outcomes:

1. **Driver wet-weather interaction** `δ_d · w_r`: a driver-specific wet-weather
   skill modifier that activates only in wet races. Crucially, this is a
   multiplicative interaction with the rain indicator `w_r`, not an additive term —
   `δ_d` on its own would affect all races, which is wrong. `β_w` captures the
   average wet-weather effect across all drivers; `δ_d` captures each driver's
   deviation from that average.

2. **Pit-stop covariate** `β_π · π_{d,r}`: normalised pit-stop duration enters as
   a fixed observed covariate. Conditioning on it allows `c_k` to be interpreted as
   pure car pace — operational execution is attributed to a separate coefficient.

3. **Bernoulli reliability term**: mechanical DNFs are excluded from the Plackett-Luce
   ranking (avoiding the asymmetric bias described above), but they carry real
   constructor signal. A separate observation equation `sigmoid(−α_rel − c_k)` models
   the probability of a mechanical failure: better constructors fail less often.
   `α_rel` absorbs the baseline rate; `c_k` adjusts each constructor relative to
   the field.

**[FIGURE 1: Three plate diagrams side by side in tikz — one per model,**
**showing how the structure grows from Model 1 to Model 3]**

---

## Section 4 — Inference (~0.5 pages)

**Goal:** Describe SVI, justify NUTS as a validation step, and show the model is
correctly implemented.

**Content:**

**Stochastic Variational Inference (SVI) — all three models:**  
The posterior `p(s, c | data)` is intractable in closed form. SVI approximates it
by choosing a parameterised family of distributions `q_φ(s, c)` — the *guide* —
and optimising its parameters `φ` to minimise the KL divergence from the true
posterior. This is equivalent to maximising the Evidence Lower BOund (ELBO):
```
ELBO(φ) = E_{q_φ}[log p(data, s, c)] − E_{q_φ}[log q_φ(s, c)]
```
We use a *mean-field* guide: independent Normal distributions for every latent
variable. This factorises completely across drivers, constructors, and seasons,
making gradient steps cheap but ignoring posterior correlations. Optimiser:
ClippedAdam (lr = 0.01, gradient clip norm = 10 to prevent instability in early
training). Steps: 3,000 for Model 1; 5,000 for Models 2–3. The ELBO decreases
monotonically in all three runs, confirming convergence.

**NUTS on Model 1 — inference validation:**  
Variational inference is approximate by construction — the mean-field guide may
be too restrictive to capture the true posterior shape. To verify that SVI is
producing trustworthy results, we run MCMC using the No-U-Turn Sampler (NUTS)
on Model 1. NUTS is a gradient-based MCMC method that simulates Hamiltonian
dynamics to propose distant, high-probability samples, avoiding the random-walk
behaviour of simpler MCMC methods. Unlike SVI, it is asymptotically exact: given
sufficient samples, it converges to the true posterior.

NUTS is feasible only on Model 1 (D + K − 1 ≈ 93 parameters; ~9 min on CPU).
Models 2 and 3 have ~1,350 temporal latents — NUTS cost scales poorly with
dimension and would require days to converge. The strategy is therefore: validate
on Model 1 that SVI approximates the true posterior adequately, then use SVI
for Models 2 and 3 with confidence.

**Results of the comparison:** All R-hat < 1.05 (max 1.025), confirming NUTS
chains converged. Driver posterior means agree closely between SVI and NUTS.
Constructor means show larger discrepancy — the sum-to-zero constraint couples
all K constructor parameters, and the mean-field guide ignores this coupling.
This is a known limitation of factorised guides discussed in Section 6.

**[FIGURE 4: SVI vs NUTS scatter + synthetic recovery]**

**Synthetic data recovery:**  
Before running on real data, we verify the model is correctly implemented using
ancestral sampling: ground-truth skills are fixed, synthetic finishing orders
generated by sampling from the generative model, and inference is run on the
synthetic data. If the model is correct, inference should recover the known
ground-truth values. Relative ordering is recovered near-perfectly (r = 0.999,
panel b). Absolute magnitudes show prior shrinkage toward zero — expected
behaviour since the Plackett-Luce likelihood is shift-invariant and the Normal
prior is the only anchor on the absolute scale.

**Prior choices:**  
`σ_s = σ_c = 1.0` — weakly informative. A prior predictive check (ancestral
sampling without conditioning on data) confirms that these priors produce win
rates in the realistic 20–80% range, neither deterministic nor random. AR(1)
innovation scales: `γ_s = 0.3` (drivers can shift ≈0.3 performance units per
season), `γ_c = 0.5` (constructors can shift more — regulation changes can
cause overnight step-changes in car performance).

---

## Section 5 — Results (~2 pages)

**Goal:** Present the findings in order of strength: constructors first (best result),
then drivers, then Model 3 scalars.

### 5.1 Can We Separate Driver from Car? (Model 1)

Brief: static rankings are plausible (Hamilton, Verstappen, Vettel in top 5) but
conflate era-specific car advantages with driver skill. Piastri ranks #1 in the static
model — a 2023–2024 driver in the best car of that era. This motivates Model 2.

**[FIGURE 3: Static vs temporal driver comparison]**

The scatter shows drivers above the diagonal were underrated by the static model
(Rosberg — competed only 2011–2016, when Mercedes was not yet dominant) and
drivers below were overrated (Piastri, Norris — benefited from the 2024 McLaren).

### 5.2 How Do Skills Evolve? (Model 2)

**Constructor trajectories** are the strongest result in the paper.

**[FIGURE 2: Annotated constructor trajectories]**

The model recovers three regulation-era transitions without any external label:
- Mercedes rises sharply from 2014 (hybrid regulations), peaks in 2019 (μ ≈ 2.3),
  then declines from 2022 as the ground-effect regulations eroded their advantage.
- Red Bull was dominant in 2011–2013 (Vettel era), competitive through the hybrid
  era, and surges in 2022–2023 (μ ≈ 2.0) — Verstappen's dominant season.
- McLaren collapses in 2015 to negative values — exactly when the Honda engine
  partnership failed — and recovers gradually to become the top constructor in 2024.

Driver career arcs (temporal average over active seasons): Hamilton ranks #1,
Verstappen #2. The temporal model correctly attributes McLaren/Red Bull era success
to the car rather than inflating recent drivers' skill estimates.

### 5.3 What Else Can We Learn? (Model 3)

**[FIGURE 5: β_w and β_π posterior densities]**

- **β_w ≈ −0.08 (σ = 0.51):** The posterior on the global wet-weather coefficient
  is centred near zero and spans the entire prior range. Wet conditions produce no
  consistent global shift in performance — any wet-weather signal is driver-specific.
- **β_π ≈ +0.26 (σ = 0.02):** Tightly estimated and positive. Counter-intuitive
  (faster pits should improve results, implying a negative coefficient), but reflects
  that pit-stop duration correlates with strategic positioning: top teams run longer
  two-stop strategies that reflect car advantage. The coefficient absorbs execution
  and strategy jointly.
- **α_rel ≈ 2.02:** Baseline mechanical DNF probability `sigmoid(−2.02) ≈ 11.7%`,
  close to the empirical 8.7% rate. Constructor `c_k` then adjusts reliability
  relative to the field average.
- **Wet-weather specialists (δ_d):** Posterior means are small and uncertainty is
  high for most drivers (~30 wet races out of 286 provide limited signal). The model
  cannot confidently identify individual wet-weather effects; this is reported as a
  limitation rather than a finding.

---

## Section 6 — Discussion & Conclusion (~0.6 pages)

**Goal:** Interpret the results honestly, name the limitations, and state what the
paper adds.

**What the temporal model adds:**  
The static model conflates car-era dominance with driver skill — the most egregious
example being Piastri ranking #1 over Hamilton due to the 2024 McLaren. The temporal
AR(1) model resolves this by attributing season-specific advantages to the constructor
trajectory. The constructor results are historically accurate and emerged purely from
race outcomes, with no external labels for regulation changes.

**Mean-field SVI limitation:**  
The mean-field guide factorises across all latents, ignoring the posterior correlation
induced by the sum-to-zero constraint on constructors. NUTS reveals that SVI
underestimates constructor uncertainty. For ranking purposes this is acceptable; for
credible interval statements about individual constructor performance, a richer guide
(multivariate Normal, normalising flow) would be needed.

**Short-career driver extrapolation:**  
The AR(1) model infers skill trajectories for all 14 seasons for every driver,
including seasons before they entered F1 or after they retired. In those seasons the
posterior collapses to the AR(1) prior — the values are meaningless extrapolations,
not data-driven estimates. Driver rankings should be interpreted only over their
active seasons.

**Future work:**  
- Learnable temperature parameter β (equivalent to TrueSkill's performance noise)
- Separate reliability latent `r_k` to disentangle pace and mechanical robustness in `c_k`
- Richer variational family (LKJ prior on constructor covariance) to reduce mean-field bias
- Circuit-weather interaction term to disentangle Spa-type confounding

**Conclusion:**  
We present a three-tier Bayesian PGM for F1 skill separation, grounded in TrueSkill's
generative framework but using the Plackett-Luce likelihood for tractable inference.
The temporal model recovers historically accurate constructor performance trajectories
— Mercedes' hybrid-era dominance, Red Bull's 2022–2023 surge, McLaren's Honda
collapse — purely from race finishing orders. Driver skill rankings are credible for
long-career drivers and correctly reflect uncertainty for short-career ones.
