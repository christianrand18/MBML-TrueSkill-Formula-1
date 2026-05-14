# Report Outline — F1 Skill Separation via Bayesian PGM

**Format:** 6-page IEEE double-column
**Deadline:** 2026-05-15

---

## Figure List

| # | File | Content |
|---|---|---|
| 1 | LaTeX (tikz) | Plate diagram — Model 1 (Baseline, Static) |
| 2 | LaTeX (tikz) | Plate diagram — Model 2 (Extended, Temporal) |
| 3 | LaTeX (tikz) | Plate diagram — Model 3 (Full) |
| 4 | `fig4_constructor_trajectories.png` | Constructor performance over time with era annotations |
| 5 | `fig5_static_vs_temporal.png` | Static vs temporal driver skill comparison |
| 6 | `fig6_inference_validation.png` | (a) SVI vs NUTS scatter; (b) Synthetic recovery |
| 7 | `fig7_model3_scalars.png` | Posterior densities for β_w and β_π |

*(Figures moved to appendix if space is tight.)*

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
  3. Extend to wet-weather interactions and pit-stop covariates

---

## Section 2 — Data (~0.5 pages)

**Goal:** Describe the dataset, preprocessing decisions, and key statistics that constrain
the model.

**Content:**

**Source and scope:** The Ergast F1 database provides race finishing orders, pit-stop
timings, weather labels, and DNF status codes for 14 seasons (2011–2024). After
preprocessing:

| Quantity | Count |
|---|---|
| Seasons | 14 (2011–2024) |
| Races | 286 |
| Drivers | 77 |
| Constructors (after rebrands) | 17 |
| Circuits | 35 |
| Total driver-race entries | 5,980 |
| Ranking entries (excl. mechanical DNFs) | 5,520 |
| Mechanical DNF rate | 7.7% (460 entries) |
| Wet races | ~10% (~30 races) |

**Constructor rebranding merges:** The Ergast database assigns new IDs at each rebranding
(e.g. Force India → Racing Point → Aston Martin). The AR(1) temporal model requires
identity continuity across seasons, so six remappings are applied before building integer
indices: Racing Point (211)→Force India (10), Aston Martin (117)→Force India (10),
Alpine (214)→Renault (4), AlphaTauri (213)→Toro Rosso (5), Racing Bulls (215)→Toro Rosso (5),
Alfa Romeo (51)→Sauber (15).

**DNF classification:** 33 mechanical-fault status IDs are used to distinguish mechanical
DNFs from driver-fault DNFs. Only 24 of these 33 IDs actually appear in the dataset,
yielding a mechanical DNF rate of 7.7% — lower than the ~17% initially estimated.
Mechanical DNFs are excluded from Plackett-Luce ranking in all three models.

**Observed covariates (Model 3 only):**
- $w_r$ — binary wet indicator per race (from weather metadata).
- $\pi_{d,r}$ — per-driver-race pit-stop duration, normalised per season (z-scored,
  zero-imputed for non-pitting drivers, winsorised at 99th percentile).

---

## Section 3 — Model (~2.5 pages)

**Goal:** Motivate the Plackett-Luce likelihood, justify design decisions, present all three
model tiers with plate diagrams and generative stories, and describe the inference strategy.

### 3.1 Background: From TrueSkill to Plackett-Luce

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

**The sum-to-zero constraint:** The performance equation $p = s_d + c_k$ is
unidentified up to a global shift — adding a constant to all $s_d$ and subtracting
it from all $c_k$ leaves the likelihood unchanged. We fix this by requiring
$\sum_k c_k = 0$ via reparameterisation: sample $K-1$ free constructor values
$c_{raw}$ and derive $c_K = -\sum c_{raw}$. The variational guide never samples
$c$ directly. The constraint holds exactly throughout training.

### 3.2 Design Decisions

**Grid position excluded:** Qualifying performance is itself an expression of the
latent variables we are estimating — driver skill and constructor quality jointly
determine where a car starts on the grid. Grid position is downstream of the skill
signal, not independent of it. Conditioning on it as a covariate would partial out
this information, producing skill estimates that reflect only race-execution ability
rather than total driver quality. We leave joint qualifying-race modelling as
future work.

**DNF handling:** Including mechanical DNFs at last place in the Plackett-Luce
ranking creates an asymmetric bias: a high-performing constructor receives a larger
gradient penalty for a mechanical failure than a low-performing one, because the
model is more surprised by a last-place Mercedes than a last-place HRT. Models 1
and 2 therefore exclude mechanical DNFs from the ranking entirely.

### 3.3 Three-Tier Complexity Ladder

The three models share the Plackett-Luce likelihood and sum-to-zero constraint.
They differ in the richness of the latent structure and the covariates included.
Each tier answers a strictly harder scientific question than the one before it.

**[TABLE — side-by-side model comparison]**

| Feature | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Driver skill per season | ✗ (static) | ✓ (AR(1)) | ✓ (AR(1)) |
| Constructor perf per season | ✗ (static) | ✓ (AR(1)) | ✓ (AR(1)) |
| Circuit effects ($e_c$) | ✗ | ✗ | ✓ |
| Global wet weather ($\beta_w$) | ✗ | ✗ | ✓ |
| Driver wet skill ($\delta_d$) | ✗ | ✗ | ✓ |
| Pit-stop covariate ($\beta_\pi$) | ✗ | ✗ | ✓ |
| Observed covariates | None | None | $w_r$, $\pi_{d,r}$ |
| Mechanical DNF handling | Excluded | Excluded | Excluded |
| Number of latent variables | 93 | 1,344 | 1,454 |
| Inference method | SVI + NUTS | SVI | SVI |
| SVI steps | 3,000 | 5,000 | 5,000 |

---

#### 3.3.1 Model 1 — Baseline (Static)

**Plate diagram: Figure 1.** Single plate over races; driver and constructor nodes
are global (no temporal index).

**Generative story:** For each driver $d$, sample a fixed skill $s_d \sim \mathcal{N}(0, 1)$.
For each constructor $k$, sample $c_k$ from a $K-1$ dimensional Normal subject to
the sum-to-zero constraint. For each race $r$, compute the performance of each entrant
as $p_{d,r} = s_d + c_{k(d,r)}$ and draw the finishing order $\pi_r$ from
Plackett-Luce$(p_r)$. Mechanical DNFs are omitted from the ranking.

**Performance equation:**
```
p_{d,r} = s_d + c_{k(d,r)}
```

**Priors:**
- $s_d \sim \mathcal{N}(0, 1)$
- $c_{raw} \sim \mathcal{N}(0, 1)^{K-1}$ with $c_K = -\sum c_{raw}$

**Motivation:** The simplest possible instantiation of the skill-separation problem.
Serves as an identifiability proof of concept — if inference cannot recover sensible
driver and constructor rankings from 286 races, the Plackett-Luce signal is insufficient
for skill separation. Its fundamental limitation is that it averages skill over 14 seasons,
conflating car-era dominance (e.g. Mercedes 2014–2021) with driver ability. Piastri and
Norris rank #1–2 in the static model not because they are the greatest drivers, but
because their short careers (2023–2024) have been in a strong McLaren.

---

#### 3.3.2 Model 2 — Extended (Temporal)

**Plate diagram: Figure 2.** Adds a season plate around driver and constructor nodes.
Arrows connect $s_{d,t-1} \to s_{d,t}$ (and symmetrically for constructors).

**Generative story:** At season $t=0$, each driver draws an initial skill
$s_{d,0} \sim \mathcal{N}(0, \sigma_s)$ and each constructor draws an initial
performance $c_{k,0} \sim \mathcal{N}(0, \sigma_c)$ subject to sum-to-zero. For each
subsequent season $t > 0$, driver skill evolves via a random walk:
$s_{d,t} = s_{d,t-1} + \varepsilon_{d,t}$ where $\varepsilon_{d,t} \sim \mathcal{N}(0, \gamma_s)$.
Constructor performance evolves symmetrically with innovation variance $\gamma_c$.
For each race $r$ in season $t(r)$, performance is
$p_{d,r} = s_{d,t(r)} + c_{k(d,r),t(r)}$ and the finishing order $\pi_r$ is drawn
from Plackett-Luce$(p_r)$. Mechanical DNFs are excluded.

**Performance equation:**
```
p_{d,r} = s_{d,t(r)} + c_{k(d,r),t(r)}
```

**Priors:**
- $s_{d,0} \sim \mathcal{N}(0, \sigma_s)$, $\sigma_s = 1.0$
- $c_{k,0}$ from $K-1$ dimensional Normal, subject to sum-to-zero, $\sigma_c = 1.0$
- Driver innovation: $\varepsilon_{d,t} \sim \mathcal{N}(0, \gamma_s)$, $\gamma_s = 0.3$
  (drivers change gradually)
- Constructor innovation: $\varepsilon_{k,t} \sim \mathcal{N}(0, \gamma_c)$, $\gamma_c = 0.5$
  (regulation changes can cause overnight step-changes in car performance)

**Implementation:** AR(1) walks are implemented via cumulative sums of sampled
innovation vectors rather than a recursive `pyro.sample` loop, keeping the latent
space fully vectorised (2 sample sites per walk, not $D \times T$).

**Motivation:** Answers a single clean question: do skills change over time?
It recovers regulation-era step-changes in constructor performance that the static
model completely misses — Mercedes' hybrid-era dominance (2014–2021), Red Bull's
ground-effect surge (2022–2023), and McLaren's Honda-era collapse (2015–2018) all
emerge purely from race finishing orders.

---

#### 3.3.3 Model 3 — Full

**Plate diagram: Figure 3.** Extends the Model 2 plate with circuit nodes and observed
covariates ($w_r$, $\pi_{d,r}$) feeding into the performance equation.

**Generative story:** Model 3 extends the temporal generative process (Section 3.3.2)
with four additional sources of variance, each sampled independently:

1. For each circuit $c$, draw a track-specific effect $e_c \sim \mathcal{N}(0, \sigma_e)$.
2. Draw a global wet-weather coefficient $\beta_w \sim \mathcal{N}(0, 0.5)$.
3. For each driver $d$, draw a wet-weather interaction $\delta_d \sim \mathcal{N}(0, 0.3)$.
4. Draw a pit-stop coefficient $\beta_\pi \sim \mathcal{N}(0, 0.3)$.

For each race $r$, the per-driver performance is the sum of the temporal skill terms,
circuit effect, wet-weather effects (global $\beta_w \cdot w_r$ and driver-specific
$\delta_d \cdot w_r$), and the pit-stop adjustment $\beta_\pi \cdot \pi_{d,r}$.
The finishing order is drawn from Plackett-Luce$(p_r)$.

**Full performance equation:**
$$p_{d,r} = s_{d,t(r)} + c_{k(d,r),t(r)} + e_{circ(r)} + \beta_w w_r + \delta_d w_r + \beta_\pi \pi_{d,r}$$

**Priors:**
- $\sigma_e = 0.5$, $\sigma_{\beta_w} = \sigma_{\beta_\pi} = \sigma_{\delta} = 0.3$
- AR(1) innovation scales and initial skill priors as in Model 2

**Motivation — the four additions:**

1. **Circuit effects** $e_c \sim \mathcal{N}(0, \sigma_e)$: per-circuit latent effects absorb
   track-specific biases independent of car and driver. Circuits like Monaco or
   Monza impose fundamentally different demands, and without circuit effects,
   constructor skill estimates would be confounded by which circuits each team
   happened to race well at.

2. **Global wet-weather coefficient** $\beta_w \sim \mathcal{N}(0, 0.5)$: tests whether
   rain shifts all drivers' performance equally.

3. **Driver wet-weather interaction** $\delta_d \cdot w_r$: a driver-specific wet-weather
   skill modifier that activates only in wet races. Crucially, this is a
   multiplicative interaction with the rain indicator $w_r$, not an additive term —
   $\delta_d$ on its own would affect all races, which is wrong. $\beta_w$ captures the
   average wet-weather effect across all drivers; $\delta_d$ captures each driver's
   deviation from that average.

4. **Pit-stop covariate** $\beta_\pi \cdot \pi_{d,r}$: normalised pit-stop duration enters as
   a fixed observed covariate. Conditioning on it allows $c_k$ to be interpreted as
   pure car pace — operational execution is attributed to a separate coefficient.
   $\pi_{d,r}$ is z-scored per season with zero-imputation for non-pitting drivers
   and winsorisation at the 99th percentile to clamp extreme outliers.

### 3.4 Inference

**Stochastic Variational Inference (SVI) — all three models:**
The posterior $p(s, c \mid \text{data})$ is intractable in closed form. SVI approximates it
by choosing a parameterised family of distributions $q_\phi(s, c)$ — the *guide* —
and optimising its parameters $\phi$ to minimise the KL divergence from the true
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
This is a known limitation of factorised guides discussed in Section 5.

**[FIGURE 6: SVI vs NUTS scatter + synthetic recovery]**

**Synthetic data recovery:**
Before running on real data, we verify the model is correctly implemented using
ancestral sampling: ground-truth skills are fixed, synthetic finishing orders
generated by sampling from the generative model, and inference is run on the
synthetic data. If the model is correct, inference should recover the known
ground-truth values. Relative ordering is recovered near-perfectly (r = 0.999,
panel b). Absolute magnitudes show prior shrinkage toward zero — expected
behaviour since the Plackett-Luce likelihood is shift-invariant and the Normal
prior is the only anchor on the absolute scale.

**Prior rationale:**
$\sigma_s = \sigma_c = 1.0$ — weakly informative. A prior predictive check (ancestral
sampling without conditioning on data) confirms that these priors produce win
rates in the realistic 20–80% range, neither deterministic nor random. AR(1)
innovation scales: $\gamma_s = 0.3$ (drivers can shift ≈0.3 performance units per
season), $\gamma_c = 0.5$ (constructors can shift more — regulation changes can
cause overnight step-changes in car performance).

---

## Section 4 — Results (~2 pages)

**Goal:** Present the findings in order of strength: constructors first (best result),
then drivers, then Model 3 scalars.

### 4.1 Can We Separate Driver from Car? (Model 1)

Brief: static rankings are plausible for top drivers (Verstappen top 3, Hamilton
top 6) but Vettel ranks near #50 due to his long career averaging both dominant Red
Bull years and uncompetitive backmarker stints. Piastri and Norris rank #1–2 —
2023–2024 drivers in the best car of that era. This motivates Model 2.

**[FIGURE 5: Static vs temporal driver comparison]**

The scatter shows drivers above the diagonal were underrated by the static model
(Nico Rosberg — competed only 2011–2016, when Mercedes was not yet dominant) and
drivers below were overrated (Piastri, Norris — benefited from the 2024 McLaren).

### 4.2 How Do Skills Evolve? (Model 2)

**Constructor trajectories** are the strongest result in the paper.

**[FIGURE 4: Annotated constructor trajectories]**

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

### 4.3 What Else Can We Learn? (Model 3)

**[FIGURE 7: β_w and β_π posterior densities]**

- **β_w ≈ −0.03 (σ = 0.51):** The posterior on the global wet-weather coefficient
  is centred near zero and spans the entire prior range. Wet conditions produce no
  consistent global shift in performance — any wet-weather signal is driver-specific.
- **β_π ≈ 0.02 (σ = 0.03):** Centred near zero with small uncertainty. After correcting
  data artefacts in the pit-stop covariate (zero-imputation for non-pitting drivers
  and winsorisation of extreme outliers), pit-stop duration shows no detectable effect
  on race performance. The previously observed +0.26 was an artefact of the zero-
  duration confound (early DNFs with zero pit time systematically finishing last) and
  extreme-value inflation in the z-score normalisation. This null result is reported
  as a finding: within-season relative pit-stop timing does not predict race outcomes.
- **Wet-weather specialists (δ_d):** Posterior means are small and uncertainty is
  high for most drivers (~30 wet races out of 286 provide limited signal). The model
  cannot confidently identify individual wet-weather effects; this is reported as a
  limitation rather than a finding.

---

## Section 5 — Discussion & Conclusion (~0.6 pages)

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
- Richer variational family (LKJ prior on constructor covariance) to reduce mean-field bias
- Circuit-weather interaction term to disentangle Spa-type confounding

**Conclusion:**
We present a three-tier Bayesian PGM for F1 skill separation, grounded in TrueSkill's
generative framework but using the Plackett-Luce likelihood for tractable inference.
The temporal model recovers historically accurate constructor performance trajectories
— Mercedes' hybrid-era dominance, Red Bull's 2022–2023 surge, McLaren's Honda
collapse — purely from race finishing orders. Driver skill rankings are credible for
long-career drivers and correctly reflect uncertainty for short-career ones.
