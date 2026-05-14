# Model Architecture — In Depth

## What are we even doing?

Formula 1 results always confound two things: **how good the driver is** and
**how good the car is**. When Hamilton wins a race, was it because he drove
brilliantly, or because Mercedes built the fastest car? You can never observe
these separately — you only see the finishing order.

Our goal is to **disentangle** them. We build a probabilistic model that looks
at 14 seasons of race results (who finished where) and infers two hidden
("latent") quantities:

- **Driver skill** ($s$): how good each driver is, independent of their car
- **Constructor performance** ($c$): how good each car is, independent of who drives it

We know this separation is possible because:
- **Teammates drive the same car.** If Hamilton consistently beats Bottas, the
  difference must be driver skill — the car is identical.
- **Drivers switch teams.** When a driver moves from team A to team B, their
  change in results tells us about the car difference between A and B.

We build three models of increasing complexity. Each answers a progressively
harder question by adding one new idea to the previous model.

---

## The data at a glance

| What | How many |
|---|---|
| Seasons | 14 (2011–2024) |
| Races | 286 |
| Drivers | 77 |
| Constructors (teams) | 17 (after merging rebrands — see below) |
| Circuits (tracks) | 35 |
| Total driver-race entries | 5980 |
| Mechanical DNFs | 460 (7.7%) |

**Constructor rebranding:** F1 teams change names over time. Force India became
Racing Point, then Aston Martin — but it's the same factory and staff. We merge
these into a single "constructor" so the model can track continuity across
seasons. Without this, the model would think a team disappeared and a new one
appeared, breaking the temporal link.

**DNFs (Did Not Finish):** Not all retirements are the same.
- *Mechanical DNF*: engine blows up, gearbox fails. The car broke — not the
  driver's fault. The driver shouldn't be penalised for this.
- *Driver-fault DNF*: crashed into a wall, spun off. The driver made a mistake.
  Both driver and constructor should be penalised.

This distinction matters because if you just put all DNFs at the bottom of the
ranking, a Mercedes engine failure hurts Mercedes more than a Haas engine failure
hurts Haas — because the model is more "surprised" that a fast car finished last.
That's wrong. So we handle mechanical DNFs differently across the models (see below).

---

## How the math works (shared by all three models)

### The Plackett-Luce likelihood

Imagine a race as a sequence of eliminations. First, the winner is chosen from
all drivers — but drivers with higher performance scores are more likely to win.
Then the winner is removed, and second place is chosen from the remaining
drivers. Then third, and so on.

The probability of a driver being chosen at each step is proportional to
$e^{p}$, where $p$ is that driver's performance score. A driver with $p = 2.0$
is $e^{2} \approx 7.4$ times more "selectable" than a driver with $p = 0$
(who has weight $e^{0} = 1$).

The full likelihood for an observed finishing order $\pi = [A, B, C]$ is:

$$P(A \text{ wins}, B \text{ 2nd}, C \text{ 3rd}) =
\frac{e^{p_A}}{e^{p_A} + e^{p_B} + e^{p_C}} \times
\frac{e^{p_B}}{e^{p_B} + e^{p_C}} \times
\frac{e^{p_C}}{e^{p_C}}$$

The first term: A wins among all three. The second: B beats C for second place.
The third: C is the only one left (probability 1).

In log form (what the code actually computes):

$$\log P = p_A - \log(e^{p_A} + e^{p_B} + e^{p_C}) + p_B - \log(e^{p_B} + e^{p_C})$$

This is the **exact** probability of the observed order. It is not an
approximation — unlike decomposing the race into independent A-vs-B, A-vs-C,
B-vs-C comparisons (the "pairwise" approach used in older work).

### The sum-to-zero constraint (identifiability)

Here's the problem: if we have $p_{d,r} = s_d + c_k$, we could add +1.0 to
every driver's skill AND subtract −1.0 from every constructor's performance,
and all the $p$ values would stay exactly the same. The likelihood cannot tell
these two explanations apart — the model has a "flat direction" in parameter
space.

We fix this by requiring that constructor performances sum to zero:
$\sum c_k = 0$. Now constructors are measured relative to the field average —
positive $c_k$ means "above average car," negative means "below average car."
There is no ambiguity about whether the overall level comes from drivers or
constructors.

**How it's implemented:** We only sample $K{-}1$ "free" constructor values
(the first 16 teams). The 17th team is automatically set to the negative sum
of the other 16, guaranteeing the constraint is perfectly satisfied. The
variational guide (see "How we train these models" below) never touches $c$
directly — it only works with the free parameters $c_{raw}$.

---

## How we train these models

### The inference problem

We have a model that says "given driver skills $s$ and constructor
performances $c$, here's the probability of observing this finishing order."
But we want the reverse: given the finishing order, what are the most likely
values of $s$ and $c$?

This is a **posterior inference** problem: find $P(s, c \mid \text{data})$.
For our models, this posterior has no closed-form solution — it's too complex.
We need approximate inference.

### Stochastic Variational Inference (SVI)

Instead of trying to compute the exact posterior, SVI picks a simpler family
of distributions — in our case, independent Normal (Gaussian) distributions
for every latent variable — and tweaks their means and standard deviations to
be as close as possible to the true posterior.

Think of it this way: the true posterior is a complex shape in a
high-dimensional space. SVI fits a simpler, tractable shape around it. The
"closeness" is measured by the **ELBO** (Evidence Lower BOund), which you'll
see decreasing during training. Lower ELBO = better fit.

Our guide is **mean-field**: every latent variable gets its own independent
Normal distribution, ignoring correlations between them. This is a
simplification — in reality, driver skills and constructor performances are
correlated — but it makes training fast and the results are good enough for
ranking purposes. The trade-off is that SVI underestimates uncertainty (the
posterior standard deviations are narrower than the true posterior).

**Optimizer:** ClippedAdam with learning rate 0.01 and gradient clipping at 10.
Gradient clipping prevents the optimizer from taking huge steps when the
likelihood surface is steep, which can happen early in training.

### NUTS (No-U-Turn Sampler) — validation only

NUTS is a Markov Chain Monte Carlo (MCMC) method. Unlike SVI, it is
**asymptotically exact** — given enough samples, it converges to the true
posterior. The trade-off is that it's much slower and scales poorly with
dimension.

We run NUTS only on Model 1 (93 parameters). If NUTS and SVI produce similar
results for Model 1, we can trust SVI for the larger models where NUTS is
impractical. This is our **inference validation** step.

---

## Model 1 — Baseline (Static Skills)

### Plate diagram

```
                         σ_s                σ_c
                          │                  │
                          ▼                  ▼
         ┌──────────────────────────────────────────────┐
         │                 D drivers                     │
         │  s_d ~ N(0, σ_s)          (s_d)              │
         └──────────────────────────────────────────────┘
                          │
                          │  s_d
                          ▼
         ┌──────────────────────────────────────────────┐
         │                 K constructors               │  sum-to-zero
         │  c_raw ~ N(0, σ_c)    (c_k)  ◄──────────────  Σ c_k = 0
         └──────────────────────────────────────────────┘
                          │
                          │  c_k(d,r)
                          ▼
                    p_{d,r} = s_d + c_{k(d,r)}
                          │
                          ▼
         ┌──────────────────────────────────────────────┐
         │              R races (N_dr per race)         │
         │  [π_r]  ◄── Plackett-Luce(softmax(p_dr))    │
         └──────────────────────────────────────────────┘

   LEGEND:  ( ) latent    [ ] observed    ┌──┐ plate (repetition)
```

- **$D$ plate:** One driver skill $s_d$ per driver. Repeated 77 times.
- **$K$ plate:** One constructor performance $c_k$ per constructor. Repeated 17 times. Constrained so all 17 sum to zero.
- **$R$ plate:** For each race, the observed finishing order $[\pi_r]$ is generated from the performance scores via Plackett-Luce.
- **No arrow between $s$ and $c$:** They are independent in the prior (both $\mathcal{N}(0,1)$). They only become coupled through the likelihood — the data tells us how they combine.

### What it answers

**Can we separate driver from constructor at all?** This is the simplest possible
model. If this doesn't work, nothing else will.

### What it assumes

Every driver has one fixed skill level across all 14 seasons. Every constructor
has one fixed performance level. A driver who raced from 2011–2024 gets the
same skill score in every race they entered — no improvement, no decline.

### The variables

| Variable | How many | What it means | Prior belief |
|---|---|---|---|
| $s_d$ | 77 (one per driver) | How good driver $d$ is. Positive = above average. | We start assuming all drivers are average ($s_d \sim \mathcal{N}(0, 1)$). The data pushes strong drivers up and weak drivers down. |
| $c_k$ | 17 (one per constructor) | How good constructor $k$'s car is. Positive = faster than average car. | Same as drivers ($\mathcal{N}(0, 1)$), but constrained so the 17 values sum to zero. |

"Prior $\mathcal{N}(0, 1)$" means "before seeing any data, we think the most
likely value is 0 (average), with a standard deviation of 1." A skill of +2
means "two standard deviations above the mean" — roughly a top-2% driver.
A skill of −1 means below average but not terrible.

### The performance equation

$$p_{d,r} = s_d + c_{k(d,r)}$$

For each driver $d$ in each race $r$, their performance is just their driver
skill plus their constructor's car performance. That's it. No other factors.

### How it's trained

- **SVI:** 3000 steps. Takes about 2 minutes on an M1 Pro.
- **NUTS:** 500 warmup + 500 samples. Takes about 9 minutes. Used to validate
  that SVI is giving reasonable answers (R-hat < 1.05 for all parameters,
  confirming the chains converged).

### What it can and can't do

**Can:** Rank drivers and constructors. The top 5 in the static model are
plausible — Verstappen, Hamilton, Vettel, Alonso all appear near the top.
Mercedes is the top constructor, followed by Red Bull and Ferrari. The
sum-to-zero constraint holds perfectly ($\sum c_k \approx 10^{-6}$).

**Cannot:** Distinguish driver skill from car-era dominance. Oscar Piastri
ranks #1 in the static model — not because he's the greatest driver ever,
but because his entire short career (2023–2024) has been in a strong McLaren.
The model can't tell if McLaren's results come from the car or the driver,
because both are static. This is the core limitation that motivates Model 2.

---

## Model 2 — Temporal

### Plate diagram

```
         σ_s                           γ_s
          │                             │
          ▼                             ▼
    s_{d,0} ~ N(0, σ_s)    ε_{d,t} ~ N(0, γ_s)
          │                             │
          └──────────┬──────────────────┘
                     │  cumsum: s_t = s_0 + Σ_{τ=1}^t ε_τ
                     ▼
   ┌─────────────────────────────────────────────────────────┐
   │               T=14 seasons × D=77 drivers              │
   │                                                         │
   │  (s_{d,0}) ──► (s_{d,1}) ──► (s_{d,2}) ──► ... ──► (s_{d,13})  │
   │     AR(1) random walk: s_{d,t} = s_{d,t-1} + ε_{d,t}  │
   └─────────────────────────────────────────────────────────┘
                     │  s_{d, t(r)}
                     ▼
               p = s_{d,t(r)} + c_{k,t(r)}
                     ▲
                     │  c_{k, t(r)}
                     │
   ┌─────────────────────────────────────────────────────────┐
   │               T=14 seasons × K=17 constructors         │
   │                                                         │
   │  (c_{k,0}) ──► (c_{k,1}) ──► (c_{k,2}) ──► ... ──► (c_{k,13})  │
   │     AR(1) random walk, sum-to-zero per season          │
   │     γ_c = 0.5  (larger than drivers — reg changes)     │
   └─────────────────────────────────────────────────────────┘
          ▲                             ▲
          │                             │
          ▼                             ▼
         σ_c                           γ_c

                     │
                     ▼
   ┌─────────────────────────────────────────────────────────┐
   │                   R=286 races                           │
   │  [π_r]  ◄── Plackett-Luce(softmax(p))                  │
   └─────────────────────────────────────────────────────────┘
```

- **Two AR(1) walks:** Driver skill and constructor performance now evolve per season. Each depends only on the previous season's value plus a random innovation.
- **Innovation scales:** $\gamma_s = 0.3$ means drivers change gradually. $\gamma_c = 0.5$ means constructors can shift faster — regulation changes are larger shocks than individual driver development.
- **Cumsum implementation:** Not drawn as $T$ separate sample sites. In code, the entire $(T,D)$ trajectory is built from 2 Pyro samples (initial state + innovations tensor) with a single `cumsum`.
- **Sum-to-zero per season:** Each season $t$ has its own $\sum_k c_{k,t} = 0$ constraint. A constructor's performance is always measured relative to the field average in that season.

### What it answers

**Do skills change over time?** Drivers improve with experience and decline
with age. Constructor performance shifts dramatically with regulation changes.
The static model averages these changes away; the temporal model tracks them.

### What it adds (from Model 1)

Instead of one skill per driver for all 14 seasons, we now have **14 skills
per driver** (one per season), linked by an AR(1) random walk. AR(1) means
"this season's skill is last season's skill plus some random change."
Nothing else is added — no circuit effects, no weather, no covariates. This is
a pure test of whether temporal dynamics matter.

### The variables

| Variable | How many | What it means | Prior belief |
|---|---|---|---|
| $s_{d,0}$ | 77 | Driver skill in 2011 (season $t=0$) | $\mathcal{N}(0, 1)$ — same as Model 1's prior for the starting point |
| $\varepsilon_{d,t}$ | 13 × 77 = 1001 | How much driver $d$'s skill changed from season $t{-}1$ to $t$ (innovation) | $\mathcal{N}(0, 0.3^2)$ — small changes. A driver can shift about ±0.3 per season |
| $c_{raw,k,0}$ | 16 | Constructor performance free parameters in 2011 | $\mathcal{N}(0, 1)$ |
| $\eta_{k,t}$ | 13 × 16 = 208 | Constructor innovation per season | $\mathcal{N}(0, 0.5^2)$ — larger than driver innovations. Regulation changes can cause step-changes |

Derived from these: $s_{d,t} = s_{d,0} + \varepsilon_{d,1} + \varepsilon_{d,2} + \ldots + \varepsilon_{d,t}$.
The skill in season $t$ is the starting skill plus all accumulated changes.
Same derivation for constructors, with the sum-to-zero constraint re-applied
separately in each season.

### How the AR(1) is implemented (cumsum trick)

A naive implementation would be a loop:

```python
for t in range(1, T):
    s[t] = pyro.sample(f"s_{t}", Normal(s[t-1], gamma_s))
```

This creates $D \times T = 1078$ individual Pyro sample sites — one for every
driver-season combination. The computational graph becomes enormous.

Instead we use **vectorised sampling + cumulative sum**:

```python
s0 = sample("s0", Normal(0, sigma_s), shape=(D,))           # 1 sample site
s_innov = sample("s_innov", Normal(0, gamma_s), shape=(T-1, D))  # 1 sample site
s = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + s_innov.cumsum(dim=0)])  # (T, D)
```

Two sample sites total — not 1078. The model remains tractable for SVI.

### The performance equation

$$p_{d,r} = s_{d, t(r)} + c_{k(d,r), t(r)}$$

Same form as Model 1, but $s$ and $c$ now depend on **which season** the race
happens in. A driver in 2014 uses $s_{d,3}$ (season index 3 = 2014); the same
driver in 2024 uses $s_{d,13}$.

### How it's trained

SVI only, 5000 steps. About 1 minute. NUTS is not run — the latent space has
grown to ~1344 dimensions, and NUTS would take days.

### What it can do that Model 1 can't

The temporal model recovers **regulation-era transitions** purely from race
outcomes, with no external labels:
- **Mercedes** rises sharply in 2014 (hybrid engine regulations introduced),
  peaks around 2019, and declines from 2022 (ground-effect regulations).
- **Red Bull** dominates 2011–2013 (Vettel era), stays competitive through the
  hybrid era, and surges in 2022–2023 (Verstappen era).
- **McLaren** collapses in 2015 — exactly when the disastrous Honda engine
  partnership began — and recovers gradually to become the top constructor
  by 2024.

Driver rankings now reflect **current form** rather than career averages.
Verstappen ranks #1 in 2024 (reflecting his recent dominance). Vettel drops
from #5 (static) to outside the top 15 in 2024 (he retired after 2022).
The AR(1) innovation scale ($\gamma_s = 0.3$) allows enough flexibility to
track career arcs while still sharing information across seasons.

---

## Model 3 — Full

### Plate diagram

```
  LIKELIHOOD:

    σ_s    γ_s    σ_c    γ_c    σ_e      σ_δ
     │      │      │      │      │        │
     ▼      ▼      ▼      ▼      ▼        ▼
  (s_{d,0}) (ε_d) (c_{k,0}) (η_k) (e_c)  (δ_d)   (β_w)   (β_π)
     │      │      │      │      │        │        │       │       │
     └──┬───┘      └──┬───┘      │        │        │       │       │
        │ cumsum      │ cumsum    │        │        │       │       │
        ▼             ▼           │        │        │       │       │
   ┌─────────────────────────┐    │        │        │       │       │
   │  (s_{d,t}) T×D          │    │        │        │       │       │
   │  AR(1) per driver       │    │        │        │       │       │
   └─────────────────────────┘    │        │        │       │       │
        │                         │        │        │       │       │
        │  s_{d,t(r)}             │        │        │       │       │
        ▼                         ▼        │        │       │       │
   ┌─────────────────────────┐  (e_c)     │        │       │       │
   │  (c_{k,t}) T×K          │   │        │        │       │       │
   │  AR(1), Σc=0 per t      │   │        │        │       │       │
   └─────────────────────────┘   │        │        │       │       │
        │                        │        │        │       │       │
        │  c_{k,t(r)}            │        │        │       │       │
        │                        │        │        │       │       │
        ▼                        ▼        │        │       │       │
        └──────────┬─────────────┘        │        │       │       │
                   │                      │        │       │       │
                   │    e_{circ(r)}       │        │       │       │
                   ▼                      ▼        ▼       ▼       │
                   └──────────────────────┴────────┴───────┘       │
                   │                                               │
                   │  p = s + c + e_circ + β_w·w_r + δ_d·w_r + β_π·π
                   │                                               │
                   ▼                                               │
   ┌─────────────────────────────────────────────────────────┐     │
   │              R=286 races (ranking entries only)         │     │
   │              [w_r] = wet indicator (observed)           │     │
   │              [π_{d,r}] = pit duration (observed)        │     │
   │                                                         │     │
   │  [ordering π_r]  ◄── Plackett-Luce(softmax(p))         │     │
   └─────────────────────────────────────────────────────────┘     │
                                                                    │

```

### How to read this diagram

- **Top row:** Priors. Each latent variable has a prior distribution (specified by its hyperparameters $\sigma_s$, $\gamma_s$, etc.). The priors encode our beliefs before seeing data.

- **Middle rows (AR walks):** The temporal structure from Model 2 — driver skills $s_{d,t}$ and constructor performances $c_{k,t}$ evolve as random walks across 14 seasons.

- **Bottom-left (performance):** The performance equation $p_{d,r}$ combines all latent contributions (driver, constructor, circuit, weather, pit) into a single score per driver per race. The Plackett-Luce likelihood converts these scores into the probability of the observed finishing order.

- **Observed covariates:** $w_r$ (is it raining?) and $\pi_{d,r}$ (normalised pit duration) are read from the data, not inferred. They act as switches — $w_r = 0$ (dry race) zeroes out both $\beta_w$ and $\delta_d$, making them irrelevant for that race.



### What it answers

**What additional structure exists beyond temporal dynamics?** After accounting
for driver skill changing over time and constructor performance changing over
time, what else matters? We add four things, each testing a distinct hypothesis.

### What it adds (from Model 2)

1. **Circuit effects ($e_c$):** Different tracks are harder or easier
   independent of car and driver. Monaco demands completely different skills
   than Monza. If we don't account for this, constructor estimates get
   confounded by which tracks each team happened to race well at.

2. **Global wet-weather coefficient ($\beta_w$):** Does rain shift all
   drivers' performance equally? A positive $\beta_w$ would mean "drivers with
   high $\delta_d$ benefit in wet races" (relative to their dry performance).
   Actually, $\beta_w$ captures the *average* wet effect — the driver-specific
   part is $\delta_d$ (see below).

   Wait — let me be more precise. $\beta_w$ is the performance shift that
   applies to EVERY driver when it rains. Think of it as "wet races are
   fundamentally different from dry races, and on average everyone is
   shifted by $\beta_w$." $\delta_d$ is then each driver's *deviation* from
   that average. A driver with $\delta_d = +0.5$ earns +0.5 more than the
   field average in wet conditions.

3. **Driver wet-weather interaction ($\delta_d \cdot w_r$):** Some drivers are
   known as "rain masters" — they excel in wet conditions. $\delta_d$ is
   the driver-specific modifier that **only** activates when $w_r = 1$
   (race is wet). Crucially, this is a *multiplication*, not an addition.
   If we just added $\delta_d$ to all races, dry or wet, $\delta_d$ would
   just be absorbed into $s_d$ and become meaningless.

4. **Pit-stop covariate ($\beta_\pi \cdot \pi_{d,r}$):** Pit-stop execution
   varies. Some crews are fast, some are slow. By including pit-stop time as
   an observed covariate (we already know it), we let the model attribute
   some performance variance to pit execution rather than car pace. This
   gives a cleaner interpretation of $c_k$ as "pure car pace."

   $\pi_{d,r}$ is normalised per season: z-scored (subtract season mean,
   divide by season std), with zero-imputation for drivers who never pitted
   (early DNFs), and winsorisation at the 99th percentile to clamp extreme
   data artefacts (e.g., pit times of 61 minutes from timing system glitches).

### The variables

**Latent** (inferred by the model):

| Variable | How many | Prior | What it means |
|---|---|---|---|
| All of Model 2 | ~1344 | — | Temporal driver skills and constructor performances |
| $e_c$ | 35 | $\mathcal{N}(0, 0.5^2)$ | Per-circuit effect. A difficult track (e.g., Spa) might have a large negative $e_c$, meaning performances are lower there independent of who's racing. |
| $\beta_w$ | 1 | $\mathcal{N}(0, 0.5^2)$ | Global wet-weather shift. Does rain change all performances by a constant amount? |
| $\delta_d$ | 77 | $\mathcal{N}(0, 0.5^2)$ | Driver deviation from $\beta_w$. Positive = better than average in rain. Negative = worse than average in rain. |
| $\beta_\pi$ | 1 | $\mathcal{N}(0, 0.5^2)$ | Effect of pit-stop time. A positive value means "longer pit stops correlate with better results" (which turned out to be an artefact — see results). |

**Observed** (read from the data, not inferred):

| Variable | What it is | How it enters |
|---|---|---|
| $w_r$ | Binary: 1 = wet race, 0 = dry | Multiplied by $\beta_w$ and $\delta_d$ |
| $\pi_{d,r}$ | Normalised pit duration per driver-race | Multiplied by $\beta_\pi$ |

### The performance equation

$$p_{d,r} = \underbrace{s_{d,t(r)} + c_{k(d,r),t(r)}}_{\text{temporal (Model 2)}} + \underbrace{e_{circ(r)}}_{\text{circuit}} + \underbrace{\beta_w w_r}_{\text{global wet}} + \underbrace{\delta_d w_r}_{\text{driver wet}} + \underbrace{\beta_\pi \pi_{d,r}}_{\text{pit stop}}$$

### How it's trained

SVI only, 5000 steps. About 2 minutes. ~1454 latent dimensions.

### What we found (key results)

| Parameter | Posterior mean ± std | Interpretation |
|---|---|---|
| $\beta_w$ | −0.03 ± 0.51 | Centred at zero, spans the full prior range. **No global wet-weather effect.** Rain doesn't shift all performances equally — any wet-weather signal is driver-specific. |
| $\beta_\pi$ | +0.02 ± 0.03 | Tightly estimated near zero. **Pit-stop duration has no detectable effect** on race outcomes. The previously reported +0.26 was a data artefact (early DNFs with zero pit time were systematically at the bottom, creating a spurious correlation). |
| $\delta_d$ top 5 | Verstappen +0.62, Rosberg +0.52, Norris +0.44, Hülkenberg +0.42, Hamilton +0.33 | Top wet-weather drivers by model inference. Differs from historical "rain master" reputations (Alonso is 6th, Webber is negative). Only ~30 wet races exist — uncertainty is high. |

---

## Side-by-side model comparison

| What's in the model | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Driver skill per season | ✗ (static) | ✓ (AR(1)) | ✓ (AR(1)) |
| Constructor perf per season | ✗ (static) | ✓ (AR(1)) | ✓ (AR(1)) |
| Circuit effects ($e_c$) | ✗ | ✗ | ✓ |
| Global wet weather ($\beta_w$) | ✗ | ✗ | ✓ |
| Driver wet skill ($\delta_d$) | ✗ | ✗ | ✓ |
| Pit-stop covariate ($\beta_\pi$) | ✗ | ✗ | ✓ |
| Observed covariates | None | None | $w_r$, $\pi_{d,r}$ |
| Mechanical DNF handling | Excluded | Excluded | Excluded |
| Number of latent variables | 93 | 1344 | 1454 |
| Inference method | SVI + NUTS | SVI | SVI |
| Training time | ~2 + 9 min | ~1 min | ~2 min |

---

## Architecture constraints (things we never do)

These are hard rules — any code that violates them is wrong:

1. **Plackett-Luce likelihood only.** We do not use pairwise probit or Bradley-Terry
   approximations. The Plackett-Luce is the exact joint probability of the finishing order.
2. **Sum-to-zero via reparameterisation.** Sample $K{-}1$ free constructor values, derive the
   $K$-th. The guide never samples $c$ directly. The constraint holds exactly.
3. **Mechanical DNFs excluded from ranking.** Including them creates an
   asymmetric bias where good constructors are penalised more for failures.
   Mechanical DNFs are excluded from the Plackett-Luce ranking in all three models.
4. **AR(1) via cumsum of innovations.** No recursive `pyro.sample` loop. Two vectorised
   sample sites per temporal variable (initial state + innovations), not $D \times T$.
5. **NUTS on Model 1 only.** NUTS does not scale to the ~1344 latent dimensions of
   Models 2 and 3.
6. **No grid position.** Grid position (qualifying result) is a blocking variable — it
   sits on the causal path from skill to result. Including it would prevent the model
   from learning anything about skill from race outcomes.
7. **Constructor rebranding merges applied before indexing.** Racing Point and Aston Martin
   are mapped to Force India; Alpine to Renault; AlphaTauri and Racing Bulls to Toro Rosso;
   Alfa Romeo to Sauber. This preserves AR(1) continuity across rebranding epochs.
