r"""
Pyro Bayesian skill‑rating model for Formula 1.

Implements a pairwise ranking (Thurstone‑Mosteller / probit) model with:

* **Static variant** — one latent skill per driver / constructor for the
  entire 2011–2024 window.
* **Temporal variant** — per‑season skills linked by a random‑walk prior.

Both variants accept grid position as a covariate and support
weather‑dependent performance noise scaling.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.infer
import pyro.optim
import torch

from models.pyro_backend.data_preparation import PairwiseDataset

logger = logging.getLogger("f1_pyro.model")


class F1SkillModel:
    """Bayesian PGM for F1 skill rating via pairwise ranking.

    Args:
        dataset: Pre‑built pairwise comparison dataset.
        temporal: If *True*, use per‑season skills with random‑walk prior.
        use_weather_noise: If *True*, allow performance noise to scale
            with wet‑weather conditions.
    """

    def __init__(
        self,
        dataset: PairwiseDataset,
        temporal: bool = True,
        use_weather_noise: bool = False,
    ) -> None:
        self.temporal = temporal
        self.use_weather_noise = use_weather_noise
        self._n_d = dataset.n_drivers
        self._n_c = dataset.n_constructors
        self._n_s = dataset.n_seasons
        self._n_pairs = dataset.n_pairs
        self._dataset = dataset

        # Index of 'is_wet' in the weather feature vector (for noise scaling)
        self._wet_idx = 4  # after grid, temp, wind, humidity

    # ==================================================================
    # Model
    # ==================================================================

    def model(self, dataset: Optional[PairwiseDataset] = None) -> None:
        """Generative model for pairwise ranking data.

        Args:
            dataset: Pairwise data.  If *None*, uses the dataset stored
                at initialisation.
        """
        ds = dataset or self._dataset
        n_d, n_c, n_s = self._n_d, self._n_c, self._n_s

        # ---- Global parameters -----------------------------------------
        beta_grid = pyro.sample(
            "beta_grid", dist.Normal(0.0, 1.0)
        )
        beta_perf = pyro.param(
            "beta_perf",
            torch.tensor(25.0 / 6.0),
            constraint=constraints.positive,
        )

        beta_wet = torch.tensor(0.0)  # default: no weather effect
        if self.use_weather_noise:
            beta_wet = pyro.sample(
                "beta_wet", dist.Normal(0.0, 0.5)
            )

        # ---- Driver skills ---------------------------------------------
        driver_skills = self._sample_entity_skills(
            "driver", n_d, n_s
        )

        # ---- Constructor skills ----------------------------------------
        constructor_skills = self._sample_entity_skills(
            "constructor", n_c, n_s
        )

        # ---- Pairwise ranking likelihood -------------------------------
        with pyro.plate("pairs", self._n_pairs, subsample_size=1024) as ind:
            d_i = ds.driver_i[ind]
            d_j = ds.driver_j[ind]
            c_i = ds.cons_i[ind]
            c_j = ds.cons_j[ind]
            s = ds.season[ind]
            grid_i = ds.feats_i[ind, 0]
            grid_j = ds.feats_j[ind, 0]

            # Skill sums
            if self.temporal:
                skill_i = (
                    driver_skills[s, d_i] + constructor_skills[s, c_i]
                )
                skill_j = (
                    driver_skills[s, d_j] + constructor_skills[s, c_j]
                )
            else:
                skill_i = driver_skills[d_i] + constructor_skills[c_i]
                skill_j = driver_skills[d_j] + constructor_skills[c_j]

            diff = skill_i - skill_j + beta_grid * (grid_i - grid_j)

            # Weather‑dependent noise
            if self.use_weather_noise:
                wet = ds.weather[ind, self._wet_idx]
                noise_scale = beta_perf * (1.0 + beta_wet * wet)
            else:
                noise_scale = beta_perf

            # Probit win probability
            z = diff / (math.sqrt(2.0) * noise_scale)
            prob = dist.Normal(0.0, 1.0).cdf(z)
            prob = torch.clamp(prob, 1e-6, 1.0 - 1e-6)

            pyro.sample(
                "obs", dist.Bernoulli(probs=prob), obs=torch.ones_like(prob)
            )

    # ==================================================================
    # Guide
    # ==================================================================

    def guide(self, dataset: Optional[PairwiseDataset] = None) -> None:
        """Mean‑field variational guide — independent Normals for all
        latent variables.

        Args:
            dataset: Ignored (provided for SVI compatibility).
        """
        n_d, n_c, n_s = self._n_d, self._n_c, self._n_s

        # Global parameters
        pyro.sample(
            "beta_grid",
            dist.Normal(
                pyro.param("beta_grid_loc", torch.tensor(-0.1)),
                pyro.param(
                    "beta_grid_scale",
                    torch.tensor(0.1),
                    constraint=constraints.positive,
                ),
            ),
        )

        if self.use_weather_noise:
            pyro.sample(
                "beta_wet",
                dist.Normal(
                    pyro.param("beta_wet_loc", torch.tensor(0.0)),
                    pyro.param(
                        "beta_wet_scale_p",
                        torch.tensor(0.1),
                        constraint=constraints.positive,
                    ),
                ),
            )

        # Entity skills
        self._guide_entity_skills("driver", n_d, n_s)
        self._guide_entity_skills("constructor", n_c, n_s)

    # ==================================================================
    # Skill sampling (model)
    # ==================================================================

    def _sample_entity_skills(
        self, prefix: str, n_entities: int, n_seasons: int
    ) -> torch.Tensor:
        """Return a tensor of entity skills.

        **Static:** shape ``(n_entities,)``, each ~ N(0, 10).

        **Temporal:** shape ``(n_seasons, n_entities)``, where
        ``skills[0, e] ~ N(0, 10)`` and
        ``skills[s, e] ~ N(skills[s-1, e], tau)`` with learned *tau*.
        """
        if not self.temporal:
            with pyro.plate(f"{prefix}s", n_entities):
                return pyro.sample(
                    f"{prefix}_skill",
                    dist.Normal(0.0, 10.0),
                )

        # -- Temporal (random walk) -------------------------------------
        tau_name = f"tau_{prefix}"
        tau = pyro.param(
            tau_name,
            torch.tensor(0.5),
            constraint=constraints.positive,
        )

        tau_scaled = tau * 10.0  # scale with prior std

        skills = torch.zeros(n_seasons, n_entities)
        for e in range(n_entities):
            skills[0, e] = pyro.sample(
                f"{prefix}_{e}_s0",
                dist.Normal(0.0, 10.0),
            )
            for s in range(1, n_seasons):
                skills[s, e] = pyro.sample(
                    f"{prefix}_{e}_s{s}",
                    dist.Normal(skills[s - 1, e], tau_scaled),
                )
        return skills

    # ==================================================================
    # Skill guide
    # ==================================================================

    def _guide_entity_skills(
        self, prefix: str, n_entities: int, n_seasons: int
    ) -> None:
        """Variational guide for entity skills (mean‑field Normals)."""
        if not self.temporal:
            loc = pyro.param(
                f"{prefix}_loc",
                torch.zeros(n_entities),
            )
            scale = pyro.param(
                f"{prefix}_scale",
                torch.ones(n_entities) * 5.0,
                constraint=constraints.positive,
            )
            with pyro.plate(f"{prefix}s", n_entities):
                pyro.sample(
                    f"{prefix}_skill",
                    dist.Normal(loc, scale),
                )
            return

        # -- Temporal guide ---------------------------------------------
        for e in range(n_entities):
            for s in range(n_seasons):
                loc = pyro.param(
                    f"{prefix}_{e}_s{s}_loc",
                    torch.tensor(0.0),
                )
                scale = pyro.param(
                    f"{prefix}_{e}_s{s}_scale",
                    torch.tensor(5.0),
                    constraint=constraints.positive,
                )
                pyro.sample(
                    f"{prefix}_{e}_s{s}",
                    dist.Normal(loc, scale),
                )


# ==================================================================
# Training helper
# ==================================================================


def train_svi(
    model: F1SkillModel,
    n_steps: int = 5000,
    lr: float = 0.005,
    log_every: int = 500,
) -> pyro.infer.SVI:
    """Run Stochastic Variational Inference.

    Args:
        model: Configured ``F1SkillModel``.
        n_steps: Number of SVI iterations.
        lr: Adam learning rate.
        log_every: Log ELBO every *N* steps.

    Returns:
        Trained ``SVI`` object (contains the optimised parameter store).
    """
    optimizer = pyro.optim.ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = pyro.infer.SVI(model.model, model.guide, optimizer, loss=pyro.infer.Trace_ELBO())

    pyro.clear_param_store()
    loss_history: list = []

    for step in range(1, n_steps + 1):
        loss = svi.step()
        loss_history.append(loss)

        if step % log_every == 0 or step == 1:
            logger.info(
                "  SVI step %5d / %d  ELBO = %10.2f",
                step, n_steps, -loss,
            )

    # Store the final loss
    model._loss_history = loss_history
    return svi
