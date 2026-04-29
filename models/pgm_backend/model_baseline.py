import pyro
import pyro.distributions as dist
import torch

from models.pgm_backend.likelihood import plackett_luce_log_prob

SIGMA_S = 1.0
SIGMA_C = 1.0


class BaselineModel:
    def __init__(self, n_drivers: int, n_constructors: int):
        self.D = n_drivers
        self.K = n_constructors

    def model(self, driver_idx, cons_idx, race_lengths):
        D, K = self.D, self.K

        s = pyro.sample(
            "s",
            dist.Normal(0.0, SIGMA_S).expand([D]).to_event(1),
        )

        c_raw = pyro.sample(
            "c_raw",
            dist.Normal(0.0, SIGMA_C).expand([K - 1]).to_event(1),
        )
        c = torch.cat([c_raw, -c_raw.sum(dim=0, keepdim=True)])

        p = s[driver_idx] + c[cons_idx]

        log_prob = plackett_luce_log_prob(p, race_lengths)
        pyro.factor("race_obs", log_prob)

    def guide(self, driver_idx, cons_idx, race_lengths):
        D, K = self.D, self.K

        s_loc = pyro.param("s_loc", torch.zeros(D))
        s_scale = pyro.param(
            "s_scale",
            torch.ones(D),
            constraint=dist.constraints.positive,
        )
        pyro.sample("s", dist.Normal(s_loc, s_scale).to_event(1))

        c_loc = pyro.param("c_loc", torch.zeros(K - 1))
        c_scale = pyro.param(
            "c_scale",
            torch.ones(K - 1),
            constraint=dist.constraints.positive,
        )
        pyro.sample("c_raw", dist.Normal(c_loc, c_scale).to_event(1))
