import pyro
import pyro.distributions as dist
import torch

from models.pgm_backend.likelihood import plackett_luce_log_prob

SIGMA_S = 1.0
SIGMA_C = 1.0
GAMMA_S = 0.3
GAMMA_C = 0.5


class ExtendedModel:
    def __init__(self, n_drivers: int, n_constructors: int, n_seasons: int, n_circuits: int = 0):
        self.D = n_drivers
        self.K = n_constructors
        self.T = n_seasons

    def model(self, driver_idx, cons_idx, season_idx, race_lengths):
        D, K, T = self.D, self.K, self.T

        s0 = pyro.sample(
            "s0",
            dist.Normal(0.0, SIGMA_S).expand([D]).to_event(1),
        )
        s_innov = pyro.sample(
            "s_innov",
            dist.Normal(0.0, GAMMA_S).expand([T - 1, D]).to_event(2),
        )
        s = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + s_innov.cumsum(0)], dim=0)

        c0_raw = pyro.sample(
            "c0_raw",
            dist.Normal(0.0, SIGMA_C).expand([K - 1]).to_event(1),
        )
        c_innov = pyro.sample(
            "c_innov",
            dist.Normal(0.0, GAMMA_C).expand([T - 1, K - 1]).to_event(2),
        )
        c_raw = torch.cat([c0_raw.unsqueeze(0), c0_raw.unsqueeze(0) + c_innov.cumsum(0)], dim=0)
        c = torch.cat([c_raw, -c_raw.sum(dim=1, keepdim=True)], dim=1)

        p = s[season_idx, driver_idx] + c[season_idx, cons_idx]

        log_prob = plackett_luce_log_prob(p, race_lengths)
        pyro.factor("race_obs", log_prob)

    def guide(self, driver_idx, cons_idx, season_idx, race_lengths):
        D, K, T = self.D, self.K, self.T

        s0_loc = pyro.param("s0_loc", torch.zeros(D))
        s0_scale = pyro.param("s0_scale", torch.ones(D), constraint=dist.constraints.positive)
        pyro.sample("s0", dist.Normal(s0_loc, s0_scale).to_event(1))

        s_innov_loc = pyro.param("s_innov_loc", torch.zeros(T - 1, D))
        s_innov_scale = pyro.param(
            "s_innov_scale", torch.ones(T - 1, D), constraint=dist.constraints.positive
        )
        pyro.sample("s_innov", dist.Normal(s_innov_loc, s_innov_scale).to_event(2))

        c0_raw_loc = pyro.param("c0_raw_loc", torch.zeros(K - 1))
        c0_raw_scale = pyro.param(
            "c0_raw_scale", torch.ones(K - 1), constraint=dist.constraints.positive
        )
        pyro.sample("c0_raw", dist.Normal(c0_raw_loc, c0_raw_scale).to_event(1))

        c_innov_loc = pyro.param("c_innov_loc", torch.zeros(T - 1, K - 1))
        c_innov_scale = pyro.param(
            "c_innov_scale", torch.ones(T - 1, K - 1), constraint=dist.constraints.positive
        )
        pyro.sample("c_innov", dist.Normal(c_innov_loc, c_innov_scale).to_event(2))
