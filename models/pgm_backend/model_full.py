import pyro
import pyro.distributions as dist
import torch

from models.pgm_backend.likelihood import plackett_luce_log_prob

SIGMA_S = 1.0
SIGMA_C = 1.0
GAMMA_S = 0.3
GAMMA_C = 0.5
SIGMA_E = 0.5
SIGMA_DELTA = 0.5


class FullModel:
    def __init__(self, n_drivers: int, n_constructors: int, n_seasons: int, n_circuits: int):
        self.D = n_drivers
        self.K = n_constructors
        self.T = n_seasons
        self.C = n_circuits

    def model(
        self,
        driver_idx,
        cons_idx,
        season_idx,
        circuit_idx,
        race_idx,
        wet,
        race_lengths,
        pit_norm,
        is_mech,
        cons_idx_all,
        season_idx_all,
    ):
        D, K, T, C = self.D, self.K, self.T, self.C

        # ---- Driver AR(1) skills ----
        s0 = pyro.sample("s0", dist.Normal(0.0, SIGMA_S).expand([D]).to_event(1))
        s_innov = pyro.sample(
            "s_innov", dist.Normal(0.0, GAMMA_S).expand([T - 1, D]).to_event(2)
        )
        s = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + s_innov.cumsum(0)], dim=0)  # (T, D)

        # ---- Constructor AR(1) skills (sum-to-zero per season) ----
        c0_raw = pyro.sample("c0_raw", dist.Normal(0.0, SIGMA_C).expand([K - 1]).to_event(1))
        c_innov = pyro.sample(
            "c_innov", dist.Normal(0.0, GAMMA_C).expand([T - 1, K - 1]).to_event(2)
        )
        c_raw = torch.cat([c0_raw.unsqueeze(0), c0_raw.unsqueeze(0) + c_innov.cumsum(0)], dim=0)  # (T, K-1)
        c = torch.cat([c_raw, -c_raw.sum(dim=1, keepdim=True)], dim=1)  # (T, K)

        # ---- Circuit effects ----
        e_circ = pyro.sample("e_circ", dist.Normal(0.0, SIGMA_E).expand([C]).to_event(1))

        # ---- Global weather coefficient ----
        beta_w = pyro.sample("beta_w", dist.Normal(0.0, 0.5))

        # ---- Driver wet-weather skill modifier ----
        delta_d = pyro.sample("delta_d", dist.Normal(0.0, SIGMA_DELTA).expand([D]).to_event(1))

        # ---- Pit-stop coefficient ----
        beta_pi = pyro.sample("beta_pi", dist.Normal(0.0, 0.5))

        # ---- Reliability intercept ----
        alpha_rel = pyro.sample("alpha_rel", dist.Normal(0.0, 1.0))

        # ---- Performance (ranking entries only) ----
        p = (
            s[season_idx, driver_idx]
            + c[season_idx, cons_idx]
            + e_circ[circuit_idx]
            + beta_w * wet[race_idx]
            + delta_d[driver_idx] * wet[race_idx]
            + beta_pi * pit_norm
        )

        log_prob = plackett_luce_log_prob(p, race_lengths)
        pyro.factor("race_obs", log_prob)

        # ---- Mechanical DNF reliability term (all rows) ----
        mech_prob = torch.sigmoid(-alpha_rel - c[season_idx_all, cons_idx_all])
        pyro.factor("reliability", dist.Bernoulli(mech_prob).log_prob(is_mech.float()).sum())

    def guide(
        self,
        driver_idx,
        cons_idx,
        season_idx,
        circuit_idx,
        race_idx,
        wet,
        race_lengths,
        pit_norm,
        is_mech,
        cons_idx_all,
        season_idx_all,
    ):
        D, K, T, C = self.D, self.K, self.T, self.C

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

        e_circ_loc = pyro.param("e_circ_loc", torch.zeros(C))
        e_circ_scale = pyro.param(
            "e_circ_scale", torch.ones(C), constraint=dist.constraints.positive
        )
        pyro.sample("e_circ", dist.Normal(e_circ_loc, e_circ_scale).to_event(1))

        beta_w_loc = pyro.param("beta_w_loc", torch.tensor(0.0))
        beta_w_scale = pyro.param(
            "beta_w_scale", torch.tensor(1.0), constraint=dist.constraints.positive
        )
        pyro.sample("beta_w", dist.Normal(beta_w_loc, beta_w_scale))

        delta_d_loc = pyro.param("delta_d_loc", torch.zeros(D))
        delta_d_scale = pyro.param(
            "delta_d_scale", torch.ones(D), constraint=dist.constraints.positive
        )
        pyro.sample("delta_d", dist.Normal(delta_d_loc, delta_d_scale).to_event(1))

        beta_pi_loc = pyro.param("beta_pi_loc", torch.tensor(0.0))
        beta_pi_scale = pyro.param(
            "beta_pi_scale", torch.tensor(1.0), constraint=dist.constraints.positive
        )
        pyro.sample("beta_pi", dist.Normal(beta_pi_loc, beta_pi_scale))

        alpha_rel_loc = pyro.param("alpha_rel_loc", torch.tensor(0.0))
        alpha_rel_scale = pyro.param(
            "alpha_rel_scale", torch.tensor(1.0), constraint=dist.constraints.positive
        )
        pyro.sample("alpha_rel", dist.Normal(alpha_rel_loc, alpha_rel_scale))
