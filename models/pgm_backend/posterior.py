"""Unified posterior extractor that converts Pyro param-store tensors
into a tidy DataFrame for a given model.
"""

import pandas as pd
import pyro
import torch


def _expand_raw_c_scale(raw_scale):
    """Append K-th constructor scale as the mean of the K-1 raw scales."""
    mean = raw_scale.mean()
    return torch.cat([raw_scale, mean.unsqueeze(0)])


def _build_temporal_rows(rows, dataset):
    """Append rows for driver, constructor, circuit, and beta_w (shared by extended and full)."""
    T = dataset.n_seasons
    D = dataset.n_drivers
    K = dataset.n_constructors
    C = dataset.n_circuits

    s0_loc = pyro.param("s0_loc").detach().clone()
    s_innov_loc = pyro.param("s_innov_loc").detach().clone()
    s_loc = torch.cat(
        [s0_loc.unsqueeze(0), s0_loc.unsqueeze(0) + s_innov_loc.cumsum(0)], dim=0
    )

    s0_scale = pyro.param("s0_scale").detach().clone()
    s_innov_scale = pyro.param("s_innov_scale").detach().clone()
    s_scale = torch.cat([s0_scale.unsqueeze(0), s_innov_scale], dim=0)

    for t in range(T):
        for i in range(D):
            rows.append({
                "entity_type": "driver",
                "entity_id": dataset.driver_map[i],
                "entity_name": "",
                "season": t,
                "mu": s_loc[t, i].item(),
                "sigma": s_scale[t, i].item(),
            })

    c0_raw_loc = pyro.param("c0_raw_loc").detach().clone()
    c_innov_loc = pyro.param("c_innov_loc").detach().clone()
    c_raw_loc = torch.cat(
        [c0_raw_loc.unsqueeze(0), c0_raw_loc.unsqueeze(0) + c_innov_loc.cumsum(0)], dim=0
    )
    c_loc = torch.cat([c_raw_loc, -c_raw_loc.sum(dim=1, keepdim=True)], dim=1)

    c0_raw_scale = pyro.param("c0_raw_scale").detach().clone()
    c_innov_scale = pyro.param("c_innov_scale").detach().clone()
    c_raw_scale = torch.cat([c0_raw_scale.unsqueeze(0), c_innov_scale], dim=0)
    c_scale = torch.cat(
        [c_raw_scale, c_raw_scale.mean(dim=1, keepdim=True)], dim=1
    )

    for t in range(T):
        for i in range(K):
            rows.append({
                "entity_type": "constructor",
                "entity_id": dataset.constructor_map[i],
                "entity_name": "",
                "season": t,
                "mu": c_loc[t, i].item(),
                "sigma": c_scale[t, i].item(),
            })

    e_circ_loc = pyro.param("e_circ_loc").detach().clone()
    e_circ_scale = pyro.param("e_circ_scale").detach().clone()
    for i in range(C):
        rows.append({
            "entity_type": "circuit",
            "entity_id": -1,
            "entity_name": "",
            "season": "all",
            "mu": e_circ_loc[i].item(),
            "sigma": e_circ_scale[i].item(),
        })

    beta_w_loc = pyro.param("beta_w_loc").detach().clone()
    beta_w_scale = pyro.param("beta_w_scale").detach().clone()
    rows.append({
        "entity_type": "global",
        "entity_id": -1,
        "entity_name": "beta_w",
        "season": "all",
        "mu": beta_w_loc.item(),
        "sigma": beta_w_scale.item(),
    })


def extract_posterior(model_name, dataset):
    """Return a tidy DataFrame of posterior means and stds for a trained model.

    Parameters
    ----------
    model_name : {"baseline", "extended", "full"}
        Which model's param store to read.
    dataset : F1RankingDataset
        Needed for driver_map, constructor_map, n_seasons, etc.

    Returns
    -------
    pd.DataFrame with columns: entity_type, entity_id, entity_name, season, mu, sigma
    """
    D = dataset.n_drivers
    K = dataset.n_constructors
    rows = []

    if model_name == "baseline":
        s_loc = pyro.param("s_loc").detach().clone()
        s_scale = pyro.param("s_scale").detach().clone()

        c_loc_raw = pyro.param("c_loc").detach().clone()
        c_scale_raw = pyro.param("c_scale").detach().clone()

        c_loc = torch.cat([c_loc_raw, -c_loc_raw.sum(dim=0, keepdim=True)])
        c_scale = _expand_raw_c_scale(c_scale_raw)

        for i in range(D):
            rows.append({
                "entity_type": "driver",
                "entity_id": dataset.driver_map[i],
                "entity_name": "",
                "season": "all",
                "mu": s_loc[i].item(),
                "sigma": s_scale[i].item(),
            })
        for i in range(K):
            rows.append({
                "entity_type": "constructor",
                "entity_id": dataset.constructor_map[i],
                "entity_name": "",
                "season": "all",
                "mu": c_loc[i].item(),
                "sigma": c_scale[i].item(),
            })

    elif model_name == "extended":
        _build_temporal_rows(rows, dataset)

    elif model_name == "full":
        _build_temporal_rows(rows, dataset)

        delta_d_loc = pyro.param("delta_d_loc").detach().clone()
        delta_d_scale = pyro.param("delta_d_scale").detach().clone()
        for i in range(D):
            rows.append({
                "entity_type": "driver",
                "entity_id": dataset.driver_map[i],
                "entity_name": "delta_d",
                "season": "all",
                "mu": delta_d_loc[i].item(),
                "sigma": delta_d_scale[i].item(),
            })

        beta_pi_loc = pyro.param("beta_pi_loc").detach().clone()
        beta_pi_scale = pyro.param("beta_pi_scale").detach().clone()
        rows.append({
            "entity_type": "global",
            "entity_id": -1,
            "entity_name": "beta_pi",
            "season": "all",
            "mu": beta_pi_loc.item(),
            "sigma": beta_pi_scale.item(),
        })

        alpha_rel_loc = pyro.param("alpha_rel_loc").detach().clone()
        alpha_rel_scale = pyro.param("alpha_rel_scale").detach().clone()
        rows.append({
            "entity_type": "global",
            "entity_id": -1,
            "entity_name": "alpha_rel",
            "season": "all",
            "mu": alpha_rel_loc.item(),
            "sigma": alpha_rel_scale.item(),
        })

    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            "Expected 'baseline', 'extended', or 'full'."
        )

    return pd.DataFrame(rows)
