import logging
import os

import numpy as np
import pandas as pd
import pyro
import pyro.optim
import torch
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO

from models.pgm_backend.model_baseline import BaselineModel

logger = logging.getLogger(__name__)


def train_svi(model, dataset, n_steps=3000, lr=0.01, log_every=100):
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(n_steps):
        loss = svi.step(
            driver_idx=dataset.driver_idx,
            cons_idx=dataset.cons_idx,
            race_lengths=dataset.race_lengths,
        )
        losses.append(loss)
        if step % log_every == 0 or step == n_steps - 1:
            print(f"Step {step:5d}  ELBO loss: {loss:.2f}")

    return losses


def extract_svi_posterior(model) -> dict[str, torch.Tensor]:
    s_loc = pyro.param("s_loc").detach().clone()
    s_scale = pyro.param("s_scale").detach().clone()
    c_loc_raw = pyro.param("c_loc").detach().clone()
    c_scale = pyro.param("c_scale").detach().clone()

    c_loc = torch.cat([c_loc_raw, -c_loc_raw.sum(dim=0, keepdim=True)])

    return {
        "s_loc": s_loc,
        "s_scale": s_scale,
        "c_loc": c_loc,
        "c_scale": c_scale,
    }


def run_nuts(model, dataset, num_warmup=500, num_samples=500):
    pyro.clear_param_store()
    kernel = NUTS(model.model)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_warmup)
    mcmc.run(
        driver_idx=dataset.driver_idx,
        cons_idx=dataset.cons_idx,
        race_lengths=dataset.race_lengths,
    )
    return mcmc


def compare_svi_nuts(svi_posterior, mcmc, dataset) -> pd.DataFrame:
    samples = mcmc.get_samples()
    s_samples = samples["s"]
    c_raw_samples = samples["c_raw"]

    c_samples = torch.cat(
        [c_raw_samples, -c_raw_samples.sum(dim=-1, keepdim=True)],
        dim=-1,
    )

    s_nuts_mean = s_samples.mean(0)
    s_nuts_std = s_samples.std(0)
    c_nuts_mean = c_samples.mean(0)
    c_nuts_std = c_samples.std(0)

    s_svi = svi_posterior["s_loc"]
    s_disc = torch.abs(s_svi - s_nuts_mean) / s_nuts_std

    c_svi = svi_posterior["c_loc"]
    c_disc = torch.abs(c_svi - c_nuts_mean) / c_nuts_std

    diag = mcmc.diagnostics()
    s_rhat = diag["s"]["r_hat"]
    c_raw_rhat = diag["c_raw"]["r_hat"]

    all_rhats = np.concatenate([s_rhat, c_raw_rhat])
    bad = all_rhats[all_rhats >= 1.05]
    if len(bad) > 0:
        logger.warning(
            f"R-hat >= 1.05 for {len(bad)} / {len(all_rhats)} latents"
        )

    D = len(s_svi)
    K = len(c_svi)

    rows = []

    for i in range(D):
        rows.append({
            "entity_type": "driver",
            "entity_id": dataset.driver_map[i],
            "entity_name": "",
            "svi_mean": s_svi[i].item(),
            "nuts_mean": s_nuts_mean[i].item(),
            "nuts_std": s_nuts_std[i].item(),
            "discrepancy": s_disc[i].item(),
            "r_hat": float(s_rhat[i]),
        })

    for i in range(K):
        rhat_val = float(c_raw_rhat[i]) if i < K - 1 else float("nan")
        rows.append({
            "entity_type": "constructor",
            "entity_id": dataset.constructor_map[i],
            "entity_name": "",
            "svi_mean": c_svi[i].item(),
            "nuts_mean": c_nuts_mean[i].item(),
            "nuts_std": c_nuts_std[i].item(),
            "discrepancy": c_disc[i].item(),
            "r_hat": rhat_val,
        })

    df = pd.DataFrame(rows)

    os.makedirs("outputs/pgm_model", exist_ok=True)
    df.to_csv("outputs/pgm_model/nuts_vs_svi_comparison.csv", index=False)

    return df
