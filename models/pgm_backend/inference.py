import pyro
import pyro.optim
import torch
from pyro.infer import SVI, Trace_ELBO

from models.pgm_backend.model_baseline import BaselineModel


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
