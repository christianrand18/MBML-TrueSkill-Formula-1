import types

import pyro
import torch

from models.pgm_backend.model_baseline import BaselineModel
from models.pgm_backend.inference import train_svi, extract_svi_posterior


def test_baseline_recovery():
    """
    Generate synthetic races from known parameters, fit Model 1 with SVI,
    and assert posterior means recover true values within +/-0.8.
    """
    torch.manual_seed(123)

    true_s = torch.tensor([2.5, 2.0, 1.5, 0.0, -1.5])
    true_c = torch.tensor([2.0, 0.5, -2.5])
    N_RACES = 50

    all_driver_idx = []
    all_cons_idx = []

    for _ in range(N_RACES):
        cons_assignment = torch.randint(0, 3, (5,))
        p = true_s + true_c[cons_assignment]

        remaining = list(range(5))
        order = []
        for _ in range(5):
            scores = p[torch.tensor(remaining)]
            probs = torch.softmax(scores, dim=0)
            pick = torch.multinomial(probs, 1).item()
            order.append(remaining[pick])
            remaining.pop(pick)

        all_driver_idx.extend(order)
        all_cons_idx.extend(cons_assignment[torch.tensor(order)].tolist())

    dataset = types.SimpleNamespace(
        driver_idx=torch.tensor(all_driver_idx, dtype=torch.long),
        cons_idx=torch.tensor(all_cons_idx, dtype=torch.long),
        race_lengths=torch.full((N_RACES,), 5, dtype=torch.long),
    )

    pyro.clear_param_store()

    model = BaselineModel(n_drivers=5, n_constructors=3)
    losses = train_svi(model, dataset, n_steps=3000, lr=0.01, log_every=500)
    posterior = extract_svi_posterior(model)

    assert len(losses) == 3000, f"Expected 3000 losses, got {len(losses)}"
    assert losses[-1] < losses[0], (
        f"ELBO did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}"
    )

    for d in range(5):
        err = abs(posterior["s_loc"][d].item() - true_s[d].item())
        assert err < 0.8, (
            f"Driver {d}: true={true_s[d]:.2f}, inferred={posterior['s_loc'][d]:.2f}, error={err:.3f}"
        )

    for k in range(3):
        err = abs(posterior["c_loc"][k].item() - true_c[k].item())
        assert err < 0.8, (
            f"Constructor {k}: true={true_c[k]:.2f}, inferred={posterior['c_loc'][k]:.2f}, error={err:.3f}"
        )

    assert abs(posterior["c_loc"].sum().item()) < 1e-4, (
        f"c_loc sum {posterior['c_loc'].sum().item():.6f} not zero"
    )

    inferred_order = torch.argsort(posterior["s_loc"])
    true_order = torch.argsort(true_s)
    assert torch.equal(inferred_order, true_order) or torch.equal(inferred_order.flip(0), true_order), (
        f"Ranking sign mismatch: inferred order {inferred_order.tolist()}, true order {true_order.tolist()}"
    )

    print("\nSynthetic recovery results:")
    for d in range(5):
        print(f"  Driver {d}: true={true_s[d]:.2f}, inferred={posterior['s_loc'][d]:.2f}")
    for k in range(3):
        print(f"  Constructor {k}: true={true_c[k]:.2f}, inferred={posterior['c_loc'][k]:.2f}")
