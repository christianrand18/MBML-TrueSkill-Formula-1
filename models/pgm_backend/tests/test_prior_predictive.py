import torch
import pytest


def test_prior_predictive():
    torch.manual_seed(42)

    D, K = 20, 10
    N_DRAWS = 100
    SIGMA_S, SIGMA_C = 1.0, 1.0

    cons_assignment = torch.arange(D) % K  # fixed round-robin

    fastest_won_list = []
    gaps = []

    for _ in range(N_DRAWS):
        s = torch.randn(D) * SIGMA_S
        c_raw = torch.randn(K - 1) * SIGMA_C
        c = torch.cat([c_raw, -c_raw.sum(dim=0, keepdim=True)])
        p = s + c[cons_assignment]

        fastest_driver = p.argmax().item()
        gaps.append((p.max() - p.min()).item())

        # PL sequential draw
        remaining = list(range(D))
        order = []
        for _ in range(D):
            scores = p[torch.tensor(remaining)]
            probs = torch.softmax(scores, dim=0)
            pick = torch.multinomial(probs, 1).item()
            order.append(remaining[pick])
            remaining.pop(pick)

        fastest_won_list.append(order[0] == fastest_driver)

    win_rate = sum(fastest_won_list) / N_DRAWS
    mean_gap = sum(gaps) / N_DRAWS

    print(f"\nPrior predictive check ({N_DRAWS} draws, {D} drivers, {K} constructors):")
    print(f"  Prior-fastest driver win rate: {win_rate:.2f}")
    print(f"  Mean P1-P20 performance gap:   {mean_gap:.2f}")

    assert 0.20 <= win_rate <= 0.80, (
        f"Prior win rate {win_rate:.2f} outside [0.20, 0.80] — "
        "priors may be too flat or too sharp"
    )
    assert 1.0 <= mean_gap <= 5.0, (
        f"Mean P1-P20 performance gap {mean_gap:.2f} outside [1.0, 5.0]"
    )
