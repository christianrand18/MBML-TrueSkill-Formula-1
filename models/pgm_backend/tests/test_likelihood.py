import math
import torch

from models.pgm_backend.likelihood import plackett_luce_log_prob


def test_plackett_luce_hand_check():
    """
    3 drivers, performances [2.0, 1.0, 0.0], race_lengths [3].
    Expected log-prob ~ -0.7209 (tolerance 1e-3).
    """
    performances = torch.tensor([2.0, 1.0, 0.0])
    race_lengths = torch.tensor([3])

    expected = (
        math.log(math.exp(2) / (math.exp(2) + math.exp(1) + 1))
        + math.log(math.exp(1) / (math.exp(1) + 1))
        + 0.0
    )

    actual = plackett_luce_log_prob(performances, race_lengths).item()

    assert abs(actual - expected) < 1e-3, f"Expected ~{expected:.4f}, got {actual:.4f}"


def test_plackett_luce_log_prob_nonpos():
    """
    Log-prob must be <= 0 for any valid input.
    Test with 20 random races of random lengths.
    """
    torch.manual_seed(42)
    R = 20
    max_L = 10
    min_L = 2

    race_lengths = torch.randint(min_L, max_L + 1, (R,))
    N_total = race_lengths.sum().item()

    performances = torch.randn(N_total)

    log_prob = plackett_luce_log_prob(performances, race_lengths).item()

    assert log_prob <= 0, f"Log-prob {log_prob:.6f} > 0"
