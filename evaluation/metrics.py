r"""
Prediction evaluation metrics for race finishing‑order forecasts.

All functions operate on per‑race inputs:

* ``y_true`` — 1‑D array of actual finishing positions (1 = winner).
* ``y_pred`` — 1‑D array of predicted skill scores; **higher** means
  stronger / more likely to win.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of driver pairs where the higher‑rated driver finished ahead.

    For *N* drivers in a race there are N·(N−1)/2 unordered pairs.  The
    model scores 1 for a pair when ``y_pred[i] > y_pred[j] ⟹ y_true[i] <
    y_true[j]`` (higher skill → better finish) and 0 otherwise.

    Args:
        y_true: Actual finish positions (lower = better).
        y_pred: Predicted skill scores (higher = better).

    Returns:
        Accuracy in [0, 1].
    """
    n = len(y_true)
    if n < 2:
        return 1.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            # TrueSkill: higher mu → better.  True rank: lower position is better.
            pred_i_better = y_pred[i] > y_pred[j]
            actual_i_better = y_true[i] < y_true[j]
            if pred_i_better == actual_i_better:
                correct += 1
    return correct / total if total > 0 else 1.0


def top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 1) -> float:
    """Does the highest‑rated driver finish in the top‑*k*?

    Args:
        y_true: Actual finish positions.
        y_pred: Predicted skill scores.
        k: Position threshold (1 = win, 3 = podium).

    Returns:
        1.0 if the best‑rated driver finished ≤ k, else 0.0.
    """
    if len(y_pred) == 0:
        return 0.0
    best_idx = int(np.argmax(y_pred))
    return 1.0 if y_true[best_idx] <= k else 0.0


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted skill order and actual
    finish order.

    Args:
        y_true: Actual finish positions.
        y_pred: Predicted skill scores.

    Returns:
        ρ in [−1, 1]; positive = model ranks correlate with reality.
    """
    if len(y_true) < 2:
        return 0.0
    # Convert higher skill → predicted rank (1 = best)
    pred_ranks = stats.rankdata(-y_pred)  # negative so largest → rank 1
    true_ranks = stats.rankdata(y_true)   # smallest position → rank 1
    rho, _ = stats.spearmanr(pred_ranks, true_ranks)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def mean_reciprocal_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MRR — reciprocal of the predicted rank of the actual winner.

    Rank all drivers by descending ``y_pred``; find the position of the
    actual winner (``y_true == 1``) in that ordering; return 1 / rank.

    Args:
        y_true: Actual finish positions.
        y_pred: Predicted skill scores.

    Returns:
        MRR in (0, 1]; 1 = winner was top‑rated.
    """
    winner_mask = y_true == 1
    if not winner_mask.any():
        # Multiple drivers could have position 1 (tie); take first
        winner_mask = y_true == y_true.min()
    winner_idx = int(np.where(winner_mask)[0][0])
    # Rank: 1 = highest y_pred
    pred_rank = int(np.sum(y_pred > y_pred[winner_idx])) + 1
    return 1.0 / pred_rank


def mse_position(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between assigned rank and actual finish position.

    Drivers are ordered by descending ``y_pred`` and assigned ranks
    1, 2, …, N.  MSE is computed between the assigned rank and the true
    ``positionOrder``.

    Args:
        y_true: Actual finish positions.
        y_pred: Predicted skill scores.

    Returns:
        Mean squared error (≥ 0).
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    # Predicted rank: 1 = highest y_pred
    pred_ranks = np.empty(n, dtype=int)
    order = np.argsort(-y_pred)
    for rank, idx in enumerate(order, 1):
        pred_ranks[idx] = rank
    return float(np.mean((pred_ranks - y_true) ** 2))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def compute_race_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute all per‑race metrics at once.

    Args:
        y_true: Actual finish positions.
        y_pred: Predicted skill scores.

    Returns:
        Dictionary of metric name → value.
    """
    return {
        "pairwise_accuracy": pairwise_accuracy(y_true, y_pred),
        "top_1_accuracy": top_k_accuracy(y_true, y_pred, k=1),
        "top_3_accuracy": top_k_accuracy(y_true, y_pred, k=3),
        "top_5_accuracy": top_k_accuracy(y_true, y_pred, k=5),
        "spearman_rho": spearman_rho(y_true, y_pred),
        "mrr": mean_reciprocal_rank(y_true, y_pred),
        "mse_position": mse_position(y_true, y_pred),
    }


def compute_fold_metrics(
    race_predictions: List[Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """Aggregate per‑race metric dicts across a fold.

    For accuracy‑based metrics (pairwise, top‑k) we compute the weighted
    average (more competitors = more pairs).  For Spearman ρ and MRR we
    use simple means.

    Args:
        race_predictions: List of dicts with keys ``y_true``, ``y_pred``.

    Returns:
        Dict of metric name → mean value.
    """
    collected: Dict[str, List[float]] = {
        "pairwise_accuracy": [],
        "top_1_accuracy": [],
        "top_3_accuracy": [],
        "top_5_accuracy": [],
        "spearman_rho": [],
        "mrr": [],
        "mse_position": [],
    }
    pair_weights: List[int] = []

    for pred in race_predictions:
        m = compute_race_metrics(pred["y_true"], pred["y_pred"])
        for k in collected:
            collected[k].append(m[k])
        n = len(pred["y_true"])
        pair_weights.append(n * (n - 1) / 2 if n >= 2 else 1)

    result: Dict[str, float] = {}
    for k in collected:
        if k == "pairwise_accuracy":
            # Weighted by number of pairs
            total_weight = sum(pair_weights)
            result[k] = (
                sum(v * w for v, w in zip(collected[k], pair_weights)) / total_weight
                if total_weight > 0
                else 0.0
            )
        else:
            result[k] = float(np.mean(collected[k]))
    return result
