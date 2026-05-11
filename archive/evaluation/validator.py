r"""
Chronological cross‑validation for F1 skill‑rating models.

Splits the dataset by season boundaries, ensuring that training data
always precedes test data temporally.  For each fold every registered
predictor is trained on historical seasons and evaluated on a hold‑out
season.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from evaluation.baselines import SkillPredictor
from evaluation.metrics import compute_fold_metrics

logger = logging.getLogger("f1_evaluation.validator")


class ChronologicalValidator:
    """Time‑aware cross‑validation for race‑result predictors.

    Args:
        min_train_years: Minimum number of seasons required for training.
        test_window: Number of seasons to hold out per fold (default 1).

    Example:
        With ``min_train_years=4`` on data spanning 2011–2024:

        * Fold 1: train 2011–2014, test 2015
        * Fold 2: train 2011–2015, test 2016
        * …
        * Fold 10: train 2011–2023, test 2024
    """

    def __init__(
        self, min_train_years: int = 4, test_window: int = 1
    ) -> None:
        self._min_train_years = min_train_years
        self._test_window = test_window

    def generate_folds(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int]]:
        """Generate (train_df, test_df, test_year) splits.

        Args:
            df: Full race DataFrame with a ``year`` column.

        Returns:
            List of (train, test, test_year) tuples.
        """
        years = sorted(df["year"].unique())
        folds = []
        first_test_idx = self._min_train_years
        for i in range(first_test_idx, len(years)):
            test_year = int(years[i])
            train_years = [int(y) for y in years[:i]]
            train = df[df["year"].isin(train_years)]
            test = df[df["year"] == test_year]
            folds.append((train, test, test_year))
        return folds

    def evaluate_fold(
        self,
        model: SkillPredictor,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[str, Dict[str, float]]:
        """Train *model* on *train_df*, evaluate on *test_df*.

        Args:
            model: Predictor instance (already configured).
            train_df: Training races.
            test_df: Test races (one season).

        Returns:
            ``(model.name, metrics_dict)``.
        """
        logger.info("  Fitting %s on %d training races …", model.name, train_df["raceId"].nunique())
        model.fit(train_df)

        race_predictions: List[Dict[str, np.ndarray]] = []
        for race_id, race in test_df.groupby("raceId", sort=True):
            skills = model.predict_driver_skills(race)
            # Build ordered arrays aligned by driverId
            y_true_list: List[float] = []
            y_pred_list: List[float] = []
            for _, row in race.iterrows():
                d_id = int(row["driverId"])
                y_true_list.append(int(row["positionOrder"]))
                y_pred_list.append(skills.get(d_id, 0.0))
            race_predictions.append(
                {
                    "y_true": np.array(y_true_list),
                    "y_pred": np.array(y_pred_list),
                }
            )

        if not race_predictions:
            return model.name, {}

        metrics = compute_fold_metrics(race_predictions)
        return model.name, metrics

    def run(
        self,
        df: pd.DataFrame,
        models: Dict[str, SkillPredictor],
    ) -> pd.DataFrame:
        """Run chronological CV over all folds and all models.

        Args:
            df: Full race DataFrame with ``year``.
            models: ``{name: SkillPredictor}``.

        Returns:
            Long‑form DataFrame with columns: ``model``, ``fold_test_year``,
            ``pairwise_accuracy``, ``top_1_accuracy``, …, ``mse_position``.
        """
        folds = self.generate_folds(df)
        if not folds:
            logger.warning("No folds generated (need >=%d training years).", self._min_train_years)
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for train_df, test_df, test_year in folds:
            logger.info(
                "=== Fold: test year %d | %d test races ===",
                test_year,
                test_df["raceId"].nunique(),
            )
            for name, model in models.items():
                # Each model gets a fresh training fold
                m_name, metrics = self.evaluate_fold(model, train_df.copy(), test_df)
                row = {"model": m_name, "fold_test_year": test_year}
                row.update(metrics)
                rows.append(row)
                logger.info(
                    "    %-12s  pairwise_acc=%.3f  top1=%.3f  spearman=%.3f  mse=%.2f",
                    m_name,
                    metrics.get("pairwise_accuracy", 0),
                    metrics.get("top_1_accuracy", 0),
                    metrics.get("spearman_rho", 0),
                    metrics.get("mse_position", 0),
                )

        return pd.DataFrame(rows)
