r"""
F1 Skill Model Evaluation — Entry Point
=========================================

Runs chronological cross‑validation over all five skill‑rating models
(TrueSkill, Grid, Elo, PreviousSeason, Random) and produces:

* ``outputs/evaluation/metrics_summary.csv`` — per‑fold metrics
* ``outputs/evaluation/validation_report.md`` — Markdown summary
* ``outputs/evaluation/*.png`` — comparison figures

Usage
-----
.. code-block:: bash

    .venv/Scripts/python.exe -m evaluation.run_evaluation
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import pandas as pd

from evaluation.baselines import SkillPredictor, build_all_predictors
from evaluation.reporter import (
    generate_report,
    plot_fold_consistency,
    plot_model_comparison,
    set_style as set_plot_style,
)
from evaluation.validator import ChronologicalValidator

logger = logging.getLogger("f1_evaluation")

NA_VALUES = [r"\N"]


class EvaluationRunner:
    """Orchestrates model evaluation from data loading through to report
    generation.

    Args:
        data_dir: Path to the ``data/`` folder.
        model_data_path: Path to ``f1_model_ready.csv``.
        rating_history_path: Path to ``driver_rating_history.csv``.
        output_dir: Directory for evaluation outputs.
        min_train_years: Minimum training seasons per fold.
    """

    def __init__(
        self,
        data_dir: str,
        model_data_path: str,
        rating_history_path: str,
        output_dir: str,
        min_train_years: int = 4,
    ) -> None:
        self._data_dir = data_dir
        self._model_data_path = model_data_path
        self._rating_history_path = rating_history_path
        self._output_dir = output_dir
        self._min_train_years = min_train_years

    def run(self) -> None:
        """Execute the full evaluation pipeline."""
        os.makedirs(self._output_dir, exist_ok=True)
        set_plot_style()

        # -- Load data ----------------------------------------------------
        logger.info("Loading preprocessed data from %s", self._model_data_path)
        df = pd.read_csv(self._model_data_path)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("  → %d rows, %d–%d", len(df), int(df["year"].min()), int(df["year"].max()))

        # -- Prepare PreviousSeason data ----------------------------------
        results_with_year = self._load_results_with_year()
        rating_history = self._load_rating_history()

        # -- Build models -------------------------------------------------
        models = build_all_predictors(
            results_with_year=results_with_year,
            rating_history=rating_history,
        )
        logger.info("Models registered: %s", list(models.keys()))

        # -- Run CV -------------------------------------------------------
        validator = ChronologicalValidator(min_train_years=self._min_train_years)
        metrics_df = validator.run(df, models)

        if metrics_df.empty:
            logger.error("No metrics produced — check data and fold generation.")
            return

        # -- Save raw metrics ---------------------------------------------
        metrics_path = os.path.join(self._output_dir, "metrics_summary.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Metrics saved to %s", metrics_path)

        # -- Generate figures ---------------------------------------------
        plot_metrics = [
            "pairwise_accuracy",
            "top_1_accuracy",
            "spearman_rho",
            "mse_position",
        ]
        for metric in plot_metrics:
            if metric in metrics_df.columns:
                plot_model_comparison(metrics_df, metric, self._output_dir)
                plot_fold_consistency(metrics_df, metric, self._output_dir)

        # -- Generate report ----------------------------------------------
        from evaluation.reporter import compute_summary_table

        summary = compute_summary_table(metrics_df)
        summary_path = os.path.join(self._output_dir, "metrics_summary.csv")
        # Save a clean pivoted version too
        summary.to_csv(
            os.path.join(self._output_dir, "metrics_summary_pivoted.csv"), index=False
        )
        generate_report(summary, metrics_df, self._output_dir)

        logger.info("Evaluation complete.  Outputs in %s", self._output_dir)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _load_results_with_year(self) -> pd.DataFrame | None:
        """Load ``results.csv`` joined with ``races.csv`` to get year info.

        Needed by ``PreviousSeasonPredictor`` for season‑aggregate points.
        """
        results_path = os.path.join(self._data_dir, "results.csv")
        races_path = os.path.join(self._data_dir, "races.csv")
        if not os.path.exists(results_path) or not os.path.exists(races_path):
            logger.warning("results.csv or races.csv missing — PrevSeason baseline skipped.")
            return None
        results = pd.read_csv(results_path, na_values=NA_VALUES, keep_default_na=True)
        races = pd.read_csv(races_path, na_values=NA_VALUES, keep_default_na=True)
        merged = results.merge(races[["raceId", "year"]], on="raceId", how="left")
        return merged

    def _load_rating_history(self) -> pd.DataFrame | None:
        """Load the TrueSkill rating history CSV."""
        try:
            df = pd.read_csv(self._rating_history_path)
            df["date"] = pd.to_datetime(df["date"])
            logger.info("Loaded rating history: %d rows.", len(df))
            return df
        except FileNotFoundError:
            logger.warning("Rating history not found — TrueSkill predictor skipped.")
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure logging and launch the evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, ".."))

    runner = EvaluationRunner(
        data_dir=os.path.join(project_root, "data"),
        model_data_path=os.path.join(
            project_root, "data_preprocessing", "f1_model_ready.csv"
        ),
        rating_history_path=os.path.join(
            project_root, "outputs", "history", "driver_rating_history.csv"
        ),
        output_dir=os.path.join(project_root, "outputs", "evaluation"),
        min_train_years=4,
    )
    runner.run()


if __name__ == "__main__":
    main()
