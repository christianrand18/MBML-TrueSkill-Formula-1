r"""
F1 Pyro Bayesian Model — Entry Point
======================================

Trains a pairwise‑ranking Pyro model with driver / constructor latent
skills, grid‑position covariate, and optional weather‑dependent noise.
Exports posterior ratings and compares against the TrueSkill baseline.

Usage
-----
.. code-block:: bash

    .venv/Scripts/python.exe -m models.pyro_backend.run_pyro_model
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import pyro
import torch

from evaluation.baselines import (
    GridPredictor,
    RandomPredictor,
    SkillPredictor,
    TrueSkillHistoryPredictor,
)
from evaluation.metrics import compute_fold_metrics
from evaluation.validator import ChronologicalValidator
from models.pyro_backend.data_preparation import DataPreparer, PairwiseDataset
from models.pyro_backend.pyro_evaluator import PyroSkillPredictor
from models.pyro_backend.pyro_model import F1SkillModel, train_svi

logger = logging.getLogger("f1_pyro")


# ======================================================================
# Orchestrator
# ======================================================================


class PyroModelRunner:
    """Train a Pyro F1 skill model and compare it against baselines.

    Args:
        enriched_data_path: Path to ``f1_enriched.csv``.
        rating_history_path: Path to TrueSkill ``driver_rating_history.csv``.
        output_dir: Directory for posterior files and comparison figures.
    """

    def __init__(
        self,
        enriched_data_path: str,
        rating_history_path: str,
        output_dir: str,
    ) -> None:
        self._data_path = enriched_data_path
        self._history_path = rating_history_path
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

    # ==================================================================
    # Run
    # ==================================================================

    def run(self) -> None:
        """Train static → temporal Pyro model, export, compare."""

        # ---- Data preparation -----------------------------------------
        preparer = DataPreparer(self._data_path)
        dataset, df_norm = preparer.build()

        # ---- Stage 1: Static model (fast) ----------------------------
        logger.info("=" * 55)
        logger.info("STAGE 1: Static Pyro Model")
        logger.info("=" * 55)
        self._train_static(dataset, preparer, df_norm)

        # ---- Stage 2: Temporal model (random walk) --------------------
        # NOTE: Temporal model with 1400+ latent variables is computationally
        # expensive (~20 minutes for SVI).  Run by setting SKIP_TEMPORAL = False.
        SKIP_TEMPORAL = True
        if not SKIP_TEMPORAL:
            logger.info("=" * 55)
            logger.info("STAGE 2: Temporal Pyro Model")
            logger.info("=" * 55)
            self._train_temporal(dataset, preparer, df_norm)
        else:
            logger.info("=" * 55)
            logger.info("STAGE 2: Temporal Pyro Model — SKIPPED (computationally heavy)")
            logger.info("=" * 55)

        # ---- Comparison vs TrueSkill ---------------------------------
        logger.info("=" * 55)
        logger.info("COMPARISON: Pyro vs TrueSkill vs Baselines")
        logger.info("=" * 55)
        self._compare(df_norm, preparer)

        logger.info("All outputs saved to %s", self._output_dir)

    # ==================================================================
    # Training
    # ==================================================================

    def _train_static(
        self, dataset: PairwiseDataset, preparer: DataPreparer, df: pd.DataFrame
    ) -> None:
        """Train the static variant and export posteriors."""
        model = F1SkillModel(dataset, temporal=False, use_weather_noise=False)
        n_steps = 3000
        train_svi(model, n_steps=n_steps, lr=0.005, log_every=300)

        # Export posteriors
        self._export_posteriors(preparer, suffix="_static")
        logger.info("Static model trained (%d steps).", n_steps)

    def _train_temporal(
        self, dataset: PairwiseDataset, preparer: DataPreparer, df: pd.DataFrame
    ) -> None:
        """Train the temporal (random‑walk) variant and export posteriors."""
        model = F1SkillModel(dataset, temporal=True, use_weather_noise=False)
        n_steps = 1500
        train_svi(model, n_steps=n_steps, lr=0.003, log_every=300)

        # Export posteriors
        self._export_posteriors(preparer, suffix="_temporal")
        logger.info("Temporal model trained (%d steps).", n_steps)

    # ==================================================================
    # Export
    # ==================================================================

    def _export_posteriors(self, preparer: DataPreparer, suffix: str) -> None:
        """Extract posterior means from the param store and save CSVs."""
        store = pyro.get_param_store()
        n_d = preparer.n_drivers
        n_c = preparer.n_constructors
        n_s = preparer.n_seasons
        temporal = "temporal" in suffix

        # Load names
        driver_names = self._load_driver_names()
        constructor_names = self._load_constructor_names()

        # ---- Driver ratings -------------------------------------------
        driver_rows: List[Dict] = []

        if not temporal:
            loc_key = "driver_loc"
            if loc_key in store:
                loc = store[loc_key].detach().numpy()
                for idx in range(n_d):
                    d_id = preparer.reverse_driver(idx)
                    name = driver_names.get(d_id, f"Driver_{d_id}")
                    driver_rows.append({
                        "driverId": d_id,
                        "driverName": name,
                        "mu": float(loc[idx]),
                        "sigma": 5.0,  # approximate
                    })
        else:
            for idx in range(n_d):
                d_id = preparer.reverse_driver(idx)
                name = driver_names.get(d_id, f"Driver_{d_id}")
                # Average over seasons
                mus = []
                sigmas = []
                for s in range(n_s):
                    loc_k = f"driver_{idx}_s{s}_loc"
                    scale_k = f"driver_{idx}_s{s}_scale"
                    if loc_k in store:
                        mus.append(float(store[loc_k].detach()))
                    if scale_k in store:
                        sigmas.append(float(store[scale_k].detach()))
                driver_rows.append({
                    "driverId": d_id,
                    "driverName": name,
                    "mu": float(np.mean(mus)) if mus else 0.0,
                    "sigma": float(np.mean(sigmas)) if sigmas else 5.0,
                })

        drivers_df = pd.DataFrame(driver_rows).sort_values("mu", ascending=False)
        drivers_df.to_csv(
            os.path.join(self._output_dir, f"driver_ratings{suffix}.csv"),
            index=False,
        )

        # ---- Constructor ratings --------------------------------------
        cons_rows: List[Dict] = []
        if not temporal:
            loc_key = "constructor_loc"
            if loc_key in store:
                loc = store[loc_key].detach().numpy()
                for idx in range(n_c):
                    c_id = preparer.reverse_constructor(idx)
                    name = constructor_names.get(c_id, f"Constructor_{c_id}")
                    cons_rows.append({
                        "constructorId": c_id,
                        "constructorName": name,
                        "mu": float(loc[idx]),
                        "sigma": 5.0,
                    })
        else:
            for idx in range(n_c):
                c_id = preparer.reverse_constructor(idx)
                name = constructor_names.get(c_id, f"Constructor_{c_id}")
                mus = []
                for s in range(n_s):
                    loc_k = f"constructor_{idx}_s{s}_loc"
                    if loc_k in store:
                        mus.append(float(store[loc_k].detach()))
                cons_rows.append({
                    "constructorId": c_id,
                    "constructorName": name,
                    "mu": float(np.mean(mus)) if mus else 0.0,
                    "sigma": 5.0,
                })

        cons_df = pd.DataFrame(cons_rows).sort_values("mu", ascending=False)
        cons_df.to_csv(
            os.path.join(self._output_dir, f"constructor_ratings{suffix}.csv"),
            index=False,
        )

        # ---- Beta_grid ------------------------------------------------
        if "beta_grid_loc" in store:
            beta = float(store["beta_grid_loc"].detach())
            pd.DataFrame(
                [{"parameter": "beta_grid", "value": beta}]
            ).to_csv(
                os.path.join(self._output_dir, f"coefficients{suffix}.csv"),
                index=False,
            )
            logger.info("  Beta_grid = %.4f", beta)

        logger.info(
            "  Exported %d drivers, %d constructors%s.",
            len(drivers_df), len(cons_df), suffix,
        )

        # ---- Log top 10 -----------------------------------------------
        logger.info("  Top 5 Pyro drivers:")
        for _, row in drivers_df.head(5).iterrows():
            logger.info("    %-25s mu=%.3f", row["driverName"], row["mu"])

    # ==================================================================
    # Comparison
    # ==================================================================

    def _compare(
        self, df: pd.DataFrame, preparer: DataPreparer
    ) -> None:
        """Run chronological CV: Pyro vs TrueSkill vs Grid vs Random."""
        # Load TrueSkill history
        history = None
        try:
            history = pd.read_csv(self._history_path)
            history["date"] = pd.to_datetime(history["date"])
        except FileNotFoundError:
            logger.warning("TrueSkill history not found — skipping comparison.")

        # Build predictors
        pyro_pred = PyroSkillPredictor(preparer, temporal=False)
        models: Dict[str, SkillPredictor] = {
            "Grid": GridPredictor(),
            "Random": RandomPredictor(),
        }
        if history is not None:
            models["TrueSkill"] = TrueSkillHistoryPredictor(history)
        models["Pyro"] = pyro_pred

        # Chronological CV
        validator = ChronologicalValidator(min_train_years=4)
        metrics_df = validator.run(df, models)

        if metrics_df.empty:
            logger.warning("No evaluation metrics produced.")
            return

        # Save metrics
        metrics_path = os.path.join(self._output_dir, "pyro_comparison_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Comparison metrics saved to %s", metrics_path)

        # Summarise
        logger.info("-" * 55)
        logger.info("MODEL COMPARISON (mean over folds)")
        logger.info("-" * 55)
        for model_name in ["Grid", "TrueSkill", "Pyro", "Random"]:
            sub = metrics_df[metrics_df["model"] == model_name]
            if len(sub) == 0:
                continue
            logger.info(
                "  %-12s  pairwise_acc=%.3f  top1=%.3f  spearman=%.3f  mse=%.1f",
                model_name,
                sub["pairwise_accuracy"].mean(),
                sub["top_1_accuracy"].mean(),
                sub["spearman_rho"].mean(),
                sub["mse_position"].mean(),
            )

    # ==================================================================
    # Name lookups
    # ==================================================================

    def _load_driver_names(self) -> Dict[int, str]:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"
        )
        path = os.path.join(data_dir, "drivers.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path, na_values=[r"\N"], keep_default_na=True)
        return {
            int(r["driverId"]): f"{r['forename']} {r['surname']}".strip()
            for _, r in df.iterrows()
        }

    def _load_constructor_names(self) -> Dict[int, str]:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"
        )
        path = os.path.join(data_dir, "constructors.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path, na_values=[r"\N"], keep_default_na=True)
        return {
            int(r["constructorId"]): str(r["name"])
            for _, r in df.iterrows()
        }


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    """Configure logging and launch the Pyro pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, "..", ".."))

    runner = PyroModelRunner(
        enriched_data_path=os.path.join(
            project_root, "data_preprocessing", "f1_enriched.csv"
        ),
        rating_history_path=os.path.join(
            project_root, "outputs", "history", "driver_rating_history.csv"
        ),
        output_dir=os.path.join(project_root, "outputs", "pyro_model"),
    )
    runner.run()


if __name__ == "__main__":
    main()
