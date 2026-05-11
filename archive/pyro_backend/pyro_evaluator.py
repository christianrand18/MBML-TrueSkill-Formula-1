r"""
PyroSkillPredictor — plugs the trained Pyro model into the evaluation
framework via the ``SkillPredictor`` ABC.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyro
import torch

from evaluation.baselines import SkillPredictor
from models.pyro_backend.data_preparation import DataPreparer

logger = logging.getLogger("f1_pyro.evaluator")


class PyroSkillPredictor(SkillPredictor):
    """Extract posterior skill scores from a trained Pyro model.

    Supports both static and temporal variants.  The temporal variant
    returns the skill for the most recent training season for each
    driver / constructor.

    Args:
        preparer: ``DataPreparer`` instance (holds index mappings and
            normalisation stats).
        temporal: Whether the model is temporal.
    """

    name = "Pyro"

    def __init__(
        self, preparer: DataPreparer, temporal: bool = True
    ) -> None:
        self._preparer = preparer
        self._temporal = temporal
        self._driver_skills: Optional[torch.Tensor] = None
        self._constructor_skills: Optional[torch.Tensor] = None
        self._beta_grid: float = 0.0
        self._last_season: int = -1
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # SkillPredictor interface
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> None:
        """Extract posterior means from the global Pyro param store.

        This MUST be called after ``train_svi()`` has populated the
        param store.  The *train_df* is only used to determine the last
        training season.
        """
        if not self._temporal:
            driver_loc = pyro.get_param_store()[f"driver_loc"].detach()
            self._driver_skills = driver_loc  # shape (n_d,)
            cons_loc = pyro.get_param_store()[f"constructor_loc"].detach()
            self._constructor_skills = cons_loc  # shape (n_c,)
        else:
            n_d = self._preparer.n_drivers
            n_c = self._preparer.n_constructors
            driver_mat = torch.zeros(self._preparer.n_seasons, n_d)
            cons_mat = torch.zeros(self._preparer.n_seasons, n_c)
            for e in range(n_d):
                for s in range(self._preparer.n_seasons):
                    key = f"driver_{e}_s{s}_loc"
                    if key in pyro.get_param_store():
                        driver_mat[s, e] = pyro.get_param_store()[key].detach()
            for e in range(n_c):
                for s in range(self._preparer.n_seasons):
                    key = f"constructor_{e}_s{s}_loc"
                    if key in pyro.get_param_store():
                        cons_mat[s, e] = pyro.get_param_store()[key].detach()
            self._driver_skills = driver_mat
            self._constructor_skills = cons_mat

        # Beta grid
        if "beta_grid_loc" in pyro.get_param_store():
            self._beta_grid = float(
                pyro.get_param_store()["beta_grid_loc"].detach()
            )

        self._last_season = int(train_df["year"].max())
        self._fitted = True

    def predict_driver_skills(self, race_df: pd.DataFrame) -> Dict[int, float]:
        """Return predicted skill for each driver in a race.

        Uses posterior means plus the grid covariate effect.
        """
        skills: Dict[int, float] = {}
        test_year = int(race_df["year"].iloc[0])

        for _, row in race_df.iterrows():
            d_id = int(row["driverId"])
            c_id = int(row["constructorId"])
            grid = float(row["grid"])

            d_idx = self._preparer.driver_map.get(d_id, -1)
            c_idx = self._preparer.constructor_map.get(c_id, -1)

            if d_idx < 0 or c_idx < 0:
                skills[d_id] = 25.0
                continue

            # Skill component
            if self._temporal and self._driver_skills is not None:
                # Use skill from the last training season
                s_idx = self._preparer.season_map.get(self._last_season, 0)
                skill = float(self._driver_skills[s_idx, d_idx])
                skill += float(self._constructor_skills[s_idx, c_idx])
            elif self._driver_skills is not None:
                skill = float(self._driver_skills[d_idx])
                skill += float(self._constructor_skills[c_idx])
            else:
                skill = 25.0

            # Add grid effect (normalised)
            grid_norm = (grid - self._preparer._norm_mean.get("grid", 12)) / max(
                self._preparer._norm_std.get("grid", 5), 0.01
            )
            skill += self._beta_grid * grid_norm

            skills[d_id] = skill

        return skills
