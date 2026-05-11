r"""
F1 Data Exploration — Orchestrator
====================================

Loads the preprocessed F1 dataset, runs a battery of statistical analyses,
generates publication‑quality visualisations, and exports all plots to
``outputs/exploration/figures/``.

Usage
-----
.. code-block:: bash

    .venv/Scripts/python.exe exploration/f1_data_exploration.py
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from typing import Dict

import pandas as pd

from exploration.analysis import (
    circuit_overtaking_metric,
    compute_overview,
    constructor_dominance,
    constructor_wins_per_year,
    dnf_rate_per_season,
    driver_career_spans,
    grid_vs_finish_summary,
    load_auxiliary_tables,
    load_rating_history,
    pit_stop_trends,
    podium_counts,
    positions_gained,
    sigma_vs_races,
    teammate_comparison,
    top_constructors_by_wins,
    top_dnf_reasons,
    top_driver_trajectories,
    top_drivers_by_wins,
)
from exploration.visualisations import (
    plot_circuit_map,
    plot_circuit_overtaking,
    plot_constructor_heatmap,
    plot_dnf_rate,
    plot_dnf_reasons,
    plot_driver_careers,
    plot_driver_rating_trajectories,
    plot_grid_vs_finish,
    plot_participants_per_year,
    plot_pit_stop_trends,
    plot_position_change_distribution,
    plot_races_per_year,
    plot_sigma_vs_races,
    plot_teammate_comparison,
    plot_top_constructors_wins,
    plot_top_drivers_wins,
    set_style,
)

logger = logging.getLogger("f1_exploration")


class F1DataExplorer:
    """Orchestrates all exploratory analyses and visualisations.

    Args:
        data_dir: Path to the ``data/`` folder containing raw CSVs.
        model_data_path: Path to ``f1_model_ready.csv``.
        rating_history_path: Path to ``driver_rating_history.csv`` (optional).
        output_dir: Root directory for exploration outputs.
    """

    def __init__(
        self,
        data_dir: str,
        model_data_path: str,
        rating_history_path: str,
        output_dir: str,
    ) -> None:
        self._data_dir = data_dir
        self._model_data_path = model_data_path
        self._rating_history_path = rating_history_path
        self._output_dir = output_dir
        self._figures_dir = os.path.join(output_dir, "figures")

        # Will be populated during run()
        self._df: pd.DataFrame | None = None
        self._aux: Dict[str, pd.DataFrame] = {}
        self._driver_map: Dict[int, str] = {}
        self._constructor_map: Dict[int, str] = {}
        self._circuit_map: Dict[int, str] = {}
        self._status_map: Dict[int, str] = {}
        self._history: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_everything(self) -> None:
        """Load all data sources and build lookup maps."""
        logger.info("Loading preprocessed data from %s", self._model_data_path)
        self._df = pd.read_csv(self._model_data_path)
        self._df["date"] = pd.to_datetime(self._df["date"])
        logger.info(
            "  → %d rows, %d unique races, %d drivers, %d constructors",
            len(self._df),
            self._df["raceId"].nunique(),
            self._df["driverId"].nunique(),
            self._df["constructorId"].nunique(),
        )

        self._aux = load_auxiliary_tables(self._data_dir)

        # Build name maps
        if not self._aux["drivers"].empty:
            for _, r in self._aux["drivers"].iterrows():
                self._driver_map[int(r["driverId"])] = f"{r['forename']} {r['surname']}".strip()

        if not self._aux["constructors"].empty:
            for _, r in self._aux["constructors"].iterrows():
                self._constructor_map[int(r["constructorId"])] = str(r["name"])

        if not self._aux["circuits"].empty:
            for _, r in self._aux["circuits"].iterrows():
                self._circuit_map[int(r["circuitId"])] = str(r["name"])

        if not self._aux["status"].empty:
            for _, r in self._aux["status"].iterrows():
                self._status_map[int(r["statusId"])] = str(r["status"])

        self._history = load_rating_history(self._rating_history_path)
        if self._history is not None:
            logger.info(
                "  → Loaded rating history: %d snapshots across %d races.",
                len(self._history),
                self._history["raceId"].nunique(),
            )

    # ------------------------------------------------------------------
    # Analysis + plotting helpers
    # ------------------------------------------------------------------

    def _run_analysis(self, name: str, func, *args, **kwargs) -> object:
        """Run an analysis function with logging."""
        logger.info("[%s] Computing ...", name)
        return func(*args, **kwargs)

    def _plot(self, name: str, func, *args, **kwargs) -> str:
        """Run a plotting function with logging."""
        logger.info("[%s] Rendering ...", name)
        return func(*args, **kwargs, save_dir=self._figures_dir)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all analyses and save all figures."""
        os.makedirs(self._figures_dir, exist_ok=True)
        set_style()
        self._load_everything()

        df: pd.DataFrame = self._df  # type: ignore[assignment]

        # ---------------------------------------------------------------
        # 1. Data overview
        # ---------------------------------------------------------------
        overview = self._run_analysis("Overview", compute_overview, df)
        self._plot("Races/year", plot_races_per_year, overview)
        self._plot("Participants/year", plot_participants_per_year, overview)

        # ---------------------------------------------------------------
        # 2. Driver analysis
        # ---------------------------------------------------------------
        top_wins = self._run_analysis(
            "Top drivers (wins)", top_drivers_by_wins, df, self._driver_map
        )
        self._plot("Top drivers wins", plot_top_drivers_wins, top_wins)

        podiums = self._run_analysis(
            "Podium counts", podium_counts, df, self._driver_map
        )
        logger.info("  → Top 5 by podiums: %s", podiums.head(5)["driverName"].tolist())

        spans = self._run_analysis(
            "Career spans", driver_career_spans, df, self._driver_map
        )
        self._plot("Career spans", plot_driver_careers, spans)

        # ---------------------------------------------------------------
        # 3. Constructor analysis
        # ---------------------------------------------------------------
        top_cons = self._run_analysis(
            "Top constructors (wins)",
            top_constructors_by_wins,
            df,
            self._constructor_map,
        )
        self._plot("Top constructors wins", plot_top_constructors_wins, top_cons)

        pivot = self._run_analysis(
            "Constructor heatmap",
            constructor_wins_per_year,
            df,
            self._constructor_map,
        )
        self._plot("Constructor heatmap", plot_constructor_heatmap, pivot)

        # ---------------------------------------------------------------
        # 4. Grid vs Finish
        # ---------------------------------------------------------------
        grid_summary = self._run_analysis("Grid-finish stats", grid_vs_finish_summary, df)
        logger.info(
            "  → Avg gain from pole: %.2f, from P20: %.2f",
            grid_summary.loc[grid_summary["grid"] == 1, "avg_gain"].values[0]
            if 1 in grid_summary["grid"].values
            else 0,
            grid_summary.loc[grid_summary["grid"] == 20, "avg_gain"].values[0]
            if 20 in grid_summary["grid"].values
            else 0,
        )
        self._plot("Grid vs finish", plot_grid_vs_finish, df)
        self._plot("Position change dist", plot_position_change_distribution, df)

        # ---------------------------------------------------------------
        # 5. Pit stops
        # ---------------------------------------------------------------
        pit_trends = self._run_analysis("Pit stop trends", pit_stop_trends, df)
        self._plot("Pit stop trends", plot_pit_stop_trends, pit_trends)

        # ---------------------------------------------------------------
        # 6. DNF analysis
        # ---------------------------------------------------------------
        dnf = self._run_analysis("DNF rate", dnf_rate_per_season, df, self._status_map)
        self._plot("DNF rate", plot_dnf_rate, dnf)

        reasons = self._run_analysis("DNF reasons", top_dnf_reasons, df, self._status_map)
        self._plot("DNF reasons", plot_dnf_reasons, reasons)

        # ---------------------------------------------------------------
        # 7. Circuits
        # ---------------------------------------------------------------
        circ = self._run_analysis(
            "Circuit overtaking", circuit_overtaking_metric, df, self._circuit_map
        )
        self._plot("Circuit overtaking", plot_circuit_overtaking, circ)
        self._plot("Circuit map", plot_circuit_map, df, self._circuit_map)

        # ---------------------------------------------------------------
        # 8. Rating trajectories
        # ---------------------------------------------------------------
        if self._history is not None:
            top_traj = self._run_analysis(
                "Rating trajectories",
                top_driver_trajectories,
                self._history,
            )
            self._plot("Rating trajectories", plot_driver_rating_trajectories, top_traj)

            sigma_df = self._run_analysis("Sigma vs races", sigma_vs_races, self._history)
            self._plot("Sigma vs races", plot_sigma_vs_races, sigma_df)

        # ---------------------------------------------------------------
        # 9. Teammate comparison
        # ---------------------------------------------------------------
        teammate = self._run_analysis("Teammate analysis", teammate_comparison, df)
        if not teammate.empty:
            self._plot(
                "Teammate comparison",
                plot_teammate_comparison,
                teammate,
                self._constructor_map,
                self._driver_map,
            )

        self._print_summary()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        """Print a text summary of key findings."""
        df = self._df
        n_races = df["raceId"].nunique()
        n_drivers = df["driverId"].nunique()
        n_constructors = df["constructorId"].nunique()
        years = f"{int(df['year'].min())}–{int(df['year'].max())}"

        wins = df[df["positionOrder"] == 1].groupby("driverId").size()
        top_driver_id = wins.idxmax()
        top_driver_name = self._driver_map.get(top_driver_id, str(top_driver_id))

        cons_wins = df[df["positionOrder"] == 1].groupby("constructorId").size()
        top_cons_id = cons_wins.idxmax()
        top_cons_name = self._constructor_map.get(top_cons_id, str(top_cons_id))

        # --- DNF rate using classified-finish logic (Finished, +N Laps, DSQ) ---
        dnf_total = 0
        if "statusName" not in df.columns and self._status_map:
            df = df.copy()
            df["statusName"] = df["statusId"].map(self._status_map)
        if "statusName" in df.columns:
            classified = {"Finished", "Disqualified"}
            dnf_total = (~df["statusName"].apply(
                lambda s: isinstance(s, str) and (s in classified or s.startswith("+"))
            )).sum()
        dnf_pct = (dnf_total / len(df)) * 100 if len(df) > 0 else 0.0
        avg_stops = df["num_pit_stops"].mean()
        avg_pos_gain = (df["grid"] - df["positionOrder"]).mean()

        summary = textwrap.dedent(f"""
        {'='*60}
        F1 DATA EXPLORATION SUMMARY  ({years})
        {'='*60}
          Races processed:        {n_races}
          Unique drivers:         {n_drivers}
          Unique constructors:    {n_constructors}
          Most successful driver: {top_driver_name} ({wins.max()} wins)
          Most successful team:   {top_cons_name} ({cons_wins.max()} wins)
          DNF rate:               {dnf_pct:.1f}%
          Avg pit stops/race:     {avg_stops:.2f}
          Avg position gain:      {avg_pos_gain:+.2f}
        {'='*60}
        Figures saved to: {self._figures_dir}
        """)
        logger.info(summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Set up paths and launch the explorer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, ".."))

    data_dir = os.path.join(project_root, "data")
    model_data_path = os.path.join(
        project_root, "data_preprocessing", "f1_model_ready.csv"
    )
    rating_history_path = os.path.join(
        project_root, "outputs", "history", "driver_rating_history.csv"
    )
    output_dir = os.path.join(project_root, "outputs", "exploration")

    explorer = F1DataExplorer(
        data_dir=data_dir,
        model_data_path=model_data_path,
        rating_history_path=rating_history_path,
        output_dir=output_dir,
    )
    explorer.run()


if __name__ == "__main__":
    main()
