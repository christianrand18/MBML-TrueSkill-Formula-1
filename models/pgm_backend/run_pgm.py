"""Orchestrator: trains all three PGM models, exports CSVs, and generates 10 plots.

Usage:
    uv run python -m models.pgm_backend.run_pgm
"""

import os
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch

from models.pgm_backend.data_preparation import load_dataset
from models.pgm_backend.inference import (
    compare_svi_nuts,
    extract_svi_posterior,
    extract_svi_posterior_extended,
    extract_svi_posterior_full,
    run_nuts,
    train_svi,
)
from models.pgm_backend.model_baseline import BaselineModel
from models.pgm_backend.model_extended import ExtendedModel
from models.pgm_backend.model_full import FullModel
from models.pgm_backend.posterior import extract_posterior


OUTPUT_DIR = "outputs/pgm_model"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

LABEL_MAP = {
    1: "Hamilton",
    830: "Verstappen",
    4: "Alonso",
    822: "Leclerc",
    832: "Sainz",
    847: "Russell",
    846: "Norris",
    857: "Piastri",
    3: "Rosberg",
    20: "Vettel",
    13: "Webber",
    18: "Massa",
    5: "Button",
    8: "Raikkonen",
    2: "Heidfeld",
}

CONSTRUCTOR_LABEL_MAP = {
    131: "Mercedes",
    9: "Red Bull",
    6: "Ferrari",
    1: "McLaren",
    3: "Williams",
    4: "Renault",
    5: "Toro Rosso",
    10: "Force India",
    15: "Sauber",
}


def _label_for_driver(driver_id):
    return LABEL_MAP.get(driver_id, f"#{driver_id}")


def _label_for_constructor(cons_id):
    return CONSTRUCTOR_LABEL_MAP.get(cons_id, f"#{cons_id}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _plot_prior_predictive():
    """Plot 1: Prior predictive win rate bar chart."""
    D, K = 20, 10
    SIGMA_S, SIGMA_C = 1.0, 1.0
    N_DRAWS = 100

    cons_assignment = torch.arange(D) % K
    fastest_won = 0

    for _ in range(N_DRAWS):
        s = torch.randn(D) * SIGMA_S
        c_raw = torch.randn(K - 1) * SIGMA_C
        c = torch.cat([c_raw, -c_raw.sum(dim=0, keepdim=True)])
        p = s + c[cons_assignment]

        fastest_driver = p.argmax().item()

        remaining = list(range(D))
        for pos in range(D):
            scores = p[torch.tensor(remaining)]
            probs = torch.softmax(scores, dim=0)
            pick = torch.multinomial(probs, 1).item()
            if remaining[pick] == fastest_driver:
                if pos == 0:
                    fastest_won += 1
                break
            remaining.pop(pick)

    win_rate = fastest_won / N_DRAWS

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(["Prior predictive"], [win_rate], color="steelblue", height=0.4)
    ax.axvline(0.20, color="gray", linestyle="--", label="0.20 acceptance")
    ax.axvline(0.80, color="gray", linestyle="--", label="0.80 acceptance")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Win rate (fastest driver wins)")
    ax.set_title(f"Prior Predictive Check ({N_DRAWS} draws, {D} drivers, {K} constructors)")
    ax.legend(loc="upper right")
    ax.text(win_rate + 0.02, 0, f"{win_rate:.2f}", va="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "prior_predictive_win_rate.png"), dpi=150)
    plt.close(fig)


def _plot_svi_vs_nuts(nuts_df=None):
    """Plot 2: SVI mean vs NUTS mean scatter with error bars."""
    if nuts_df is None:
        csv_path = os.path.join(OUTPUT_DIR, "nuts_vs_svi_comparison.csv")
        if not os.path.exists(csv_path):
            print("  [WARN] NUTS comparison CSV not found — skipping SVI vs NUTS plot")
            return
        nuts_df = pd.read_csv(csv_path)

    drivers = nuts_df[nuts_df.entity_type == "driver"].copy()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(
        drivers["svi_mean"],
        drivers["nuts_mean"],
        yerr=drivers["nuts_std"],
        fmt="o",
        alpha=0.6,
        capsize=2,
        markersize=4,
        color="steelblue",
    )
    lims = [
        min(drivers["svi_mean"].min(), drivers["nuts_mean"].min()) - 0.2,
        max(drivers["svi_mean"].max(), drivers["nuts_mean"].max()) + 0.2,
    ]
    ax.plot(lims, lims, "k-", linewidth=1, alpha=0.5, label="x = y")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("SVI Posterior Mean")
    ax.set_ylabel("NUTS Posterior Mean")
    ax.set_title("SVI vs NUTS: Driver Skill Posterior Means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "svi_vs_nuts_scatter.png"), dpi=150)
    plt.close(fig)


def _plot_synthetic_recovery():
    """Plot 3: Synthetic recovery scatter (true vs inferred)."""
    import types

    from models.pgm_backend.model_baseline import BaselineModel

    torch.manual_seed(123)

    true_s = torch.tensor([2.5, 2.0, 1.5, 0.0, -1.5])
    true_c = torch.tensor([2.0, 0.5, -2.5])
    N_RACES = 50

    all_driver_idx, all_cons_idx = [], []

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
    train_svi(model, dataset, n_steps=3000, lr=0.01, log_every=1000)
    post = extract_svi_posterior(model)

    inferred_s_centred = post["s_loc"] - post["s_loc"].mean()
    true_s_centred = true_s - true_s.mean()

    true_vals = torch.cat([true_s_centred, true_c]).numpy()
    inferred_vals = torch.cat([inferred_s_centred, post["c_loc"]]).numpy()

    labels = [f"D{i}" for i in range(5)] + [f"C{i}" for i in range(3)]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_vals, inferred_vals, color="steelblue", s=60, zorder=5)
    lim = max(abs(true_vals).max(), abs(inferred_vals).max()) + 0.5
    ax.plot([-lim, lim], [-lim, lim], "k-", alpha=0.5, label="x = y")
    ax.fill_between(
        [-lim, lim], [-lim - 0.8, lim - 0.8], [-lim + 0.8, lim + 0.8],
        alpha=0.1, color="gray", label=r"$\pm$0.8 tolerance",
    )
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (true_vals[i], inferred_vals[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True value")
    ax.set_ylabel("Inferred posterior mean")
    ax.set_title("Synthetic Recovery: True vs Inferred")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "synthetic_recovery.png"), dpi=150)
    plt.close(fig)


def _plot_elbo_curves(losses1, losses2, losses3):
    """Plot 4: ELBO loss vs step for all three models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses1, alpha=0.7, linewidth=0.8, label="Model 1 (Baseline)")
    ax.plot(losses2, alpha=0.7, linewidth=0.8, label="Model 2 (Extended)")
    ax.plot(losses3, alpha=0.7, linewidth=0.8, label="Model 3 (Full)")
    ax.set_xlabel("SVI Step")
    ax.set_ylabel("ELBO Loss")
    ax.set_title("ELBO Convergence: All Three Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "elbo_curves.png"), dpi=150)
    plt.close(fig)


def _plot_temporal_drivers(posterior, dataset):
    """Plot 5: Driver s[t,d] over seasons for notable drivers."""
    s_loc = posterior["s_loc"].numpy()  # (T, D)
    T, D = s_loc.shape

    notable_ids = [1, 830, 4, 847, 846, 857]  # Hamilton, Verstappen, Alonso, Russell, Norris, Piastri
    driver_id_to_idx = {v: k for k, v in dataset.driver_map.items()}

    seasons = list(range(2011, 2011 + T))

    fig, ax = plt.subplots(figsize=(12, 6))
    for did in notable_ids:
        if did in driver_id_to_idx:
            idx = driver_id_to_idx[did]
            ax.plot(seasons, s_loc[:, idx], marker="o", linewidth=1.5,
                    label=_label_for_driver(did), markersize=4)
    ax.set_xlabel("Season")
    ax.set_ylabel("Driver Skill (s)")
    ax.set_title("Temporal Driver Skills (Model 2)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "temporal_driver_skills.png"), dpi=150)
    plt.close(fig)


def _plot_temporal_constructors(posterior, dataset):
    """Plot 6: Constructor c[t,k] over seasons for notable constructors."""
    c_loc = posterior["c_loc"].numpy()  # (T, K)
    T, K = c_loc.shape

    notable_cons_ids = [131, 9, 6, 1, 4, 10]
    cons_id_to_idx = {v: k for k, v in dataset.constructor_map.items()}

    seasons = list(range(2011, 2011 + T))

    fig, ax = plt.subplots(figsize=(12, 6))
    for cid in notable_cons_ids:
        if cid in cons_id_to_idx:
            idx = cons_id_to_idx[cid]
            ax.plot(seasons, c_loc[:, idx], marker="s", linewidth=1.5,
                    label=_label_for_constructor(cid), markersize=4)
    ax.set_xlabel("Season")
    ax.set_ylabel("Constructor Performance (c)")
    ax.set_title("Temporal Constructor Performance (Model 2)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "temporal_constructor_performance.png"), dpi=150)
    plt.close(fig)


def _plot_wet_weather_specialists(posterior, dataset):
    """Plot 7: Horizontal bar chart of delta_d for all 77 drivers."""
    delta_d_loc = posterior["delta_d_loc"].numpy()

    order = np.argsort(delta_d_loc)
    sorted_delta = delta_d_loc[order]
    sorted_ids = [dataset.driver_map[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 16))
    colors = ["steelblue" if v >= 0 else "coral" for v in sorted_delta]
    ax.barh(range(len(sorted_delta)), sorted_delta, color=colors, height=0.8)
    ax.set_yticks(range(len(sorted_delta)))
    ax.set_yticklabels([f"#{did}" for did in sorted_ids], fontsize=6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$\delta_d$ (wet-weather skill modifier)")
    ax.set_title("Wet-Weather Driver Specialists (Model 3)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "wet_weather_specialists.png"), dpi=150)
    plt.close(fig)


def _plot_beta_pi(beta_pi_loc, beta_pi_scale):
    """Plot 8: Approximate posterior density of beta_pi."""
    samples = torch.distributions.Normal(beta_pi_loc, beta_pi_scale).sample((10000,)).numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(samples, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(beta_pi_loc, color="coral", linewidth=2, linestyle="--",
               label=f"mean = {beta_pi_loc:.3f}")
    ax.set_xlabel(r"$\beta_\pi$ (pit-stop coefficient)")
    ax.set_ylabel("Density")
    ax.set_title(r"Approximate Posterior: $\beta_\pi$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "beta_pi_posterior.png"), dpi=150)
    plt.close(fig)


def _plot_cross_model_ranking(posterior1, posterior2, posterior3, dataset):
    """Plot 9: Top-15 drivers by posterior mean s_d across all three models."""
    s1 = posterior1["s_loc"].numpy()  # (D,)
    s2 = posterior2["s_loc"][13].numpy()  # season 13 (2024)
    s3 = posterior3["s_loc"][13].numpy()

    D = len(s1)
    ranks1 = np.argsort(-s1)  # descending
    top15_idx = ranks1[:15]

    labels = [_label_for_driver(dataset.driver_map[i]) for i in top15_idx]
    s1_top = s1[top15_idx]
    s2_top = s2[top15_idx]
    s3_top = s3[top15_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top15_idx))
    height = 0.25
    ax.barh(y - height, s1_top, height, label="Model 1 (all seasons)", color="steelblue")
    ax.barh(y, s2_top, height, label="Model 2 (2024)", color="coral")
    ax.barh(y + height, s3_top, height, label="Model 3 (2024)", color="seagreen")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Driver Skill Mean (s)")
    ax.set_title("Top 15 Drivers: Cross-Model Comparison (2024)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "cross_model_driver_ranking.png"), dpi=150)
    plt.close(fig)


def _plot_uncertainty_vs_races(posterior1, dataset):
    """Plot 10: Posterior std vs number of races per driver (Model 1)."""
    s_scale = posterior1["s_scale"].numpy()
    counts = torch.bincount(
        dataset.driver_idx, minlength=dataset.n_drivers
    ).numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(counts, s_scale, alpha=0.6, color="steelblue", s=30)
    ax.set_xlabel("Number of Race Entries")
    ax.set_ylabel("Posterior Std (s)")
    ax.set_title("Uncertainty vs Experience (Model 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "uncertainty_vs_races.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    t_start = time.time()

    # ---------- Load dataset ----------
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"  D={dataset.n_drivers}, K={dataset.n_constructors}, "
          f"T={dataset.n_seasons}, C={dataset.n_circuits}, "
          f"R={dataset.n_races}")

    # ---------- 1. Model 1: Baseline SVI ----------
    print("\n=== Model 1: Baseline SVI ===")
    pyro.clear_param_store()
    model1 = BaselineModel(
        n_drivers=dataset.n_drivers,
        n_constructors=dataset.n_constructors,
    )
    losses1 = train_svi(model1, dataset, n_steps=3000, lr=0.01, log_every=500)
    posterior1 = extract_svi_posterior(model1)
    df1 = extract_posterior("baseline", dataset)
    df1.to_csv(os.path.join(OUTPUT_DIR, "baseline_posterior.csv"), index=False)
    print(f"  baseline_posterior.csv: {len(df1)} rows")

    # ---------- 2. Model 1: NUTS ----------
    print("\n=== Model 1: NUTS ===")
    nuts_df = None
    try:
        pyro.clear_param_store()
        mcmc = run_nuts(model1, dataset, num_warmup=500, num_samples=500)
        nuts_df = compare_svi_nuts(posterior1, mcmc, dataset)
        print(f"  nuts_vs_svi_comparison.csv: {len(nuts_df)} rows")
    except Exception as e:
        print(f"  [WARN] NUTS failed ({e}) — falling back to existing CSV if available")

    # ---------- 3. Model 2: Extended SVI ----------
    print("\n=== Model 2: Extended SVI ===")
    pyro.clear_param_store()
    model2 = ExtendedModel(
        n_drivers=dataset.n_drivers,
        n_constructors=dataset.n_constructors,
        n_seasons=dataset.n_seasons,
        n_circuits=dataset.n_circuits,
    )
    step_kwargs2 = {
        "driver_idx": dataset.driver_idx,
        "cons_idx": dataset.cons_idx,
        "season_idx": dataset.season_idx,
        "circuit_idx": dataset.circuit_idx,
        "race_idx": dataset.race_idx,
        "wet": dataset.wet,
        "race_lengths": dataset.race_lengths,
    }
    losses2 = train_svi(model2, dataset, n_steps=5000, lr=0.01, log_every=500,
                        step_kwargs=step_kwargs2)
    posterior2 = extract_svi_posterior_extended(model2)
    df2 = extract_posterior("extended", dataset)
    df2.to_csv(os.path.join(OUTPUT_DIR, "extended_posterior.csv"), index=False)
    print(f"  extended_posterior.csv: {len(df2)} rows")

    # ---------- 4. Model 3: Full SVI ----------
    print("\n=== Model 3: Full SVI ===")
    pyro.clear_param_store()
    model3 = FullModel(
        n_drivers=dataset.n_drivers,
        n_constructors=dataset.n_constructors,
        n_seasons=dataset.n_seasons,
        n_circuits=dataset.n_circuits,
    )
    step_kwargs3 = {
        "driver_idx": dataset.driver_idx,
        "cons_idx": dataset.cons_idx,
        "season_idx": dataset.season_idx,
        "circuit_idx": dataset.circuit_idx,
        "race_idx": dataset.race_idx,
        "wet": dataset.wet,
        "race_lengths": dataset.race_lengths,
        "pit_norm": dataset.pit_norm,
        "is_mech": dataset.is_mech,
        "cons_idx_all": dataset.cons_idx_all,
        "season_idx_all": dataset.season_idx_all,
    }
    losses3 = train_svi(model3, dataset, n_steps=5000, lr=0.01, log_every=500,
                        step_kwargs=step_kwargs3)
    posterior3 = extract_svi_posterior_full(model3)
    beta_pi_scale = pyro.param("beta_pi_scale").detach().clone()
    df3 = extract_posterior("full", dataset)
    df3.to_csv(os.path.join(OUTPUT_DIR, "full_posterior.csv"), index=False)
    print(f"  full_posterior.csv: {len(df3)} rows")

    # ---------- 5. Generate all 10 plots ----------
    print("\n=== Generating plots ===")

    print("  [1/10] Prior predictive")
    _plot_prior_predictive()

    print("  [2/10] SVI vs NUTS")
    _plot_svi_vs_nuts(nuts_df)

    print("  [3/10] Synthetic recovery")
    _plot_synthetic_recovery()

    print("  [4/10] ELBO curves")
    _plot_elbo_curves(losses1, losses2, losses3)

    print("  [5/10] Temporal driver skills")
    _plot_temporal_drivers(posterior2, dataset)

    print("  [6/10] Temporal constructor performance")
    _plot_temporal_constructors(posterior2, dataset)

    print("  [7/10] Wet-weather specialists")
    _plot_wet_weather_specialists(posterior3, dataset)

    print("  [8/10] beta_pi posterior")
    _plot_beta_pi(
        posterior3["beta_pi_loc"].item(),
        beta_pi_scale.item(),
    )

    print("  [9/10] Cross-model driver ranking")
    _plot_cross_model_ranking(posterior1, posterior2, posterior3, dataset)

    print("  [10/10] Uncertainty vs races")
    _plot_uncertainty_vs_races(posterior1, dataset)

    # ---------- 6. Summary ----------
    elapsed = time.time() - t_start
    print(f"\n=== Complete ({elapsed:.0f}s) ===")

    print("\n--- Top 10 Drivers (Model 1, static) ---")
    top10 = np.argsort(-posterior1["s_loc"].numpy())[:10]
    for rank, idx in enumerate(top10):
        did = dataset.driver_map[idx]
        s_val = posterior1["s_loc"][idx].item()
        print(f"  {rank+1:2d}. #{did:4d} ({_label_for_driver(did):12s})  s = {s_val:+.4f}")

    print("\n--- Top 5 Constructors (Model 1, static) ---")
    top5c = np.argsort(-posterior1["c_loc"].numpy())[:5]
    for rank, idx in enumerate(top5c):
        cid = dataset.constructor_map[idx]
        c_val = posterior1["c_loc"][idx].item()
        print(f"  {rank+1:2d}. #{cid:3d} ({_label_for_constructor(cid):12s})  c = {c_val:+.4f}")

    print("\n--- Model 3 key parameters ---")
    print(f"  alpha_rel = {posterior3['alpha_rel_loc'].item():.4f}")
    print(f"  beta_pi   = {posterior3['beta_pi_loc'].item():.4f}")
    print(f"  beta_w    = {posterior3['beta_w_loc'].item():.4f}")


if __name__ == "__main__":
    main()
