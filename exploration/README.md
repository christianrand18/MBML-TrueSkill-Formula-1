# F1 Data Exploration

Comprehensive exploratory data analysis of the Formula 1 dataset (2011–2024).
Produces statistical summaries and publication‑quality visualisations across
drivers, constructors, circuits, race dynamics, and TrueSkill rating trajectories.

## Quick Start

```bash
.venv/Scripts/python.exe -m exploration.f1_data_exploration
```

All figures are written to `outputs/exploration/figures/`.

---

## Architecture

| File | Role |
|------|------|
| `f1_data_exploration.py` | Orchestrator — loads data, runs analyses, renders plots, prints summary |
| `analysis.py` | Pure statistical computations — returns `pd.DataFrame` objects |
| `visualisations.py` | `matplotlib` / `seaborn` plotting — produces PNG figures |

---

## Analyses Performed (16 figures)

| # | Figure | Description |
|---|--------|-------------|
| 01 | Races per year | Bar chart of race count by season |
| 02 | Participants per year | Dual‑axis: unique drivers/constructors + total entries |
| 03 | Top drivers by wins | Horizontal bar of the 15 most winning drivers |
| 04 | Driver career spans | Gantt‑style chart of driver longevity |
| 05 | Top constructors by wins | Horizontal bar of constructor win counts |
| 06 | Constructor win heatmap | Year × constructor matrix of race wins |
| 07 | Grid vs finish | Scatter plot with jitter and quadrant annotations |
| 08 | Position change distribution | Histogram of positions gained/lost from grid to finish |
| 09 | Pit stop trends | Dual‑axis: average stops and pit duration per year |
| 10 | DNF rate | Line plot of non‑finish rate per season |
| 11 | DNF reasons | Horizontal bar of most common mechanical/failure statuses |
| 12 | Circuit overtaking | Circuits ranked by average position gain (overtaking friendliness) |
| 13 | Circuit world map | Geo‑scatter of circuit locations with labels |
| 14 | Driver rating trajectories | TrueSkill mu over time for top 8 drivers |
| 15 | Sigma vs races | Uncertainty reduction as a function of career entries |
| 16 | Teammate comparison | Most dominant intra‑team driver pairings |

---

## Summary Statistics (2011–2024)

| Metric | Value |
|--------|-------|
| Races | 286 |
| Unique drivers | 77 |
| Unique constructors | 23 |
| Top driver (wins) | Lewis Hamilton — 91 wins |
| Top constructor (wins) | Mercedes — 120 wins |
| Overall DNF rate | 17.1% |
| Average pit stops per race | 1.90 |
| Average position change (grid → finish) | −0.26 |

---

## Key Insights

- **Lewis Hamilton** leads in raw wins (91), but **Nico Rosberg** has the
  highest TrueSkill μ (35.55) — Rosberg retired at his peak, while Hamilton's
  later‑career results include weaker seasons.
- **Mercedes** dominated 2014–2021 (120 wins), but **Red Bull** surged to #1 in
  the TrueSkill constructor rankings by 2024.
- **DNF rates** have steadily declined from ~21 % (2014) to ~11 % (2024),
  reflecting improved reliability.
- **Pole position** is not a guarantee: drivers starting P1 lose an average of
  2.5 positions by the finish, while back‑markers (P20) gain ~4.3 positions.
- **Monaco** has the lowest average position change — fewest overtakes — while
  circuits like **Interlagos** and **Spa** enable more passing.
- **Sigma** (uncertainty) converges roughly within 50 races: drivers with 100+
  entries have σ ≈ 0.7–0.8, while rookies with < 5 races have σ > 3.0.
