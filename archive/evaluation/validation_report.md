# F1 TrueSkill Model Validation Report

Chronological cross‑validation over **10 seasons** (2015–2024), training on all prior years.  Five models evaluated head‑to‑head.

## Model Performance Summary

| Model | Pairwise Acc | Top‑1 Win | Top‑3 Podium | Spearman ρ | MRR | MSE ↓ |
|-------|-------------|-----------|-------------|------------|-----|-------|
| **Grid** | 0.730 ± 0.023 | 0.422 | 0.577 | 0.569 | 0.625 | 29.1 |
| **Elo** | 0.678 ± 0.031 | 0.389 | 0.683 | 0.496 | 0.600 | 34.2 |
| **PrevSeason** | 0.670 ± 0.031 | 0.543 | 0.788 | 0.517 | 0.680 | 31.5 |
| **TrueSkill** | 0.659 ± 0.025 | 0.389 | 0.683 | 0.455 | 0.573 | 36.8 |
| **Random** | 0.506 ± 0.020 | 0.032 | 0.142 | 0.012 | 0.178 | 66.9 |

## Per‑Fold Breakdown (Top‑1 Accuracy)

| Test Year | Elo | Grid | PrevSeason | Random | TrueSkill |
|-----------|-----------|-----------|-----------|-----------|-----------|
| 2015 | 0.526 | 0.579 | 0.526 | 0.000 | 0.526 |
| 2016 | 0.476 | 0.571 | 0.476 | 0.000 | 0.476 |
| 2017 | 0.450 | 0.550 | 0.450 | 0.000 | 0.450 |
| 2018 | 0.524 | 0.476 | 0.524 | 0.048 | 0.524 |
| 2019 | 0.524 | 0.190 | 0.524 | 0.000 | 0.524 |
| 2020 | 0.647 | 0.471 | 0.647 | 0.000 | 0.647 |
| 2021 | 0.364 | 0.364 | 0.364 | 0.091 | 0.364 |
| 2022 | 0.000 | 0.364 | 0.682 | 0.136 | 0.000 |
| 2023 | 0.000 | 0.318 | 0.864 | 0.000 | 0.000 |
| 2024 | 0.375 | 0.333 | 0.375 | 0.042 | 0.375 |

## Key Findings

- **Grid** achieves the highest pairwise accuracy (0.730).
- The **Grid** baseline achieves 0.730 pairwise accuracy, confirming that qualifying pace is a strong predictor.
- **TrueSkill** (pairwise acc 0.659) provides full posterior uncertainty estimates (σ) in addition to point predictions (μ) — a key advantage over point‑estimate baselines.

## Generated Figures

- `fold_consistency_mse_position.png`
- `fold_consistency_pairwise_accuracy.png`
- `fold_consistency_spearman_rho.png`
- `fold_consistency_top_1_accuracy.png`
- `model_comparison_mse_position.png`
- `model_comparison_pairwise_accuracy.png`
- `model_comparison_spearman_rho.png`
- `model_comparison_top_1_accuracy.png`

---
*Report generated from 50 fold‑model combinations.*