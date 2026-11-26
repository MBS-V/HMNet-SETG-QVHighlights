# Final results — QVHighlights val (1,550 queries)

All numbers measured. Official Moment-DETR evaluator for R1@0.5/0.7 and mAP; R1@0.3 and mIoU computed from the same top-1 predictions. Best checkpoint per run selected by validation mAP avg.

## Table 1 — Main results

| Model | Settings | Epochs trained | Best epoch | R1@0.3 | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP avg | mIoU |
|---|---|---|---|---|---|---|---|---|---|---|
| HM-Net | original | 19 | 7 | 28.26 | 16.52 | 8.90 | 26.49 | 10.69 | **13.37** | 20.40 |
| HM-Net + SE-TG | original | 15 | 8 | 29.35 | 15.16 | 7.03 | 26.55 | 9.99 | **12.68** | 20.21 |
| HM-Net | corrected | 30 | 3 | 28.52 | 16.13 | 7.87 | 28.53 | 10.16 | **13.07** | 20.59 |
| HM-Net + SE-TG | corrected | 30 | 6 | 26.13 | 14.45 | 6.65 | 26.40 | 10.74 | **13.29** | 17.98 |
| Query-blind prior (measured) | — | — | — | 20.06 | 9.03 | 6.06 | 17.58 | 9.87 | 10.85 | 23.89 |
| Random windows (measured) | — | — | — | 22.84 | 8.77 | 2.39 | 15.03 | 3.41 | 5.43 | 16.72 |
| HM-Net, published (Table 2; split not stated) | — | n/a | n/a | 74.77 | 62.77 | 48.13 | 63.14 | 43.70 | 43.17 | 57.88 |

## Table 2 — By query type and moment length (mAP avg / R1@0.5)

| Model | Settings | Simple (n=1,316) | Compound (n=234) | Short <10 s | Middle 10–30 s | Long >30 s |
|---|---|---|---|---|---|---|
| HM-Net | original | 13.15 / 16.49 | 14.65 / 16.67 | 0.95 | 10.68 | 22.01 |
| HM-Net + SE-TG | original | 12.69 / 15.58 | 12.61 / 12.82 | 0.41 | 10.44 | 20.91 |
| HM-Net | corrected | 12.85 / 16.19 | 14.27 / 15.81 | 0.22 | 9.97 | 22.84 |
| HM-Net + SE-TG | corrected | 12.96 / 13.83 | 15.20 / 17.95 | 0.35 | 10.87 | 21.69 |

## Table 3 — Paired comparisons (same queries; 95% bootstrap CI, 10,000 resamples; exact McNemar on top-1 hits at IoU ≥ 0.5)

H0: no difference between A and B. 12 comparisons, no multiple-testing correction — treat single borderline intervals with caution.

| Comparison (B − A) | Subset | Δ R1@0.5 [95% CI] | McNemar p (only-A / only-B hits) | Δ mAP avg [95% CI] | Decision |
|---|---|---|---|---|---|
| SE-TG vs baseline (original settings) | all | -1.35 [-3.29, +0.65] | 0.203 (134/113) | -0.70 [-1.61, +0.21] | fail to reject H0 |
| SE-TG vs baseline (original settings) | simple | -0.91 [-3.04, +1.22] | 0.446 (110/98) | -0.46 [-1.40, +0.50] | fail to reject H0 |
| SE-TG vs baseline (original settings) | compound | -3.85 [-8.97, +1.71] | 0.200 (24/15) | -2.04 [-4.57, +0.47] | fail to reject H0 |
| SE-TG vs baseline (corrected settings) | all | -1.68 [-3.81, +0.45] | 0.142 (158/132) | +0.23 [-0.67, +1.14] | fail to reject H0 |
| SE-TG vs baseline (corrected settings) | simple | -2.36 [-4.64, +0.00] | 0.055 (138/107) | +0.10 [-0.87, +1.10] | fail to reject H0 |
| SE-TG vs baseline (corrected settings) | compound | +2.14 [-3.42, +7.69] | 0.551 (20/25) | +0.92 [-1.50, +3.30] | fail to reject H0 |
| corrected vs original (baseline) | all | -0.39 [-2.58, +1.87] | 0.776 (157/151) | -0.31 [-1.27, +0.65] | fail to reject H0 |
| corrected vs original (baseline) | simple | -0.30 [-2.74, +2.13] | 0.852 (132/128) | -0.30 [-1.33, +0.76] | fail to reject H0 |
| corrected vs original (baseline) | compound | -0.85 [-6.84, +5.13] | 0.885 (25/23) | -0.38 [-2.81, +2.13] | fail to reject H0 |
| corrected vs original (SE-TG) | all | -0.71 [-2.97, +1.61] | 0.576 (165/154) | +0.62 [-0.29, +1.51] | fail to reject H0 |
| corrected vs original (SE-TG) | simple | -1.75 [-4.26, +0.61] | 0.181 (147/124) | +0.27 [-0.70, +1.25] | fail to reject H0 |
| corrected vs original (SE-TG) | compound | +5.13 [-0.43, +10.68] | 0.111 (18/30) | +2.58 [+0.19, +5.12] | mAP CI excludes 0 (single borderline result) |

## Table 4 — Prediction behaviour (top-1 window)

| | Ends at video end (≤2 s) | Mean length (s) | Distinct top-1 windows (of 1,550) |
|---|---|---|---|
| Ground truth | 22.9% (has a window there) | 31.9 | — |
| HM-Net (original) | 22.5% | 43.8 | 1380 |
| HM-Net + SE-TG (original) | 8.5% | 41.7 | 1505 |
| HM-Net (corrected) | 16.1% | 47.7 | 1521 |
| HM-Net + SE-TG (corrected) | 62.1% | 36.2 | 1053 |
