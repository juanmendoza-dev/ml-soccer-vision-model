# 07 — Evaluation

## Splits
- By match: ~70% train / 15% val / 15% test. Test matches untouched until final reporting.

## Metrics
| Metric | Why |
|---|---|
| PR-AUC | Main metric; positives are rare |
| ROC-AUC | Comparable to papers |
| Brier score + calibration curve | Probabilities must mean what they say |
| Lead time | For each shot, seconds between probability first crossing threshold τ and the shot; report median and distribution |
| False alarms per match | At the same τ; keeps lead time honest |

## Required comparisons
1. Baseline vs. frame GNN vs. temporal GNN.
2. With vs. without player profiles.
3. H = 3 s vs. H = 5 s.
4. Dataset tracking vs. vision-pipeline tracking on the same matches, if available (measures how much vision errors hurt).

## Outputs
- `evaluation/report.md` generated per run: metrics table, calibration plot, lead-time histogram.
- Every run logs config + git commit.
