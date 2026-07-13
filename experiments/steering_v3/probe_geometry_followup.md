## Analysis 1: Mean-diff directions as probes (cross-validated AUC)

All cross-validated with the same fold splits LR was evaluated on.

| Concept | LR AUC | Mean-diff AUC | Pearson r (scores) | Flag |
|---|---|---|---|---|
| Compliance (per-turn, 9,400-conv pool) | 0.771 | 0.693 | +0.221 | |
| Harm (Stage B wins, Likert ≥4 vs ≤3) | 0.750 | 0.735 | +0.872 | |

## Analysis 2: Continuous Likert regression (harm only)

| Direction | r² | Spearman ρ | Pearson r | Flag |
|---|---|---|---|---|
| LR-harm projection vs Likert | 0.196 ⚠️ r² < 0.2 (threshold not magnitude?) | +0.431 | +0.442 | |
| Mean-diff-harm projection vs Likert | 0.177 ⚠️ r² < 0.2 (threshold not magnitude?) | +0.421 | +0.421 | |