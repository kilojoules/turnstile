# Harm ≠ Compliance — data & code manifest

Everything that goes into the paper *"Harm ≠ Compliance"* (Quick & Guidote, 2026), mapping
each figure to the script that draws it and the data that feeds it. Victim = frozen
`meta-llama/Llama-3.1-8B-Instruct`; steering is an additive forward hook at layer 16
(`α · REF_NORM · unit`, REF_NORM = 7.5626). Judges: local Llama-3.1-70B (JBB compliance) and
Qwen2.5-72B; harm is a Stage-B 1–5 uplift Likert. Generation is greedy (temperature 0), so
every steering/gen result is deterministic and re-runnable.

## Figures → script → input data

| paper figure | script (`scripts/`) | input data (in repo unless noted) |
|---|---|---|
| Adversary-training ASR by run | `plot_asr_by_category.py` / training logs | per-run ASR (self-play round metrics) |
| ASR by JBB category | `plot_asr_by_category.py`, `plot_marginal_asr_by_category.py` | pooled per-turn judgments (9,400 pool) |
| Cumulative ASR vs turn | `plot_cum_asr_by_category.py` | same pool |
| Per-category compliance ROC @ L16 | `plot_per_category_roc.py` | `experiments/pooled_hs/` (L16 residuals) ✱regen |
| **AUC by layer — PRIOR (Fig 5)** | `plot_auc_postaudit_pair.py` | `experiments/postresponse_alllayer/auc_by_layer.json` |
| Stage-A vs Stage-B harm scatter | `plot_stage_b_vs_stage_a.py` | `working/uplift/stage_b_scores_llama.jsonl` |
| **AUC by layer — POSTERIOR (Fig 7)** | `plot_auc_postaudit_pair.py` | `experiments/postresponse_alllayer/auc_by_layer.json` |
| **fig6 monotonic 2-panel (Fig 8)** | `plot_fig6_2panel_directions.py` | `steer_decoded[+ext10]`, `steer_benign_comp`, `refusal_alpha_sweep_v1`, `refusal_harm_vs_compliance_v1`, `steer_cvh_{matched,fill,ext10}` |
| **decoded==steered 3-panel (Fig 9)** | `plot_decoded_steering.py` | `steer_decoded[+ext10]`, `refusal_harm_vs_compliance_v1`, `steer_cvh_*` |
| decoded-steering scatter (compliance panel) | `plot_compliance_directions_lexical.py` | same steering jsonls |

✱regen = regenerable, not committed (see "Large data" below).

## Pipeline (how the derived data is produced)

1. **Directions** — `scripts/build_directions.py` → `directions/*.pt` (committed) from
   `directions/reps.npz` (L16 residuals; regen). Recipes:
   - `refusal_dm` = μ(harmful goals[:50]) − μ(benign alpaca[:50]), pre-response (prior, task-only).
   - `comp_pre_{llama,qwen}` = μ(complied) − μ(refused) per-turn, pre-response (prior, training corpus).
   - `comp_dm_out` = μ(complied) − μ(refused), response-mean (posterior, training corpus).
   - `harm_dm_llama` = μ(rating≥4) − μ(rating≤2), response-mean, full-600 Stage-B.
2. **Steering sweeps** — `scripts/steer_cvh_matched.py` (env-configurable: `CVH_DIRS`,
   `CVH_ALPHAS`, `CVH_SRC=benign`, `CVH_GOALS`) + `scripts/refusal_alpha_sweep.py`.
   Judged by `scripts/judge_postresponse_sweep.py` (`--model` selects the 70B/72B judge).
3. **AUC-by-layer** — `scripts/extract_postresponse_all_layers.py` (both loci, all 9 layers,
   full-600 harm + replay-1000 compliance) → `reps_{harm,comp}.npz` (regen) →
   `scripts/compute_auc_postresponse.py` → `auc_by_layer.json` (committed).
   Seam check inside `compute_auc_postresponse.py` confirms the L16 output-MD == steered
   `harm_dm_llama` (cos = 1.0).
4. **Stage-B corpus** — `working/uplift/select_stage_b.py` (100 behaviors × 3 wins + 3
   *refused* harmful conversations) → `stage_b_candidates.jsonl` + `stage_b_scores_llama.jsonl`.

## Direction taxonomy (three steerable compliance directions + controls)
- prior, task-only: `refusal_dm` (bidirectional refusal switch — jailbreaks harmful *and* over-refuses benign)
- prior, training corpus: `comp_pre_llama`, `comp_pre_qwen`
- posterior, training corpus: `comp_dm_out`
- controls: `random_1..5`; harm directions `harm_dm_llama`, `harm_dm_resid`, `harm_pre_llama` (decode but don't steer)

## Large data — NOT in git (regenerable; see `.gitignore`)
| path | size | how to get it |
|---|---|---|
| `experiments/pooled_hs/` | 6.5 GB | raw L16 per-turn residual pool (9,400 convs); re-extract from the self-play transcripts |
| `experiments/postresponse_alllayer/reps_{comp,harm,harm_winsonly}.npz` | 80–280 MB | `PR_ROOT=. python3 scripts/extract_postresponse_all_layers.py` (needs the 8B model + a GPU) |
| `experiments/clamp_obliterated_v2/replay_v2_full.pt` | 51 MB | replay corpus with hidden states; slim variant `replay_v2_slim.pt` (committed) has conv+labels only |
| `directions/reps.npz` | 33 MB | L16 direction-fit residuals; `build_directions.py` consumes it |
| `experiments/*/adapter/*.safetensors` | 160 MB | DPO/obliteration LoRA adapters |

The small JSON/JSONL under `experiments/postresponse_alllayer/` (`auc_by_layer.json`) and the
steering sweeps are the actual figure inputs and **are** committed — every figure can be
re-drawn from the repo without a GPU; only re-*computing* the residuals needs the large files.
