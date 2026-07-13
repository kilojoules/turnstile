# compliance_obliteration_v1

One-time experiment. LoRA+SFT on Llama-3.1-8B-Instruct over the existing
victim responses from the 11-run probe corpus, with a probe-AUC penalty
designed to obliterate the L16 linear compliance signal while preserving
the model's behavior on those same trajectories.

## Loss

    L = L_SFT + beta * (soft_AUC(probe(h_L16), y) - 0.5) ** 2

- `L_SFT` — next-token CE on this turn's assistant tokens (canonical
  multi-turn label masking from `turnstile.model_utils._compute_assistant_spans`).
- `probe` — closed-form ridge LR refit *every batch* on the L16 residuals
  at the pre-generation token position (the same position the canonical
  compliance probe was fit on, per `turnstile.extract_pooled_hs`).
- `soft_AUC` — Wilcoxon-style pairwise sigmoid, `mean(sigma((z+ - z-)/tau))`.
- Symmetric `(AUC - 0.5)**2` so both positive and negative linear
  decodability are penalized equally.
- `beta = 4.0` makes the obliteration term comparable to L_SFT at init.

A long-lived 2-hidden-layer MLP audit probe runs in parallel on detached
residuals (own optimizer, K=3 inner steps per outer step, 2048-element
buffer). Its AUC over training is logged as a nonlinear leakage monitor
but does not affect the model's gradient.

## Data

- 11 sources from `POOL_SOURCES` (selfplay_jbb_v1, stealth/control x2,
  4 DPO arms, frozen_v1, urgency_v1, stealth_jbb_v1) — ~9,400 convs.
- Per-turn Llama JBB compliance labels from
  `experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl`
  (filename stale; values are post-migration Llama).
- Each assistant turn is a flat training example; flattens to ~30k turns.
- Group-aware split: 10% of `(source, round)` groups held out for eval
  (no per-conversation leakage).
- Class-balanced batches (default B+ = B- = 8) so the per-batch ridge
  fit has both classes.

## Outputs

```
adapter/                     # LoRA adapter (safetensors)
audit_probe.pt               # adversarial MLP state
config.json                  # run config
train_log.jsonl              # per-step metrics + interleaved {"_eval": true, ...}
run.log                      # stdout/stderr
.training_done               # marker on clean exit
.training_failed             # marker on crash
```

`train_log.jsonl` row fields: `step`, `sft_loss`, `obl_loss`, `ridge_auc`,
`audit_loss`, `audit_auc`, `elapsed`.

## Launch

```
bash scripts/run_compliance_obliteration.sh
```

A100-80GB recommended. For 40GB add `--quant-4bit` and/or reduce
`--max-length 768` and `--batch-pos 4 --batch-neg 4`.

## What to look for

- `sft_loss` should stay near base SFT loss (~1-2) — model still reproduces
  its own behavior on these trajectories.
- `ridge_auc` should fall from ~0.75 toward ~0.5 within a few hundred steps —
  the linear compliance signal is being pushed out.
- `audit_auc` (nonlinear MLP) decides the outcome. If it stays high while
  `ridge_auc` drops, obliteration is *only* linear (signal still in residuals,
  just rotated). If it falls too, the concept is actually being scrubbed.

Followups (not in this run):
- Eval the adapter on a held-out attack corpus: does ASR / Stage-B Likert
  shift? (Behavior should be preserved on training trajectories but
  could drift on novel attacks — that's the interesting failure mode.)
- Probe AUC by layer post-training: did we just displace the signal to
  L17+?
