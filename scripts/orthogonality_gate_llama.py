"""Phase 3 GATE: re-confirm geometric orthogonality under all-Llama labels.

Cross-concept cosines must stay in the random-pair band at mid layers
(L8-L24) for the migration to proceed to Phase 4. If they materially
exit the band, the original orthogonality was partly a judge artifact
and the plan stops for a human decision.

Loads:
  directions/v_{lr,md}_comp_L{L}.pt           (Llama-labeled, unchanged)
  directions_llama/v_{lr,md}_harm_L{L}.pt     (Phase 2 Llama-relabeled)

Also for comparison:
  directions/v_{lr,md}_harm_L{L}.pt           (Qwen-labeled, original)

Outputs:
  prints a per-layer table
  writes experiments/steering_v3/layer_sweep/orthogonality_gate_llama.json
"""
import json
import math
import os
import numpy as np
import torch

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
LSW = f"{ROOT}/experiments/steering_v3/layer_sweep"
DIRS_OLD = f"{LSW}/directions"        # Qwen-harm + Llama-comp (current state)
DIRS_LLAMA = f"{LSW}/directions_llama"  # Llama-harm only (Phase 2 output)
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def load(path):
    return torch.load(path, weights_only=False).float().numpy()


def main():
    rand_sd = 1.0 / math.sqrt(4096)
    print(f"\nRandom-pair 1σ band (4096-dim): ±{rand_sd:.4f}\n")

    print(f"{'L':>3}  {'cos(LRh_L, LRc)':>16}  {'cos(MDh_L, MDc)':>16}  "
          f"{'cos(LRh_L, MDh_L)':>18}  {'cos(LRh_L, LRh_Q)':>18}")
    print("-" * 92)
    rows = []
    for L in LAYERS:
        v_lr_comp = load(f"{DIRS_OLD}/v_lr_comp_L{L}.pt")
        v_md_comp = load(f"{DIRS_OLD}/v_md_comp_L{L}.pt")
        v_lr_harm_q = load(f"{DIRS_OLD}/v_lr_harm_L{L}.pt")
        v_md_harm_q = load(f"{DIRS_OLD}/v_md_harm_L{L}.pt")
        v_lr_harm_l = load(f"{DIRS_LLAMA}/v_lr_harm_L{L}.pt")
        v_md_harm_l = load(f"{DIRS_LLAMA}/v_md_harm_L{L}.pt")

        cos_lr_hc_llama = float(np.dot(v_lr_harm_l, v_lr_comp))
        cos_md_hc_llama = float(np.dot(v_md_harm_l, v_md_comp))
        cos_internal_llama = float(np.dot(v_lr_harm_l, v_md_harm_l))
        cos_qwen_to_llama = float(np.dot(v_lr_harm_l, v_lr_harm_q))
        rows.append({
            "layer": L,
            "cos_lr_hc_llama": cos_lr_hc_llama,
            "cos_md_hc_llama": cos_md_hc_llama,
            "cos_internal_harm_llama": cos_internal_llama,
            "cos_lrharm_qwen_vs_llama": cos_qwen_to_llama,
            "cos_lr_hc_qwen": float(np.dot(v_lr_harm_q, v_lr_comp)),
            "cos_md_hc_qwen": float(np.dot(v_md_harm_q, v_md_comp)),
        })
        flag_lr = " **" if abs(cos_lr_hc_llama) > 2 * rand_sd else "   "
        flag_md = " **" if abs(cos_md_hc_llama) > 2 * rand_sd else "   "
        print(f"L{L:<2}  {cos_lr_hc_llama:+16.4f}{flag_lr}  {cos_md_hc_llama:+16.4f}{flag_md}  "
              f"{cos_internal_llama:+18.4f}  {cos_qwen_to_llama:+18.4f}")

    print(f"\n(** flag = |cos| > 2σ of random-pair band = {2*rand_sd:.4f})")
    print()

    # Gate decision: pass if cross-concept cosines stay within ±3σ at L8-L24
    mid = [r for r in rows if r["layer"] in (8, 12, 16, 20, 24)]
    max_lr = max(abs(r["cos_lr_hc_llama"]) for r in mid)
    max_md = max(abs(r["cos_md_hc_llama"]) for r in mid)
    pass_thresh = 3 * rand_sd
    gate_pass = (max_lr <= pass_thresh) and (max_md <= pass_thresh)

    print("=" * 60)
    print(f"GATE check: cross-concept cosines at mid layers (L8–L24)")
    print(f"  max |cos(LR-harm_llama, LR-comp)| = {max_lr:.4f}")
    print(f"  max |cos(MD-harm_llama, MD-comp)| = {max_md:.4f}")
    print(f"  threshold (3σ random-pair):       {pass_thresh:.4f}")
    print(f"  GATE: {'PASS' if gate_pass else '**FAIL**'}")
    print("=" * 60)
    print()
    print("Same-concept consistency check (harm probe Llama vs Qwen):")
    cqvl = [r["cos_lrharm_qwen_vs_llama"] for r in rows if r["layer"] in (8,12,16,20,24)]
    print(f"  median cos(LR-harm_qwen, LR-harm_llama) at L8-L24 = {np.median(cqvl):+.3f}")
    print(f"  (if very low, the harm direction itself materially moved under the judge swap)")

    out = {"rand_sd_floor": rand_sd, "pass_threshold": pass_thresh,
           "max_cross_lr_at_mid": max_lr, "max_cross_md_at_mid": max_md,
           "gate_pass": gate_pass, "per_layer": rows}
    with open(f"{LSW}/orthogonality_gate_llama.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {LSW}/orthogonality_gate_llama.json")


if __name__ == "__main__":
    main()
