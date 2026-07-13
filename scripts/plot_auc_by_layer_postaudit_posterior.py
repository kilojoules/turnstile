"""auc_by_layer_postaudit, POSTERIOR edition: decode AUC by layer for compliance
(left) and harm (right), comparing the PRIOR locus (pre-response, dashed) with the
POSTERIOR locus (mean over the response tokens, solid) on a MATCHED corpus. LR probe
(circles) + mean-diff direction (squares), Llama (blue) and Qwen (red) labels.

Reads experiments/postresponse_alllayer/auc_by_layer.json (written by
compute_auc_postresponse.py). Both loci are extracted with the identical turn_reps
recipe and scored with the identical grouped-CV, so the prior->posterior gap is a
clean locus effect. (Absolute prior differs from the published pooled_hs figure
because this is a re-extraction over the Stage-B / replay corpora, not the 47k pool.)
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
R = json.load(open(f"{ROOT}/experiments/postresponse_alllayer/auc_by_layer.json"))
LAYERS = R["layers"]; x = np.array(LAYERS)
BLUE, RED = "#3b6fb0", "#cc3322"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#4d4d4d", "figure.facecolor": "white"})

def col(panel, key): return np.array([R[panel][f"L{L}"][key] for L in LAYERS])

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3), sharey=True)
for ax, panel, title, extra in [
    (axes[0], "comp", "Compliance probe  (per-turn complied vs refused · replay_v2 n=1000)", ""),
    (axes[1], "harm", "Harm probe  (Stage-B ≥4 vs ≤2 · full-600 n=565 · MD = steered harm_dm_llama)", "")]:
    ax.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.5)
    for jk, jc in [("", BLUE), ("_q", RED)]:
        lr_in, lr_out = col(panel, f"lr_in{jk}"), col(panel, f"lr_out{jk}")
        md_in, md_out = col(panel, f"md_in{jk}"), col(panel, f"md_out{jk}")
        ax.fill_between(x, lr_in, lr_out, color=jc, alpha=0.08, lw=0)
        ax.plot(x, lr_out, "-o", color=jc, lw=2.4, ms=6.5, mec="white", mew=0.6, zorder=4)   # posterior LR
        ax.plot(x, lr_in, "--o", color=jc, lw=1.6, ms=5, alpha=0.55, zorder=3)               # prior LR
        ax.plot(x, md_out, "-s", color=jc, lw=1.3, ms=4.5, alpha=0.5, zorder=2)               # posterior MD
        ax.plot(x, md_in, ":s", color=jc, lw=1.1, ms=4, alpha=0.4, zorder=2)                 # prior MD
    ax.set_xticks(LAYERS); ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlabel("residual-stream layer"); ax.set_ylim(0.45, 0.95); ax.grid(alpha=0.22, lw=0.4)
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("grouped 5-fold CV AUC")

locus_leg = [Line2D([0],[0], color="#555", ls="-", lw=2.4, marker="o", ms=6, mec="white"),
             Line2D([0],[0], color="#555", ls="--", lw=1.6, marker="o", ms=5, alpha=0.6),
             Line2D([0],[0], color="#555", ls="-", lw=1.3, marker="s", ms=4.5, alpha=0.6),
             Line2D([0],[0], color="#555", ls=":", lw=1.1, marker="s", ms=4, alpha=0.5)]
axes[0].legend(locus_leg, ["LR · POSTERIOR (response tokens)", "LR · prior (pre-response)",
                           "MD · posterior", "MD · prior"], loc="lower right", fontsize=8, frameon=False)
judge_leg = [Line2D([0],[0], color=BLUE, lw=2.4), Line2D([0],[0], color=RED, lw=2.4)]
axes[1].legend(judge_leg, ["Llama labels", "Qwen labels"], loc="lower right", fontsize=8.5, frameon=False)

fig.suptitle("Decoding harm and compliance by layer: the POSTERIOR (response-token) state is more predictive than the "
             "PRIOR (pre-response) state\n(matched corpus, identical extraction + grouped CV at both loci; shaded band = "
             "prior→posterior uplift)", fontsize=10.3, y=1.02)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"{ROOT}/figures/auc_by_layer_postaudit_posterior.{e}", dpi=150 if e == "png" else None, bbox_inches="tight")
    print(f"wrote figures/auc_by_layer_postaudit_posterior.{e}")

# headline deltas
for panel in ("comp", "harm"):
    d_lr = np.mean([R[panel][f"L{L}"]["lr_out"] - R[panel][f"L{L}"]["lr_in"] for L in LAYERS])
    print(f"  {panel}: mean LR posterior−prior (Llama) = {d_lr:+.3f}")
