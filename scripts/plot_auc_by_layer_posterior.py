"""auc_by_layer at the POSTERIOR (response-token) locus, styled like the original
auc_by_layer_postaudit: two panels (compliance | harm), LR (solid, o) + MD (dashed, s),
Llama (red) + Qwen (blue) labels, chance line at 0.5.

Values are the *_out columns from experiments/postresponse_alllayer/auc_by_layer.json
(response-mean residual, grouped 5-fold CV). Compliance = replay_v2 per-turn (n=1000);
harm = Stage-B wins >=4 vs <4 (n=289, exact figure corpus).
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
R = json.load(open(f"{ROOT}/experiments/postresponse_alllayer/auc_by_layer.json"))
LAYERS = R["layers"]; x = np.array(LAYERS)
RED, BLUE = "#cc3322", "#3b6fb0"   # RED = Llama labels, BLUE = Qwen labels (consistent across panels)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#4d4d4d", "figure.facecolor": "white"})

def col(panel, key): return np.array([R[panel][f"L{L}"][key] for L in LAYERS])

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
for ax, panel, title in [
    (axes[0], "comp", "Compliance probe (per-turn complied vs refused, both judges)"),
    (axes[1], "harm", "Harm probe (Stage-B ≥4 vs ≤2, full-600 · MD = steered harm_dm_llama)")]:
    ax.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.5, label="chance (0.5)")
    ax.plot(x, col(panel, "lr_out"),   "-o",  color=RED,  lw=1.7, ms=7, label="LR (Llama labels)")
    ax.plot(x, col(panel, "lr_out_q"), "-o",  color=BLUE, lw=1.7, ms=7, label="LR (Qwen labels)")
    ax.plot(x, col(panel, "md_out"),   "--s", color=RED,  lw=1.7, ms=7, alpha=0.7, label="MD (Llama)")
    ax.plot(x, col(panel, "md_out_q"), "--s", color=BLUE, lw=1.7, ms=7, alpha=0.7, label="MD (Qwen)")
    ax.set_xticks(LAYERS); ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlabel("residual-stream layer"); ax.set_ylim(0.45, 1.0); ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(title, fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)
axes[0].set_ylabel("grouped 5-fold CV AUC")
fig.suptitle("Probe AUC by layer at the POSTERIOR (response-token) locus — MD line = the exact steered directions "
             "(comp_dm_out, harm_dm_llama), same per-fold estimator at every layer", fontsize=10.2)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"{ROOT}/figures/auc_by_layer_posterior.{e}", dpi=150 if e == "png" else None, bbox_inches="tight")
    print(f"wrote figures/auc_by_layer_posterior.{e}")
