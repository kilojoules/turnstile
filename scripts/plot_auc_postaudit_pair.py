"""Matched PRIOR / POSTERIOR pair of auc_by_layer_postaudit figures — two SEPARATE
figures, identical style, differing only by locus. Both read the SAME extraction
(experiments/postresponse_alllayer/auc_by_layer.json): compliance = replay_v2 per-turn
complied-vs-refused (n=1000); harm = Stage-B full-600 >=4-vs-<=2 (n=565, so the MD line
is exactly the steered harm_dm_llama / comp_dm_out recipe, same per-fold estimator at
every layer).

  auc_by_layer_postaudit.png            <- PRIOR  (pre-response, *_in columns)
  auc_by_layer_postaudit_posterior.png  <- POSTERIOR (response-token, *_out columns)
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
R = json.load(open(f"{ROOT}/experiments/postresponse_alllayer/auc_by_layer.json"))
LAYERS = R["layers"]; x = np.array(LAYERS)
RED, BLUE = "#cc3322", "#3b6fb0"  # RED = Llama labels, BLUE = Qwen labels
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#4d4d4d", "figure.facecolor": "white"})


def render(locus, out_name, locus_label):
    s = "in" if locus == "prior" else "out"
    def C(panel, base, judge): return np.array([R[panel][f"L{L}"][f"{base}_{s}{judge}"] for L in LAYERS])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for ax, panel, title in [(axes[0], "comp", "Compliance probe (per-turn complied vs refused, both judges)"),
                             (axes[1], "harm", "Harm probe (Stage-B ≥4 vs ≤2, full-600 · MD = steered direction)")]:
        ax.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.5, label="chance (0.5)")
        ax.plot(x, C(panel, "lr", ""),  "-o",  color=RED,  lw=1.7, ms=7, label="LR (Llama labels)")
        ax.plot(x, C(panel, "lr", "_q"), "-o",  color=BLUE, lw=1.7, ms=7, label="LR (Qwen labels)")
        ax.plot(x, C(panel, "md", ""),  "--s", color=RED,  lw=1.7, ms=7, alpha=0.7, label="MD (Llama)")
        ax.plot(x, C(panel, "md", "_q"), "--s", color=BLUE, lw=1.7, ms=7, alpha=0.7, label="MD (Qwen)")
        ax.set_xticks(LAYERS); ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
        ax.set_xlabel("residual-stream layer"); ax.set_ylim(0.45, 1.0); ax.grid(alpha=0.25, lw=0.4)
        ax.set_title(title, fontsize=10.5)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)
    axes[0].set_ylabel("grouped 5-fold CV AUC")
    fig.suptitle(locus_label, fontsize=10.5)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{ROOT}/figures/{out_name}.{e}", dpi=150 if e == "png" else None, bbox_inches="tight")
        print(f"wrote figures/{out_name}.{e}")
    plt.close(fig)


render("prior", "auc_by_layer_postaudit",
       "Probe AUC by layer — PRIOR (pre-response) locus  ·  compliance=replay_v2 (n=1000), harm=Stage-B full-600 ≥4-vs-≤2 (n=565)")
render("posterior", "auc_by_layer_postaudit_posterior",
       "Probe AUC by layer — POSTERIOR (response-token) locus  ·  compliance=replay_v2 (n=1000), harm=Stage-B full-600 ≥4-vs-≤2 (n=565)")

# quick delta summary
for panel in ("comp", "harm"):
    d = np.mean([R[panel][f"L{L}"]["lr_out"] - R[panel][f"L{L}"]["lr_in"] for L in LAYERS])
    print(f"  {panel}: mean LR posterior−prior (Llama) = {d:+.3f}")
