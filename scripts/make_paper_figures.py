"""Regenerate two paper figures from local data:

1. `depth_profile.pdf` — new headline figure for §5 showing the held-out
   single-prompt depth profile of the outcome-direction compliance steering
   (Table 12), with paired-bootstrap 95% CIs.  Replaces the deprecated
   `steering_comparison.pdf` (which visualized only the L16/L20/L31 sparse
   sweep and was overturned by the L8 finding).

2. `sae_ablation_v2.pdf` — corrected SAE-feature-ablation figure with title
   ``Not monotonic'' (the original `sae_ablation.pdf` had the now-deprecated
   title ``Markers, Not Mechanisms'' baked into the image).

Both figures read their numbers from `paper/aux/bootstrap_cis.json` and the
SAE table in `main.tex`, so this script can be re-run if the underlying
data is updated.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_DIR = "/Users/julianquick/portfolio_copy/turnstile/paper"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"


def fig_depth_profile():
    """Held-out depth profile (Table 12) with paired-bootstrap 95% CIs."""
    cis = json.load(open(f"{PAPER_DIR}/aux/bootstrap_cis.json"))
    t12 = cis["table12_depth"]
    layers = [4, 8, 12, 16, 20, 24, 31]
    alpha_cols = ["-0.50", "+0.25", "+0.50"]
    alpha_labels = [
        r"$\alpha = -0.5\,\|h\|$ (suppress)",
        r"$\alpha = +0.25\,\|h\|$ (amplify)",
        r"$\alpha = +0.5\,\|h\|$ (amplify)",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    x = np.arange(len(layers))
    width = 0.27
    offsets = [-width, 0, width]
    colors = ["#3b6fb0", "#d65a31", "#f0b840"]

    for k, (alpha, label, off, color) in enumerate(zip(alpha_cols, alpha_labels, offsets, colors)):
        deltas = []
        los = []
        his = []
        for L in layers:
            cell = t12[f"L{L}"]["deltas"].get(alpha)
            if cell is None:
                deltas.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            deltas.append(cell["delta_pp"])
            los.append(cell["ci95_lo"])
            his.append(cell["ci95_hi"])
        deltas = np.array(deltas)
        los = np.array(los)
        his = np.array(his)
        err = np.stack([deltas - los, his - deltas])
        bars = ax.bar(x + off, deltas, width, yerr=err, label=label,
                      color=color, ecolor="black", capsize=2.5, linewidth=0.4,
                      edgecolor="black")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel(r"$\Delta$ unsafe (pp) vs.\ $\alpha=0$ baseline (73%)")
    ax.set_xlabel("Steering layer (Llama-3.1-8B-Instruct, $\\mathrm{lr\\_compliance}$ direction)")
    ax.set_title("Held-out depth profile of compliance steering: L8 dominates", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, ncol=1)
    ax.set_ylim(-65, 25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    # Annotate L8 peak
    l8_idx = layers.index(8)
    ax.annotate("L8 peak", xy=(l8_idx - width, -49), xytext=(l8_idx - 0.9, -60),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))

    fig.tight_layout()
    out = f"{FIG_DIR}/depth_profile.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_sae_ablation():
    """SAE feature ablation, corrected title."""
    # From tab:sae_ablation in main.tex
    conds = ["Baseline", "Clamp\npersistent (4F)", "Clamp\ntop-diff (3F)",
             "Amplify\n2F (×3)", "Clamp\n4 random"]
    asr  = [25, 22, 18, 19, 26]
    base = 25

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = ["#888888", "#3b6fb0", "#3b6fb0", "#d65a31", "#bbbbbb"]
    bars = ax.bar(range(len(conds)), asr, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(base, color="black", linewidth=0.5, linestyle=":", label="baseline (25%)")

    for i, (a, b) in enumerate(zip(asr, bars)):
        ax.text(i, a + 0.6, f"{a}%", ha="center", fontsize=9)
        if i != 0:
            d = a - base
            ax.text(i, 1.5, f"$\\Delta={d:+d}$~pp", ha="center", fontsize=8.5, color="#222222")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, fontsize=8.5)
    ax.set_ylabel("ASR (%, n=100 JBB goals)")
    ax.set_title("SAE feature ablation: not monotonic", fontsize=11)
    ax.set_ylim(0, 32)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = f"{FIG_DIR}/sae_ablation_v2.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_depth_profile()
    fig_sae_ablation()
