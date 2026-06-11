"""Per-category compliance AUC vs layer, GroupKFold per-turn-label.

One line per category across layers {0, 4, 8, 12, 16, 20, 24, 28, 31};
shaded band = 95% CI from group-stratified 5-fold CV ($t_4$).
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/outcome_probe_v1"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"

T_CRIT_4 = 2.776
HALFWIDTH_95 = T_CRIT_4 / math.sqrt(5)


def main():
    d = json.load(open(f"{OUT_DIR}/per_turn_label_per_category_groupkfold.json"))
    layers = d["layers"]

    # order categories by peak GroupK AUC (descending top legend)
    def peak_auc(cat):
        return max(c["auc"] for c in d["per_category_per_layer"][cat].values())
    cats = sorted(d["per_category_per_layer"].keys(),
                  key=lambda c: -peak_auc(c))

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    cmap = plt.get_cmap("tab10")

    for i, c in enumerate(cats):
        by_L = d["per_category_per_layer"][c]
        aucs = []
        cis = []
        for L in layers:
            cell = by_L.get(f"L{L}")
            if cell is None:
                aucs.append(np.nan)
                cis.append(np.nan)
            else:
                aucs.append(cell["auc"])
                cis.append(HALFWIDTH_95 * cell["std"])
        aucs = np.array(aucs)
        cis = np.array(cis)
        color = cmap(i % 10)
        ax.fill_between(layers, aucs - cis, aucs + cis,
                        color=color, alpha=0.12)
        ax.plot(layers, aucs, marker="o", markersize=4.5, linewidth=1.6,
                color=color, label=c)

    # pooled global
    glob = d["global_per_layer"]
    g_aucs = np.array([glob[f"L{L}"]["auc"] if f"L{L}" in glob else np.nan
                       for L in layers])
    g_cis = np.array([HALFWIDTH_95 * glob[f"L{L}"]["std"]
                      if f"L{L}" in glob else np.nan for L in layers])
    ax.fill_between(layers, g_aucs - g_cis, g_aucs + g_cis,
                    color="black", alpha=0.10)
    ax.plot(layers, g_aucs, marker="s", markersize=5.5, linewidth=2.2,
            color="black", linestyle="--", label="pooled (all 10 cats)")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6,
               alpha=0.6)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_xlabel("Llama-3.1-8B-Instruct layer")
    ax.set_ylabel("compliance AUC (per-turn label, GroupKFold by conv., "
                  "95% CI $t_4$)")
    ax.set_title("Per-category compliance AUC vs layer",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_ylim(0.45, None)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8.5, frameon=False, title="category",
              title_fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/auc_vs_layer_groupkfold.{ext}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
