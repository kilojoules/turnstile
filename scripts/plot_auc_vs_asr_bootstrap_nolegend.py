"""Re-render the bootstrap scatter without the category legend (points
are already labeled in-figure). Loads the existing
joint_bootstrap_auc_asr.json so no recomputation is needed.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

OUT_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/outcome_probe_v1"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"


def main():
    d = json.load(open(f"{OUT_DIR}/joint_bootstrap_auc_asr.json"))
    results = d["per_category"]
    cats = sorted(results.keys(), key=lambda c: -results[c]["asr_mean"])

    asrs = [results[c]["asr_mean"] * 100 for c in cats]
    asr_lo = [(results[c]["asr_mean"] - results[c]["asr_lo95"]) * 100
              for c in cats]
    asr_hi = [(results[c]["asr_hi95"] - results[c]["asr_mean"]) * 100
              for c in cats]
    aucs = [results[c]["auc_mean"] for c in cats]
    auc_lo = [results[c]["auc_mean"] - results[c]["auc_lo95"] for c in cats]
    auc_hi = [results[c]["auc_hi95"] - results[c]["auc_mean"] for c in cats]

    r_p, p_p = pearsonr(asrs, aucs)
    r_s, p_s = spearmanr(asrs, aucs)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    cmap = plt.get_cmap("tab10")

    ax.errorbar(asrs, aucs,
                xerr=[asr_lo, asr_hi], yerr=[auc_lo, auc_hi],
                fmt="none", ecolor="black", linewidth=0.8,
                capsize=2.5, alpha=0.85, zorder=2)
    for i, c in enumerate(cats):
        ax.scatter(asrs[i], aucs[i], s=80, color=cmap(i % 10),
                   edgecolor="black", linewidth=0.5, zorder=3, label=c)

    ax.set_xlabel("per-category ASR (%, conversation-level bootstrap 95% CI)")
    ax.set_ylabel("per-category compliance AUC "
                  "(per-turn label, OOB, same bootstrap 95% CI)")
    ax.set_title(f"Compliance AUC vs ASR by category\n"
                 f"(joint conversation-level bootstrap, B={d['n_boot_target']}, "
                 f"no leakage)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8.5, frameon=False, title="category",
              title_fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/auc_vs_asr_scatter_bootstrap.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
