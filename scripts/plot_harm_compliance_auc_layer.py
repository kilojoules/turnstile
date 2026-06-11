"""Plot harm-supervised AUC vs compliance-supervised AUC, per layer, both
using their own optimally-fit linear coefficients at each layer.

  Compliance probe per layer: GroupKFold per-turn-label probe on 9,400-conv
    pool (file: per_turn_label_per_category_groupkfold.json, global_per_layer)
  Harm probe per layer: 5-fold CV on n=289 Stage B wins, target = harm
    Likert ≥ 4 vs ≤ 3 (file: harm_probe_experiments.json, exp3_per_layer)

Both use linear LR on the raw residual stream features at the given layer.
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]

T_CRIT_4 = 2.776
HALFWIDTH_95 = T_CRIT_4 / math.sqrt(5)


def main():
    comp = json.load(open(f"{OUT}/per_turn_label_per_category_groupkfold.json"))
    harm = json.load(open(f"{OUT}/harm_probe_experiments.json"))

    Ls = LAYERS
    comp_auc = [comp["global_per_layer"][f"L{L}"]["auc"] for L in Ls]
    comp_ci = [HALFWIDTH_95 * comp["global_per_layer"][f"L{L}"]["std"]
               for L in Ls]
    harm_auc = [harm["exp3_per_layer"][f"L{L}"]["auc_oob"] for L in Ls]
    harm_lo = [harm_auc[i] - harm["exp3_per_layer"][f"L{L}"]["auc_ci95_boot"][0]
               for i, L in enumerate(Ls)]
    harm_hi = [harm["exp3_per_layer"][f"L{L}"]["auc_ci95_boot"][1] - harm_auc[i]
               for i, L in enumerate(Ls)]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.errorbar(Ls, comp_auc, yerr=comp_ci,
                marker="o", markersize=6.5, linewidth=1.7,
                color="#3b6fb0", capsize=3,
                label=f"compliance probe  (per-turn label, "
                      f"9,400-conv pool, GroupKFold $t_4$ CI)")
    ax.errorbar(Ls, harm_auc, yerr=[harm_lo, harm_hi],
                marker="s", markersize=6.5, linewidth=1.7,
                color="#d65a31", capsize=3,
                label=f"harm probe  (Stage-B Likert≥4 vs ≤3, "
                      f"n=289 wins, 5-fold CV, bootstrap 95% CI)")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6)
    ax.set_xticks(Ls)
    ax.set_xticklabels([f"L{L}" for L in Ls])
    ax.set_xlabel("Llama-3.1-8B-Instruct layer")
    ax.set_ylabel("AUC (each line uses its own optimal LR coefficients)")
    ax.set_title("Linear probe AUC vs layer: harm-supervised vs "
                 "compliance-supervised",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_ylim(0.48, 0.85)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    # annotate peak layers
    peak_c = int(np.argmax(comp_auc)); peak_h = int(np.argmax(harm_auc))
    ax.annotate(f"compliance peak: L{Ls[peak_c]}\n{comp_auc[peak_c]:.3f}",
                xy=(Ls[peak_c], comp_auc[peak_c]),
                xytext=(Ls[peak_c] - 6, comp_auc[peak_c] + 0.025),
                fontsize=8, color="#1f4170",
                arrowprops=dict(arrowstyle="->", color="#1f4170", lw=0.7))
    ax.annotate(f"harm peak: L{Ls[peak_h]}\n{harm_auc[peak_h]:.3f}",
                xy=(Ls[peak_h], harm_auc[peak_h]),
                xytext=(Ls[peak_h] - 5, harm_auc[peak_h] - 0.055),
                fontsize=8, color="#a04020",
                arrowprops=dict(arrowstyle="->", color="#a04020", lw=0.7))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/harm_compliance_auc_per_layer.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    # print
    print()
    print(f"{'Layer':<6}{'Compliance AUC':>18}{'Harm AUC':>22}")
    for i, L in enumerate(Ls):
        print(f"  L{L:<3}  {comp_auc[i]:.3f} ± {comp_ci[i]:.3f}     "
              f"{harm_auc[i]:.3f} [{harm['exp3_per_layer'][f'L{L}']['auc_ci95_boot'][0]:.3f}, "
              f"{harm['exp3_per_layer'][f'L{L}']['auc_ci95_boot'][1]:.3f}]")


if __name__ == "__main__":
    main()
