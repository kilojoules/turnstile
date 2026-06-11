"""Per-category compliance AUC: per-conversation vs per-turn labels, side
by side with 95% CIs from 5-fold CV (t_4 multiplier).

Reads both result JSONs and writes
`figures/per_category_auc_compare.{pdf,png}`.
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


def peak_per_cat(by_cat_by_L):
    """Return {cat: {auc, std, layer, n, n_pos}} taking max-over-layers."""
    out = {}
    for cat, by_L in by_cat_by_L.items():
        best = None
        for L, cell in by_L.items():
            # cell may be {'auc', 'std', ...} OR nested {Tx: {...}}
            if "auc" in cell:
                if best is None or cell["auc"] > best["auc"]:
                    best = {**cell, "layer": L}
            else:
                for T, c in cell.items():
                    if best is None or c["auc"] > best["auc"]:
                        best = {**c, "layer": L, "turn": T}
        out[cat] = best
    return out


def main():
    old = json.load(open(f"{OUT_DIR}/pooled_lxt_per_category.json"))
    new = json.load(open(f"{OUT_DIR}/per_turn_label_per_category.json"))

    peak_old = peak_per_cat(old["per_category"])
    peak_new = peak_per_cat(new["per_category_per_layer"])

    # rank categories by per-turn AUC (descending top->bottom)
    cats = sorted(peak_new.keys(), key=lambda c: peak_new[c]["auc"])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y_centers = np.arange(len(cats))
    bar_h = 0.36
    y_old = y_centers - bar_h / 2
    y_new = y_centers + bar_h / 2

    aucs_old = [peak_old[c]["auc"] for c in cats]
    aucs_new = [peak_new[c]["auc"] for c in cats]
    err_old = [HALFWIDTH_95 * peak_old[c]["std"] for c in cats]
    err_new = [HALFWIDTH_95 * peak_new[c]["std"] for c in cats]

    ax.barh(y_old, aucs_old, height=bar_h, color="#bcbd22",
            edgecolor="black", linewidth=0.4, alpha=0.92,
            label="per-conversation label (old): "
                  "$y=$ did conv. ever breach?")
    ax.errorbar(aucs_old, y_old, xerr=err_old, fmt="none",
                ecolor="black", capsize=2.5, linewidth=0.9)

    ax.barh(y_new, aucs_new, height=bar_h, color="#3b6fb0",
            edgecolor="black", linewidth=0.4, alpha=0.92,
            label="per-turn label (new): $y=$ is this turn's response unsafe?")
    ax.errorbar(aucs_new, y_new, xerr=err_new, fmt="none",
                ecolor="black", capsize=2.5, linewidth=0.9)

    # global peaks
    glob_old = max(c["auc"] for by_T in old["global"].values()
                   for c in by_T.values())
    glob_new = max(c["auc"] for c in new["global_per_layer"].values())
    ax.axvline(glob_old, color="#bcbd22", linestyle="--", linewidth=0.9,
               label=f"pooled-global peak (old) {glob_old:.3f}")
    ax.axvline(glob_new, color="#3b6fb0", linestyle="--", linewidth=0.9,
               label=f"pooled-global peak (new) {glob_new:.3f}")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.6)

    ax.set_yticks(y_centers)
    ax.set_yticklabels(cats, fontsize=9)

    for yi, c in enumerate(cats):
        o = peak_old[c]; n = peak_new[c]
        d = n["auc"] - o["auc"]
        ax.text(max(o["auc"] + err_old[yi], n["auc"] + err_new[yi]) + 0.012,
                yi, f"$\\Delta=$ {d:+.3f}",
                va="center", fontsize=8.5,
                color="#1a6e1a" if d > 0 else "#a02020")

    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("compliance AUC (peak over layers $\\times$ turns), "
                  "95% CI from 5-fold CV ($t_4$)")
    ax.set_title("Per-category compliance AUC: per-conversation vs per-turn labels",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.93)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/per_category_auc_compare.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)

    print()
    print(f"{'category':<30}  {'old AUC':>11}      {'new AUC':>11}    Δ")
    for c in reversed(cats):
        o = peak_old[c]; n = peak_new[c]
        lo_o = o["auc"] - HALFWIDTH_95 * o["std"]
        hi_o = o["auc"] + HALFWIDTH_95 * o["std"]
        lo_n = n["auc"] - HALFWIDTH_95 * n["std"]
        hi_n = n["auc"] + HALFWIDTH_95 * n["std"]
        d = n["auc"] - o["auc"]
        print(f"  {c:<28}  {o['auc']:.3f} ±{HALFWIDTH_95*o['std']:.3f}  "
              f"{n['auc']:.3f} ±{HALFWIDTH_95*n['std']:.3f}   {d:+.3f}")


if __name__ == "__main__":
    main()
