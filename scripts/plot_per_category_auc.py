"""Plot per-category peak compliance AUC with 5-fold-CV uncertainty bars.

Reads pooled_lxt_per_category.json (which has CV AUC + std per
(category, layer, turn)), takes the peak over (layer, turn) per category,
and writes `figures/per_category_auc.{pdf,png}`.
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON_PATH = "/Users/julianquick/portfolio_copy/turnstile/experiments/outcome_probe_v1/pooled_lxt_per_category.json"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"

# t_{0.975, df=4} for 5-fold mean 95% CI: half-width = t/sqrt(5) * std
T_CRIT_4 = 2.776
HALFWIDTH_95 = T_CRIT_4 / math.sqrt(5)  # ≈ 1.241


def main():
    d = json.load(open(JSON_PATH))

    rows = []
    for cat, by_L in d["per_category"].items():
        best = None
        for L, by_T in by_L.items():
            for T, cell in by_T.items():
                if best is None or cell["auc"] > best["auc"]:
                    best = {**cell, "layer": L, "turn": T}
        rows.append({"cat": cat, **best})
    rows.sort(key=lambda r: r["auc"])

    # global peak across all layers/turns for the pooled set
    glob_peak = max(cell["auc"]
                    for by_T in d["global"].values()
                    for cell in by_T.values())

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    y = list(range(len(rows)))
    aucs = [r["auc"] for r in rows]
    stds = [r["std"] for r in rows]
    ci95 = [HALFWIDTH_95 * s for s in stds]

    ax.barh(y, aucs, color="#3b6fb0", edgecolor="black",
            linewidth=0.5, alpha=0.92)
    ax.errorbar(aucs, y, xerr=ci95, fmt="none", ecolor="black",
                capsize=3.5, linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels([r["cat"] for r in rows], fontsize=9)

    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.7,
               label="chance (0.5)")
    ax.axvline(glob_peak, color="#d65a31", linestyle="--", linewidth=0.9,
               label=f"pooled-corpus peak ({glob_peak:.3f})")

    for yi, r in enumerate(rows):
        L = r["layer"].lstrip("L"); T = r["turn"].lstrip("T")
        lo = r["auc"] - HALFWIDTH_95 * r["std"]
        hi = r["auc"] + HALFWIDTH_95 * r["std"]
        ax.text(r["auc"] + 0.012 + HALFWIDTH_95 * r["std"], yi,
                f"{r['auc']:.3f} [{lo:.3f}, {hi:.3f}]  "
                f"L{L}/T{T}  n={r['n_pos']}/{r['n']}",
                va="center", fontsize=8)

    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("compliance AUC (peak over layers + turns), "
                  "95% CI from 5-fold CV ($t_4$)")
    ax.set_title("Per-category peak compliance AUC on 9,400-conv pool",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/per_category_auc.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)

    print(f"\nPooled-corpus global peak AUC: {glob_peak:.4f}")
    print(f"\nPer-category peak AUC (5-fold CV; 95% CI half-width = {HALFWIDTH_95:.3f} × std):")
    print(f"{'category':<30}  {'AUC':>6}  {'95% CI':>15}  layer/turn  n_pos/n")
    for r in sorted(rows, key=lambda r: -r["auc"]):
        lo = r["auc"] - HALFWIDTH_95 * r["std"]
        hi = r["auc"] + HALFWIDTH_95 * r["std"]
        print(f"  {r['cat']:<28}  {r['auc']:>6.4f}  [{lo:.4f}, {hi:.4f}]"
              f"   {r['layer']}/{r['turn']}    {r['n_pos']}/{r['n']}")


if __name__ == "__main__":
    main()
