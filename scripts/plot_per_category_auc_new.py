"""Clean bar plot of per-category peak compliance AUC under per-turn labels,
with 95% CIs from 5-fold CV ($t_4$).
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
    d = json.load(open(f"{OUT_DIR}/per_turn_label_per_category.json"))

    rows = []
    for cat, by_L in d["per_category_per_layer"].items():
        best = None
        for L, cell in by_L.items():
            if best is None or cell["auc"] > best["auc"]:
                best = {**cell, "layer": L}
        rows.append({"cat": cat, **best})
    rows.sort(key=lambda r: r["auc"])

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    yy = list(range(len(rows)))
    aucs = [r["auc"] for r in rows]
    ci = [HALFWIDTH_95 * r["std"] for r in rows]

    ax.barh(yy, aucs, color="#3b6fb0", edgecolor="black",
            linewidth=0.5, alpha=0.92)
    ax.errorbar(aucs, yy, xerr=ci, fmt="none", ecolor="black",
                capsize=3.5, linewidth=1.0)

    ax.set_yticks(yy)
    ax.set_yticklabels([r["cat"] for r in rows], fontsize=9)

    for yi, (r, c) in enumerate(zip(rows, ci)):
        ax.text(r["auc"] + c + 0.012, yi,
                f"{r['auc']:.3f} ± {c:.3f}  (L{r['layer'].lstrip('L')}, "
                f"n={r['n_pos']}/{r['n']})",
                va="center", fontsize=8)

    ax.set_xlim(0.5, 0.92)
    ax.set_xlabel("compliance AUC (peak over layers), 95% CI from 5-fold CV ($t_4$)")
    ax.set_title("Per-category compliance AUC, per-turn labels (9,400-conv pool)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/per_category_auc_new.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
