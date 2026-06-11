"""Scatter: per-category compliance AUC (per-turn label) vs per-category
ASR on the 9,400-conv pool. Vertical bars = 95% CI on AUC (5-fold CV
$t_4$); horizontal bars = Wilson 95% CI on ASR.
"""
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

OUT_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/outcome_probe_v1"
POOL_ROOT = "/Users/julianquick/portfolio_copy/turnstile/experiments/pooled_hs"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"

T_CRIT_4 = 2.776
HALFWIDTH_95_AUC = T_CRIT_4 / math.sqrt(5)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_asr_per_cat():
    wins, trials = defaultdict(int), defaultdict(int)
    for run_dir in sorted(glob.glob(f"{POOL_ROOT}/*/")):
        for path in sorted(glob.glob(f"{run_dir}/round_*.pt")):
            blob = torch.load(path, weights_only=False)
            cats = blob["categories"]; labels = blob["labels"]
            for c, lab in zip(cats, labels):
                trials[c] += 1
                wins[c] += int(bool(lab))
    return wins, trials


def main():
    auc = json.load(open(f"{OUT_DIR}/per_turn_label_per_category.json"))

    # peak per category
    peak = {}
    for cat, by_L in auc["per_category_per_layer"].items():
        best = None
        for L, cell in by_L.items():
            if best is None or cell["auc"] > best["auc"]:
                best = {**cell, "layer": L}
        peak[cat] = best

    wins, trials = load_asr_per_cat()

    cats = sorted(peak.keys())
    asrs = [100 * wins[c] / trials[c] for c in cats]
    wcis = [wilson_ci(wins[c], trials[c]) for c in cats]
    asr_lo = [100 * (w - lo) / w if w else 0
              for w, (lo, _) in zip(asrs, wcis)]
    asr_hi = [100 * (hi) - a for a, (_, hi) in zip(asrs, wcis)]
    asr_err_lo = [a - 100 * lo for a, (lo, _) in zip(asrs, wcis)]
    asr_err_hi = [100 * hi - a for a, (_, hi) in zip(asrs, wcis)]
    aucs = [peak[c]["auc"] for c in cats]
    auc_err = [HALFWIDTH_95_AUC * peak[c]["std"] for c in cats]

    # correlations
    r_p, p_p = pearsonr(asrs, aucs)
    r_s, p_s = spearmanr(asrs, aucs)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    cmap = plt.get_cmap("tab10")

    ax.errorbar(asrs, aucs,
                xerr=[asr_err_lo, asr_err_hi],
                yerr=auc_err,
                fmt="none", ecolor="black", linewidth=0.8, capsize=2.5,
                alpha=0.85, zorder=2)
    for i, c in enumerate(cats):
        ax.scatter(asrs[i], aucs[i], s=80, color=cmap(i % 10),
                   edgecolor="black", linewidth=0.5, zorder=3, label=c)

    # label each point
    label_offsets = {  # avoid overlap
        "Fraud/Deception": (8, -2),
        "Malware/Hacking": (8, 2),
        "Privacy": (8, 0),
        "Government decision-making": (8, 0),
        "Expert advice": (-8, 8),
        "Economic harm": (-8, -10),
        "Physical harm": (8, 0),
        "Disinformation": (-8, 8),
        "Harassment/Discrimination": (-8, -10),
        "Sexual/Adult content": (8, 0),
    }
    for i, c in enumerate(cats):
        dx, dy = label_offsets.get(c, (8, 0))
        ax.annotate(c, (asrs[i], aucs[i]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5,
                    ha="left" if dx > 0 else "right",
                    va="center")

    # least-squares fit line (visual aid)
    m, b = np.polyfit(asrs, aucs, 1)
    xs = np.linspace(min(asrs) - 2, max(asrs) + 2, 100)
    ax.plot(xs, m * xs + b, color="gray", linestyle="--", linewidth=0.9,
            alpha=0.7,
            label=f"linear fit (Pearson r={r_p:+.2f}, p={p_p:.2f}; "
                  f"Spearman ρ={r_s:+.2f}, p={p_s:.2f})")

    ax.set_xlabel("per-category ASR (%, Wilson 95% CI, full pool n=9,400)")
    ax.set_ylabel("per-category peak compliance AUC (per-turn label, "
                  "5-fold-CV 95% CI)")
    ax.set_title("Compliance AUC vs ASR by category",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True,
              framealpha=0.93)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/auc_vs_asr_scatter.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)

    print()
    print(f"Pearson r = {r_p:+.3f}, p = {p_p:.3f}")
    print(f"Spearman ρ = {r_s:+.3f}, p = {p_s:.3f}")
    print()
    print(f"{'category':<30}  {'ASR':>7}      {'AUC':>11}")
    for c, a, au, ae in sorted(zip(cats, asrs, aucs, auc_err),
                                key=lambda t: -t[1]):
        print(f"  {c:<28}  {a:>5.1f}%   {au:.3f} ± {ae:.3f}")


if __name__ == "__main__":
    main()
