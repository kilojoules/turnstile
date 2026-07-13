"""Per-category ASR vs compliance-AUC scatter, with error bars on both axes.

Data: experiments/outcome_probe_v1/within_category_cv_pooled.json
  - x: compliance-prediction AUC (L16, T0); xerr = 5-fold CV std
  - y: ASR = n_pos / n; yerr = binomial SE = sqrt(p(1-p)/n)
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

ROOT = Path("/Users/julianquick/portfolio_copy/turnstile")
DATA = ROOT / "experiments" / "outcome_probe_v1" / "within_category_cv_pooled.json"
OUT = Path(__file__).parent / "asr_vs_auc_scatter.png"


def load_rows():
    blob = json.loads(DATA.read_text())
    rows = []
    for cat, s in blob["per_category"].items():
        n, n_pos = s["n"], s["n_pos"]
        p = n_pos / n
        rows.append({
            "category": cat,
            "n": n,
            "asr": p,
            "asr_se": math.sqrt(p * (1 - p) / n),
            "auc": s["auc_mean"],
            "auc_std": s["auc_std"],
        })
    return rows


def main():
    rows = load_rows()
    auc = np.array([r["auc"] for r in rows])
    auc_err = np.array([r["auc_std"] for r in rows])
    asr = np.array([r["asr"] for r in rows])
    asr_err = np.array([r["asr_se"] for r in rows])
    cats = [r["category"] for r in rows]

    pearson_r, pearson_p = scipy_stats.pearsonr(auc, asr)
    spearman_r, spearman_p = scipy_stats.spearmanr(auc, asr)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.errorbar(auc, asr, xerr=auc_err, yerr=asr_err,
                fmt="o", color="#2c3e50", ecolor="#888888",
                elinewidth=1, capsize=3, markersize=6, zorder=3)

    offsets = {  # tiny per-label nudges so they don't overlap each other / bars
        "Fraud/Deception": (8, 4),
        "Malware/Hacking": (8, -4),
        "Privacy": (8, 4),
        "Government decision-making": (8, -10),
        "Expert advice": (8, 4),
        "Economic harm": (-10, 8),
        "Physical harm": (8, -4),
        "Disinformation": (8, 4),
        "Harassment/Discrimination": (-10, -12),
        "Sexual/Adult content": (8, 0),
    }
    for cat, x, y in zip(cats, auc, asr):
        dx, dy = offsets.get(cat, (8, 4))
        ax.annotate(cat, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8)

    slope, intercept, *_ = scipy_stats.linregress(auc, asr)
    xs = np.linspace(auc.min() - 0.01, auc.max() + 0.01, 50)
    ax.plot(xs, slope * xs + intercept, color="#c0392b", linewidth=1, linestyle="--",
            label=f"OLS fit  (slope={slope:+.2f})")

    ax.set_xlabel("Compliance-prediction AUC  (L16, turn 0; ±5-fold CV std)")
    ax.set_ylabel("Attack Success Rate  (±binomial SE)")
    ax.set_title("ASR vs Compliance Prediction by Category\n"
                 f"Pearson r = {pearson_r:+.2f} (p = {pearson_p:.2f}),  "
                 f"Spearman ρ = {spearman_r:+.2f} (p = {spearman_p:.2f}),  n = 10")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"Pearson r={pearson_r:+.3f} p={pearson_p:.3f}; "
          f"Spearman ρ={spearman_r:+.3f} p={spearman_p:.3f}")


if __name__ == "__main__":
    main()
