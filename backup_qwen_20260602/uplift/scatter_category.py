"""Per-category scatter: compliance AUC (y) vs mean uplift among compliant
responses (x), one point per JBB category.

x: mean uplift rating across all "wins" in that category, from
   stage_b_scores.jsonl. Error bar = SE across wins.
y: compliance-prediction AUC for that category, from the existing pooled
   probe corpus (experiments/outcome_probe_v1/within_category_cv_pooled.json).
   Error bar = 5-fold CV std.
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).parent
SCORES = ROOT / "stage_b_scores.jsonl"
AUC_FILE = ROOT.parent.parent / "experiments" / "outcome_probe_v1" / "within_category_cv.json"
OUT = ROOT / "scatter_category_auc_vs_uplift.png"


def main():
    recs = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    auc_raw = json.loads(AUC_FILE.read_text())["per_category"]
    # Normalize: this file uses 'auc_mean' or 'auc' depending on version
    auc_blob = {}
    for cat, s in auc_raw.items():
        auc_blob[cat] = {
            "auc_mean": s.get("auc_mean") if "auc_mean" in s else s.get("auc"),
            "auc_std": s.get("auc_std", 0),
        }

    # Per-category mean uplift among wins
    by_cat = defaultdict(list)
    for r in recs:
        if r["unsafe"] and r["rating"] is not None:
            by_cat[r["category"]].append(r["rating"])

    cats, x, xerr, y, yerr, n_wins = [], [], [], [], [], []
    for cat in sorted(set(by_cat) & set(auc_blob)):
        ratings = by_cat[cat]
        cats.append(cat)
        x.append(statistics.mean(ratings))
        xerr.append(statistics.stdev(ratings) / math.sqrt(len(ratings)) if len(ratings) > 1 else 0)
        y.append(auc_blob[cat]["auc_mean"])
        yerr.append(auc_blob[cat]["auc_std"])
        n_wins.append(len(ratings))

    x, xerr, y, yerr = map(np.array, (x, xerr, y, yerr))
    pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
    slope, intercept, *_ = scipy_stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color="#2c3e50",
                ecolor="#888888", elinewidth=1, capsize=3, markersize=8,
                zorder=3)
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 50)
    ax.plot(xs, slope * xs + intercept, color="#c0392b", linewidth=1,
            linestyle="--", label=f"OLS slope = {slope:+.3f}/Likert pt")

    offsets = {
        "Malware/Hacking": (8, -2),
        "Privacy": (8, 4),
        "Government decision-making": (-12, 8),
        "Fraud/Deception": (8, -10),
        "Economic harm": (8, 4),
        "Expert advice": (8, -10),
        "Disinformation": (-12, 8),
        "Physical harm": (8, 4),
        "Sexual/Adult content": (8, 4),
        "Harassment/Discrimination": (-15, -10),
    }
    for cat, xv, yv, n in zip(cats, x, y, n_wins):
        dx, dy = offsets.get(cat, (8, 4))
        ax.annotate(f"{cat}  (n={n})", (xv, yv), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9)

    ax.axhline(0.5, color="gray", linewidth=0.5)
    ax.set_xlabel("Mean real-world uplift of compliant responses\n"
                  "(Qwen2.5-72B Likert 1-5; ±SE across wins)")
    ax.set_ylabel("Within-category compliance AUC  (L16, T0; train+test inside category)")
    ax.set_title("Per-category compliance AUC vs uplift of compliant responses\n"
                 f"n={len(cats)} categories  (Physical harm, Sexual/Adult omitted: too few wins for within-cat probe)\n"
                 f"Pearson r = {pearson_r:+.2f} (p={pearson_p:.3f}),  "
                 f"Spearman ρ = {spearman_r:+.2f} (p={spearman_p:.3f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"\n{'category':32s}  {'uplift_mean':>11s}  {'SE':>6s}    {'AUC':>6s}  {'std':>6s}    n_wins")
    for cat, xv, xv_e, yv, yv_e, n in zip(cats, x, xerr, y, yerr, n_wins):
        print(f"{cat:32s}  {xv:11.2f}  {xv_e:6.2f}    {yv:6.3f}  {yv_e:6.3f}    {n}")


if __name__ == "__main__":
    main()
