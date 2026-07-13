"""Stage A analysis: per-behavior real-world-uplift rating vs per-behavior
compliance-prediction AUC.

Inputs:
  stage_a_scores.jsonl                            (this directory)
  ../experiments/outcome_probe_v1/per_behavior_L16_T1.json  (existing)

Outputs:
  uplift_vs_auc_per_behavior.png   (n=96 scatter)
  uplift_vs_auc_per_category.png   (n=10 collapsed)
  stage_a_summary.json
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).parent
SCORES = ROOT / "stage_a_scores.jsonl"
AUC_FILE = ROOT.parent.parent / "experiments" / "outcome_probe_v1" / "per_behavior_L16_T1.json"


def load_scores():
    out = {}
    for line in SCORES.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("rating") is None:
            continue
        out[d["behavior"]] = {
            "rating": int(d["rating"]),
            "category": d["category"],
            "rationale": d.get("rationale", ""),
        }
    return out


def load_auc():
    return json.loads(AUC_FILE.read_text())


def per_behavior_panel(ax, scores, auc):
    keys = sorted(set(scores) & set(auc))
    x = np.array([scores[k]["rating"] for k in keys], dtype=float)
    y = np.array([auc[k]["auc"] for k in keys], dtype=float)
    yerr = np.array([auc[k]["std"] for k in keys], dtype=float)
    cats = sorted({scores[k]["category"] for k in keys})
    cmap = plt.get_cmap("tab10")
    color_map = {c: cmap(i) for i, c in enumerate(cats)}
    colors = [color_map[scores[k]["category"]] for k in keys]
    # Jitter x so identical-rating points don't overlap
    rng = np.random.default_rng(0)
    x_jit = x + rng.uniform(-0.15, 0.15, size=len(x))
    ax.errorbar(x_jit, y, yerr=yerr, fmt="none", ecolor="#cccccc",
                elinewidth=0.6, capsize=2, zorder=2)
    ax.scatter(x_jit, y, c=colors, s=30, edgecolor="black", linewidth=0.4,
               zorder=3)
    pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
    slope, intercept, *_ = scipy_stats.linregress(x, y)
    xs = np.linspace(0.7, 5.3, 50)
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.2,
            linestyle="--", label=f"OLS slope = {slope:+.3f}/Likert pt")
    ax.axhline(0.5, color="gray", linewidth=0.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Real-world uplift rating (Qwen2.5-72B, 1=harmless / 5=severe)")
    ax.set_ylabel("Compliance-prediction AUC (L16, T1, ±5-fold CV std)")
    ax.set_title(f"Per-behavior uplift vs compliance AUC  (n={len(keys)})\n"
                 f"Pearson r = {pearson_r:+.2f} (p={pearson_p:.3f}),  "
                 f"Spearman ρ = {spearman_r:+.2f} (p={spearman_p:.3f})")
    handles = [plt.Line2D([], [], marker="o", color=color_map[c],
                          linestyle="", markersize=6, markeredgecolor="black",
                          markeredgewidth=0.4, label=c) for c in cats]
    ax.legend(handles=handles, fontsize=7, loc="upper right",
              ncol=2, framealpha=0.9)
    return pearson_r, pearson_p, spearman_r, spearman_p


def per_category_panel(ax, scores, auc):
    by_cat = {}
    for b, s in scores.items():
        if b not in auc:
            continue
        by_cat.setdefault(s["category"], []).append((s["rating"], auc[b]["auc"]))
    cats = sorted(by_cat.keys())
    x = np.array([np.mean([r for r, _ in by_cat[c]]) for c in cats])
    xerr = np.array([np.std([r for r, _ in by_cat[c]], ddof=1) /
                     math.sqrt(len(by_cat[c])) for c in cats])
    y = np.array([np.mean([a for _, a in by_cat[c]]) for c in cats])
    yerr = np.array([np.std([a for _, a in by_cat[c]], ddof=1) /
                     math.sqrt(len(by_cat[c])) for c in cats])
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color="#2c3e50",
                ecolor="#888888", elinewidth=1, capsize=3, markersize=8)
    for cat, xv, yv in zip(cats, x, y):
        short = cat.replace("/", "/\n").replace("decision-making", "decision-mk")
        ax.annotate(short, (xv, yv), xytext=(8, 4),
                    textcoords="offset points", fontsize=8)
    pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
    slope, intercept, *_ = scipy_stats.linregress(x, y)
    xs = np.linspace(min(x) - 0.2, max(x) + 0.2, 50)
    ax.plot(xs, slope * xs + intercept, color="#c0392b", linewidth=1,
            linestyle="--", label=f"OLS slope = {slope:+.3f}")
    ax.set_xlabel("Mean real-world uplift rating (per category)")
    ax.set_ylabel("Mean compliance-prediction AUC (per category)")
    ax.set_title(f"Per-category uplift vs compliance AUC  (n={len(cats)})\n"
                 f"Pearson r = {pearson_r:+.2f} (p={pearson_p:.3f})")
    ax.legend(fontsize=8)
    return pearson_r, pearson_p


def main():
    scores = load_scores()
    auc = load_auc()
    print(f"loaded {len(scores)} ratings, {len(auc)} per-behavior AUCs")
    print(f"intersection: {len(set(scores) & set(auc))}")
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    pr, pp, sr, sp = per_behavior_panel(ax1, scores, auc)
    fig1.tight_layout()
    fig1.savefig(ROOT / "uplift_vs_auc_per_behavior.png", dpi=150,
                 bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    cr, cp = per_category_panel(ax2, scores, auc)
    fig2.tight_layout()
    fig2.savefig(ROOT / "uplift_vs_auc_per_category.png", dpi=150,
                 bbox_inches="tight")

    summary = {
        "n_behaviors": len(set(scores) & set(auc)),
        "per_behavior": {
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
        },
        "per_category": {
            "pearson_r": cr, "pearson_p": cp,
        },
    }
    (ROOT / "stage_a_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
