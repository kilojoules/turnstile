"""Stage B analysis: per-behavior real-world-uplift (judge on actual responses)
vs per-behavior compliance-prediction AUC.

Inputs:
  stage_b_scores.jsonl                                                  (600 ratings, this dir)
  ../experiments/outcome_probe_v1/per_behavior_L16_T1.json              (existing AUCs)

Plots:
  uplift_b_vs_auc_per_behavior.png   per-behavior uplift mean (wins) vs AUC
  uplift_b_per_category.png          win-vs-loss uplift by category
  win_loss_distribution.png          rating distributions for wins vs losses
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
AUC_FILE = ROOT.parent.parent / "experiments" / "outcome_probe_v1" / "per_behavior_L16_T1.json"


def load():
    recs = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    auc = json.loads(AUC_FILE.read_text())
    return recs, auc


def per_behavior_panel(ax, recs, auc):
    """Win-conditional mean uplift vs compliance AUC, per behavior."""
    by_b_wins = defaultdict(list)
    cat_of = {}
    for r in recs:
        if r["unsafe"] and r["rating"] is not None:
            by_b_wins[r["behavior"]].append(r["rating"])
            cat_of[r["behavior"]] = r["category"]
    rows = []
    for b, ratings in by_b_wins.items():
        if b not in auc or len(ratings) < 2:
            continue
        rows.append({
            "behavior": b,
            "category": cat_of[b],
            "uplift_mean": statistics.mean(ratings),
            "uplift_se": statistics.stdev(ratings) / math.sqrt(len(ratings)) if len(ratings) > 1 else 0,
            "auc": auc[b]["auc"],
            "auc_std": auc[b]["std"],
            "n_wins": len(ratings),
        })
    rows.sort(key=lambda r: r["category"])
    cats = sorted({r["category"] for r in rows})
    cmap = plt.get_cmap("tab10")
    color_map = {c: cmap(i) for i, c in enumerate(cats)}
    x = np.array([r["uplift_mean"] for r in rows])
    xerr = np.array([r["uplift_se"] for r in rows])
    y = np.array([r["auc"] for r in rows])
    yerr = np.array([r["auc_std"] for r in rows])
    colors = [color_map[r["category"]] for r in rows]

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="#bbbbbb",
                elinewidth=0.6, capsize=2, zorder=2)
    ax.scatter(x, y, c=colors, s=40, edgecolor="black", linewidth=0.4, zorder=3)
    pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
    slope, intercept, *_ = scipy_stats.linregress(x, y)
    xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 50)
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.2,
            linestyle="--", label=f"OLS slope = {slope:+.3f}/Likert pt")
    ax.axhline(0.5, color="gray", linewidth=0.5)
    ax.set_xlabel("Per-behavior mean uplift among wins (Qwen2.5-72B; Likert 1-5)")
    ax.set_ylabel("Compliance-prediction AUC (L16, T1, ±5-fold CV std)")
    ax.set_title(f"Per-behavior actual-response uplift vs compliance AUC  (n={len(rows)})\n"
                 f"Pearson r = {pearson_r:+.2f} (p={pearson_p:.3f}),  "
                 f"Spearman ρ = {spearman_r:+.2f} (p={spearman_p:.3f})")
    handles = [plt.Line2D([], [], marker="o", color=color_map[c], linestyle="",
                          markersize=6, markeredgecolor="black",
                          markeredgewidth=0.4, label=c) for c in cats]
    ax.legend(handles=handles + [ax.lines[-1]], fontsize=7,
              loc="upper right", ncol=2, framealpha=0.9)
    return {
        "n_behaviors": len(rows),
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "slope": slope,
    }


def per_category_panel(ax, recs):
    """Mean uplift among wins vs among losses, per category."""
    by_cat = defaultdict(lambda: {"wins": [], "losses": []})
    for r in recs:
        if r["rating"] is None:
            continue
        bucket = "wins" if r["unsafe"] else "losses"
        by_cat[r["category"]][bucket].append(r["rating"])
    cats = sorted(by_cat.keys())
    x = np.arange(len(cats))
    width = 0.38
    win_means = [statistics.mean(by_cat[c]["wins"]) if by_cat[c]["wins"] else 0 for c in cats]
    win_se = [statistics.stdev(by_cat[c]["wins"]) / math.sqrt(len(by_cat[c]["wins"]))
              if len(by_cat[c]["wins"]) > 1 else 0 for c in cats]
    loss_means = [statistics.mean(by_cat[c]["losses"]) if by_cat[c]["losses"] else 0 for c in cats]
    loss_se = [statistics.stdev(by_cat[c]["losses"]) / math.sqrt(len(by_cat[c]["losses"]))
               if len(by_cat[c]["losses"]) > 1 else 0 for c in cats]
    ax.bar(x - width/2, win_means, width, yerr=win_se, capsize=2,
           color="#c0392b", edgecolor="black", linewidth=0.4, label="wins")
    ax.bar(x + width/2, loss_means, width, yerr=loss_se, capsize=2,
           color="#7f8c8d", edgecolor="black", linewidth=0.4, label="losses")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("/", "/\n") for c in cats], fontsize=7, rotation=0)
    ax.set_ylabel("Mean uplift rating (1-5)")
    ax.set_title("Mean response uplift by category, wins vs losses\n"
                 "(error bars: SE)")
    ax.legend(fontsize=8)
    ax.set_ylim(0.8, 5.2)


def win_loss_dist_panel(ax, recs):
    wins = [r["rating"] for r in recs if r["unsafe"] and r["rating"] is not None]
    losses = [r["rating"] for r in recs if not r["unsafe"] and r["rating"] is not None]
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ax.hist([wins, losses], bins=bins, label=["wins", "losses"],
            color=["#c0392b", "#7f8c8d"], edgecolor="black")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Uplift rating")
    ax.set_ylabel("count")
    win_mean = statistics.mean(wins)
    loss_mean = statistics.mean(losses)
    u, p = scipy_stats.mannwhitneyu(wins, losses, alternative="greater")
    ax.set_title(f"Rating distribution by judge label\n"
                 f"wins mean={win_mean:.2f} (n={len(wins)})  "
                 f"losses mean={loss_mean:.2f} (n={len(losses)})  "
                 f"MWU p={p:.1e}")
    ax.legend()


def main():
    recs, auc = load()
    print(f"loaded {len(recs)} ratings, {len(auc)} per-behavior AUCs")

    fig1, ax1 = plt.subplots(figsize=(10, 7))
    summary = per_behavior_panel(ax1, recs, auc)
    fig1.tight_layout()
    fig1.savefig(ROOT / "uplift_b_vs_auc_per_behavior.png", dpi=150, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    per_category_panel(ax2, recs)
    fig2.tight_layout()
    fig2.savefig(ROOT / "uplift_b_per_category.png", dpi=150, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    win_loss_dist_panel(ax3, recs)
    fig3.tight_layout()
    fig3.savefig(ROOT / "win_loss_distribution.png", dpi=150, bbox_inches="tight")

    (ROOT / "stage_b_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
