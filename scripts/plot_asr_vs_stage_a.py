"""Scatter: per-behavior ASR (from the 9,400-conv pool) vs per-behavior
Stage A harm prior (Qwen-72B's hypothetical-max-uplift Likert, n=100 goals).

One point per JBB behavior. Colored by category.
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

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
FIG = f"{ROOT}/figures"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    # Stage A
    stage_a = {}
    for line in open(f"{ROOT}/working/uplift/stage_a_scores.jsonl"):
        r = json.loads(line)
        if r.get("parse_ok"):
            stage_a[r["behavior"]] = (r["rating"], r["category"])
    print(f"Stage A behaviors: {len(stage_a)}")

    # per-behavior wins/trials from the pooled corpus
    wins = defaultdict(int); trials = defaultdict(int)
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            # need behavior; not in pooled_hs schema. Pull it from goals via category+goal join.
            # Actually pooled_hs stores 'goals' (goal text). Match to stage_a via Stage A's goal field.
            pass

    # pooled_hs stores 'goals' string, not behavior. Match via goal text from Stage A.
    goal_to_behavior = {}
    for line in open(f"{ROOT}/working/uplift/goals.json").read() and open(
            f"{ROOT}/working/uplift/goals.json"):
        pass
    goals = json.load(open(f"{ROOT}/working/uplift/goals.json"))
    for g in goals:
        goal_to_behavior[g["goal"]] = g["behavior"]

    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            for lab, goal in zip(data["labels"].tolist(), data["goals"]):
                b = goal_to_behavior.get(goal)
                if b is None:
                    continue
                trials[b] += 1
                wins[b] += int(bool(lab))

    print(f"Behaviors with pool data: {len(trials)}")

    # build merged table
    rows = []
    for b in stage_a:
        if b not in trials or trials[b] == 0:
            continue
        rating, cat = stage_a[b]
        asr = wins[b] / trials[b]
        lo, hi = wilson(wins[b], trials[b])
        rows.append({"behavior": b, "category": cat,
                     "stage_a": rating, "asr": asr,
                     "asr_lo": lo, "asr_hi": hi,
                     "n": trials[b], "n_wins": wins[b]})
    print(f"Merged rows: {len(rows)}")

    # correlation
    xs = np.array([r["stage_a"] for r in rows])
    ys = np.array([r["asr"] for r in rows])
    r_p, p_p = pearsonr(xs, ys)
    r_s, p_s = spearmanr(xs, ys)

    # plot
    cats = sorted(set(r["category"] for r in rows))
    cmap = plt.get_cmap("tab10")
    color_of = {c: cmap(i % 10) for i, c in enumerate(cats)}

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    rng = np.random.default_rng(0)
    for r in rows:
        c = color_of[r["category"]]
        x = r["stage_a"] + rng.uniform(-0.18, 0.18)
        y = 100 * r["asr"]
        ax.errorbar(x, y,
                    yerr=[[100 * (r["asr"] - r["asr_lo"])],
                          [100 * (r["asr_hi"] - r["asr"])]],
                    fmt="o", color=c, ecolor=c, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.4,
                    elinewidth=0.7, capsize=2, alpha=0.85)

    # linear fit on raw (unjittered) data
    m, b = np.polyfit(xs, 100 * ys, 1)
    xs_grid = np.linspace(0.7, 5.3, 100)
    ax.plot(xs_grid, m * xs_grid + b, color="gray", linestyle="--",
            linewidth=0.9, alpha=0.7, zorder=0)

    ax.text(0.02, 0.98,
            f"Pearson r = {r_p:+.2f}, p = {p_p:.3f}\n"
            f"Spearman ρ = {r_s:+.2f}, p = {p_s:.3f}\n"
            f"n = {len(rows)} behaviors",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.93))

    # category legend outside
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=color_of[c], markeredgecolor="black",
                          markersize=7, label=c)
               for c in cats]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8.5, frameon=False, title="category",
              title_fontsize=9)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-2, max(100 * r["asr_hi"] for r in rows) + 4)
    ax.set_xlabel("Stage A harm prior (Qwen-72B hypothetical max uplift, Likert 1-5)")
    ax.set_ylabel("per-behavior ASR (%, Wilson 95% CI, full 9,400-conv pool)")
    ax.set_title("ASR vs Stage A harm prior, by JBB behavior",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/asr_vs_stage_a_scatter.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    print()
    print(f"Pearson r = {r_p:+.3f}, p = {p_p:.4f}")
    print(f"Spearman ρ = {r_s:+.3f}, p = {p_s:.4f}")
    print()
    print(f"Stage A rating distribution:")
    from collections import Counter
    rc = Counter(r["stage_a"] for r in rows)
    for rating in sorted(rc):
        bs = [r for r in rows if r["stage_a"] == rating]
        mean_asr = np.mean([r["asr"] for r in bs])
        print(f"  Stage A = {rating}: n={rc[rating]:>3} behaviors,  "
              f"mean ASR = {100*mean_asr:.1f}%")


if __name__ == "__main__":
    main()
