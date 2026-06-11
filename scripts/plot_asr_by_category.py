"""Aggregate ASR per JailbreakBench category across the 11-run probe corpus
and plot it as a horizontal bar chart with Wilson 95% CIs.

Reads `experiments/pooled_hs/<run>/round_<i>.pt` for every (run, round) pair
present, pulls the `categories` + `labels` arrays, sums wins/trials per
category, and writes `figures/asr_by_category.{pdf,png}`.
"""
import glob
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = "/Users/julianquick/portfolio_copy/turnstile/experiments/pooled_hs"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    wins = defaultdict(int)
    trials = defaultdict(int)
    runs_seen = defaultdict(set)

    for run_dir in sorted(glob.glob(f"{ROOT}/*/")):
        run = os.path.basename(run_dir.rstrip("/"))
        for path in sorted(glob.glob(f"{run_dir}/round_*.pt")):
            blob = torch.load(path, weights_only=False)
            cats = blob["categories"]
            labels = blob["labels"]
            for cat, lab in zip(cats, labels):
                trials[cat] += 1
                wins[cat] += int(bool(lab))
                runs_seen[cat].add(run)

    cats = sorted(trials.keys(), key=lambda c: wins[c] / trials[c])
    asrs = [wins[c] / trials[c] for c in cats]
    cis = [wilson_ci(wins[c], trials[c]) for c in cats]
    los = [a - lo for a, (lo, _) in zip(asrs, cis)]
    his = [hi - a for a, (_, hi) in zip(asrs, cis)]

    total_n = sum(trials.values())
    overall = sum(wins.values()) / total_n

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = list(range(len(cats)))
    pcts = [100 * a for a in asrs]
    err = [[100 * x for x in los], [100 * x for x in his]]
    ax.barh(y, pcts, xerr=err, color="#3b6fb0", edgecolor="black",
            linewidth=0.5, ecolor="black", capsize=2.5, alpha=0.92)
    ax.axvline(100 * overall, color="black", linestyle="--", linewidth=0.7,
               label=f"pooled mean ({100*overall:.1f}%, n={total_n:,})")

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel("ASR (%, pivot-only judge), Wilson 95% CI")
    ax.set_title("ASR by JailbreakBench category, pooled across 11-run probe corpus",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    ax.set_xlim(0, max(100 * (a + h) for a, h in zip(asrs, his)) + 4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    for yi, (a, n) in enumerate(zip(pcts, [trials[c] for c in cats])):
        ax.text(a + 0.4, yi, f"{a:.1f}%  (n={n:,})", va="center", fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/asr_by_category.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)

    print(f"\ntotal conversations: {total_n:,}")
    print(f"overall ASR: {100*overall:.2f}%\n")
    for c in reversed(cats):
        lo, hi = wilson_ci(wins[c], trials[c])
        print(f"  {c:<30}  ASR {100*wins[c]/trials[c]:>5.1f}%  "
              f"[{100*lo:>4.1f}, {100*hi:>4.1f}]  n={trials[c]:>5}")


if __name__ == "__main__":
    main()
