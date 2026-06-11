"""Cumulative ASR vs turn number, per JailbreakBench category, on the
11-run 9,400-conv probe pool.

Cumulative ASR at turn t = fraction of conversations whose first breach
turn is in {0, 1, ..., t}.  Reads `turns_of_breach` and `categories` from
each `pooled_hs/<run>/round_*.pt`.  Writes
`figures/cum_asr_by_category.{pdf,png}`.
"""
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = "/Users/julianquick/portfolio_copy/turnstile/experiments/pooled_hs"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"
NUM_TURNS = 5


def main():
    # per category: list of (turn_of_breach or None) over all convs
    by_cat = defaultdict(list)
    for run_dir in sorted(glob.glob(f"{ROOT}/*/")):
        for path in sorted(glob.glob(f"{run_dir}/round_*.pt")):
            blob = torch.load(path, weights_only=False)
            cats = blob["categories"]
            tobs = blob["turns_of_breach"]
            labels = blob["labels"]
            for cat, tob, lab in zip(cats, tobs, labels):
                if not bool(lab):
                    tob_eff = None
                else:
                    tob_eff = int(tob) if tob is not None else None
                by_cat[cat].append(tob_eff)

    # Order categories by their final ASR (same order as the per-category plot)
    cat_final = {c: sum(1 for t in toks if t is not None) / len(toks)
                 for c, toks in by_cat.items()}
    cats_sorted = sorted(by_cat.keys(), key=lambda c: -cat_final[c])

    # Compute cumulative ASR per turn for each category
    turns = list(range(NUM_TURNS))
    cum = {}
    for c in cats_sorted:
        toks = by_cat[c]
        n = len(toks)
        cum[c] = []
        for t in turns:
            wins_le_t = sum(1 for x in toks if x is not None and x <= t)
            cum[c].append(100 * wins_le_t / n)

    fig, ax = plt.subplots(figsize=(19.55, 5.0))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats_sorted):
        ax.plot(turns, cum[c], marker="o", markersize=5, linewidth=1.7,
                color=cmap(i % 10), label=f"{c} (n={len(by_cat[c])})")

    # Overall mean across all convs
    all_toks = [t for toks in by_cat.values() for t in toks]
    n_total = len(all_toks)
    overall = [100 * sum(1 for x in all_toks if x is not None and x <= t) / n_total
               for t in turns]
    ax.plot(turns, overall, marker="s", markersize=6, linewidth=2.2,
            color="black", linestyle="--", label=f"pooled (n={n_total:,})")

    ax.set_xlabel("turn $t$ (cumulative through this turn)")
    ax.set_ylabel("cumulative ASR (%): fraction with first breach $\\leq t$")
    ax.set_title("Cumulative ASR vs turn, by JailbreakBench category (11-run probe pool)",
                 fontsize=11)
    ax.set_xticks(turns)
    ax.set_xticklabels([f"T{t}" for t in turns])
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_xlim(-0.15, NUM_TURNS - 0.85)
    ax.set_ylim(0, None)

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, frameon=False, title="category", title_fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/cum_asr_by_category.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)

    # also print the table
    print()
    header = f"{'category':<30} " + " ".join(f"  T{t}  " for t in turns) + "   n"
    print(header)
    for c in cats_sorted:
        row = f"{c:<30} " + " ".join(f"{v:>5.1f}%" for v in cum[c]) + f"  {len(by_cat[c]):>5}"
        print(row)
    print("-" * len(header))
    print(f"{'POOLED':<30} " + " ".join(f"{v:>5.1f}%" for v in overall) +
          f"  {n_total:>5}")


if __name__ == "__main__":
    main()
