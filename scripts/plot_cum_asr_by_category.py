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

    # Overall mean across all convs
    all_toks = [t for toks in by_cat.values() for t in toks]
    n_total = len(all_toks)
    overall = [100 * sum(1 for x in all_toks if x is not None and x <= t) / n_total
               for t in turns]

    # ---- professional styling ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 13, "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#4d4d4d", "axes.linewidth": 0.9,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    # muted, distinct, publication-friendly palette (seaborn-deep style), ordered by rank
    PALETTE = ["#3b6ea5", "#e07b39", "#4f9d69", "#c0455a", "#7d6bb0",
               "#8f6f57", "#cf7fb3", "#c9a441", "#4bacc6", "#7f7f7f", "#5a4a99"]
    GRID = "#e6e6e6"

    fig, ax = plt.subplots(figsize=(19.0, 4.75))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.xaxis.grid(False)

    handles = []
    for i, c in enumerate(cats_sorted):
        (h,) = ax.plot(turns, cum[c], "-o", color=PALETTE[i % len(PALETTE)],
                       lw=2.0, ms=5, mec="white", mew=0.7, solid_capstyle="round",
                       label=f"{c}   {cum[c][-1]:.0f}%")
        handles.append(h)
    (hp,) = ax.plot(turns, overall, "--s", color="#1a1a1a", lw=2.8, ms=6.5,
                    mec="white", mew=0.8, zorder=10, label=f"pooled   {overall[-1]:.0f}%")

    ax.set_xlabel("conversation turn", labelpad=8)
    ax.set_ylabel("cumulative attack success (%)", labelpad=8)
    ax.set_xticks(turns); ax.set_xticklabels([f"T{t}" for t in turns])
    ax.set_xlim(-0.12, NUM_TURNS - 0.9)
    ax.set_ylim(0, max(cum[cats_sorted[0]]) * 1.08)
    ax.tick_params(length=0)
    ax.margins(x=0.01)

    # editorial title + subtitle (left-aligned)
    ax.set_title("Attack success accrues across turns, by JailbreakBench category",
                 fontweight="semibold", loc="left", pad=20)
    ax.text(0.0, 1.045, "fraction of conversations first breached by turn $t$  ·  "
            "11-run self-play pool  ·  n = 9,400 conversations",
            transform=ax.transAxes, fontsize=10, color="#6b6b6b", ha="left")

    leg = ax.legend([hp] + handles, [hp.get_label()] + [h.get_label() for h in handles],
                    loc="center left", bbox_to_anchor=(1.015, 0.5), fontsize=9.5,
                    frameon=False, handlelength=1.6, labelspacing=0.6,
                    title="category  ·  final ASR", title_fontsize=10, alignment="left")
    leg.get_title().set_fontweight("semibold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/cum_asr_by_category.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
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
