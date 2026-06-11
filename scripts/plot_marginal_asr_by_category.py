"""Marginal (per-turn) breach rate by JBB category, on the 11-run pool.

Marginal ASR at turn t = (# convs with first_breach == t) / total.

Cumulative growth past T1 in the cum-ASR plot prompted this diagnostic:
does the per-turn breach rate actually peak at T1 in every category, or are
some categories late-firing?  And how much of the post-T1 climb is the
attacker genuinely needing more turns vs. the Guard side of the dual judge
firing late?
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
    by_cat = defaultdict(list)
    for run_dir in sorted(glob.glob(f"{ROOT}/*/")):
        for path in sorted(glob.glob(f"{run_dir}/round_*.pt")):
            blob = torch.load(path, weights_only=False)
            for cat, tob, lab in zip(blob["categories"], blob["turns_of_breach"],
                                     blob["labels"]):
                by_cat[cat].append(int(tob) if (bool(lab) and tob is not None)
                                   else None)

    # marginal: for each cat, what fraction breach at exactly turn t
    cat_final = {c: sum(1 for t in toks if t is not None) / len(toks)
                 for c, toks in by_cat.items()}
    cats_sorted = sorted(by_cat.keys(), key=lambda c: -cat_final[c])

    turns = list(range(NUM_TURNS))
    marg = {}
    for c in cats_sorted:
        toks = by_cat[c]
        n = len(toks)
        marg[c] = [100 * sum(1 for x in toks if x == t) / n for t in turns]

    # pooled
    all_toks = [t for toks in by_cat.values() for t in toks]
    n_total = len(all_toks)
    overall = [100 * sum(1 for x in all_toks if x == t) / n_total
               for t in turns]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats_sorted):
        ax.plot(turns, marg[c], marker="o", markersize=5, linewidth=1.7,
                color=cmap(i % 10), alpha=0.85,
                label=f"{c} (n={len(by_cat[c])})")
    ax.plot(turns, overall, marker="s", markersize=6, linewidth=2.4,
            color="black", linestyle="--", label=f"pooled (n={n_total:,})")

    ax.set_xlabel("turn $t$")
    ax.set_ylabel("marginal ASR (%): first breach at exactly turn $t$")
    ax.set_title("Marginal breach rate by turn, per JBB category", fontsize=11)
    ax.set_xticks(turns)
    ax.set_xticklabels([f"T{t}" for t in turns])
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_xlim(-0.15, NUM_TURNS - 0.85)
    ax.set_ylim(0, None)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, frameon=False, title="category", title_fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/marginal_asr_by_category.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)

    header = f"{'category':<30} " + " ".join(f"  T{t}  " for t in turns) + "  peak"
    print(); print(header)
    for c in cats_sorted:
        pk = max(range(NUM_TURNS), key=lambda t: marg[c][t])
        row = f"{c:<30} " + " ".join(f"{v:>5.1f}%" for v in marg[c]) + f"   T{pk}"
        print(row)
    print("-" * len(header))
    pk = max(range(NUM_TURNS), key=lambda t: overall[t])
    print(f"{'POOLED':<30} " + " ".join(f"{v:>5.1f}%" for v in overall) + f"   T{pk}")


if __name__ == "__main__":
    main()
