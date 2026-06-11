"""Two coverage heatmaps for the Stage B subsample (n=600):
- rows: 10 JBB categories
- cols: 10 behaviors within each category, indexed 0-9 (alphabetical)
- cell: count of Stage B records for that (category, behavior)
Wins heatmap and Losses heatmap saved separately.
"""
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
FIG = f"{ROOT}/figures"


def main():
    goals = json.load(open(f"{ROOT}/working/uplift/goals.json"))
    # canonical 10 behaviors per category, alphabetical
    cat_behaviors = defaultdict(list)
    for g in goals:
        cat_behaviors[g["category"]].append(g["behavior"])
    for c in cat_behaviors:
        cat_behaviors[c] = sorted(cat_behaviors[c])
    categories = sorted(cat_behaviors.keys())

    # Stage B records — use scores file, which has unsafe + behavior + category
    sb = []
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        sb.append(r)
    # Note: parse_ok filter mirrors what the analysis script used; the
    # underlying selection had 600 records.
    print(f"Stage B parse_ok records: {len(sb)}")

    counts_w = np.zeros((10, 10), dtype=int)
    counts_l = np.zeros((10, 10), dtype=int)
    for r in sb:
        cat = r["category"]; beh = r["behavior"]
        if cat not in cat_behaviors:
            continue
        if beh not in cat_behaviors[cat]:
            continue
        ci = categories.index(cat)
        bi = cat_behaviors[cat].index(beh)
        if r["unsafe"]:
            counts_w[ci, bi] += 1
        else:
            counts_l[ci, bi] += 1

    print(f"Wins sum: {counts_w.sum()}   Losses sum: {counts_l.sum()}")
    print(f"Total cells: {counts_w.size + counts_l.size // 2} (100 (cat, beh) pairs)")
    print(f"Row sums (cat totals, wins+losses):")
    for ci, c in enumerate(categories):
        row_total = counts_w[ci].sum() + counts_l[ci].sum()
        print(f"  {c:<28}  wins={counts_w[ci].sum()}  losses={counts_l[ci].sum()}  "
              f"total={row_total}")

    # ----- summary stats -----
    zero_w = int(((counts_w == 0)).sum())
    lt3_w = int(((counts_w > 0) & (counts_w < 3)).sum())
    ge3_w = int((counts_w >= 3).sum())
    gt3_l = int((counts_l > 3).sum())

    cat_short_w = sorted(
        [(c, (counts_w[categories.index(c)] < 3).sum())
         for c in categories],
        key=lambda x: -x[1],
    )

    print(f"\n==== Win coverage ====")
    print(f"  Cells with 0 wins: {zero_w} / 100")
    print(f"  Cells with 1-2 wins (underfilled): {lt3_w} / 100")
    print(f"  Cells with >=3 wins (target met): {ge3_w} / 100")
    print(f"\n  Categories with most win shortfalls (cells where wins < 3):")
    for c, n in cat_short_w:
        if n > 0:
            print(f"    {c:<28}  {n}/10 behaviors short")
    print(f"\n==== Loss coverage ====")
    print(f"  Cells with > 3 losses (backfill present): {gt3_l} / 100")

    # ----- plot -----
    def plot_heatmap(counts, vmax, title, fname, underfill_thresh=None,
                     overfill_thresh=None):
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        im = ax.imshow(counts, cmap="viridis", vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"b{i}" for i in range(10)], fontsize=9)
        ax.set_yticks(range(10))
        ax.set_yticklabels(categories, fontsize=9)
        # cell annotations
        for ci in range(10):
            for bi in range(10):
                v = counts[ci, bi]
                # font color contrasted with cell
                color = "white" if v < vmax * 0.6 else "black"
                bold = False
                if underfill_thresh is not None and v < underfill_thresh:
                    bold = True
                    color = "white" if v == 0 else "yellow"
                if overfill_thresh is not None and v > overfill_thresh:
                    bold = True
                    color = "red"
                ax.text(bi, ci, str(int(v)), ha="center", va="center",
                        fontsize=9, color=color,
                        fontweight=("bold" if bold else "normal"))
                # red border for underfilled cells
                if underfill_thresh is not None and v < underfill_thresh:
                    rect = patches.Rectangle((bi - 0.48, ci - 0.48), 0.96, 0.96,
                                             linewidth=1.5, edgecolor="red",
                                             facecolor="none")
                    ax.add_patch(rect)
                if overfill_thresh is not None and v > overfill_thresh:
                    rect = patches.Rectangle((bi - 0.48, ci - 0.48), 0.96, 0.96,
                                             linewidth=1.5, edgecolor="red",
                                             facecolor="none")
                    ax.add_patch(rect)
        plt.colorbar(im, ax=ax, label="count")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("behavior index within category (alphabetical)")
        ax.set_ylabel("category")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out_p = f"{FIG}/{fname}.{ext}"
            fig.savefig(out_p, bbox_inches="tight",
                        dpi=150 if ext == "png" else None)
            print(f"  wrote {out_p}")
        plt.close(fig)

    plot_heatmap(counts_w, vmax=3,
                 title=f"Stage B WIN coverage  (total={counts_w.sum()}, "
                       f"target=3/cell; red border = underfilled)",
                 fname="stage_b_coverage_wins",
                 underfill_thresh=3)
    plot_heatmap(counts_l, vmax=6,
                 title=f"Stage B LOSS coverage  (total={counts_l.sum()}, "
                       f"design=3/cell, backfill up to 6; red border = >3)",
                 fname="stage_b_coverage_losses",
                 overfill_thresh=3)

    # ----- behavior-index legend table -----
    print("\n==== Behavior-index legend (col b0..b9 per category) ====")
    for c in categories:
        print(f"\n{c}:")
        for i, b in enumerate(cat_behaviors[c]):
            print(f"  b{i}: {b}")


if __name__ == "__main__":
    main()
