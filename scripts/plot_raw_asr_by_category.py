"""Two views of raw per-turn ASR by JailbreakBench category.

(A) per_turn_judge_v2_full corpus: exact per-turn unsafe rates, but
    conditioned on the conversation eventually breaching (200 wins-side
    records).

(B) 9,400-conv probe pool: raw per-turn ASR is bounded between the
    marginal (first-breach==t) and cumulative (first-breach<=t) since
    the upstream judge short-circuits at first breach.  We plot both
    bounds; the truth lives in the corridor between them.

Writes `figures/raw_asr_by_category_wins.pdf` and
`figures/raw_asr_by_category_pool_corridor.pdf`.
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

POOL_ROOT = "/Users/julianquick/portfolio_copy/turnstile/experiments/pooled_hs"
WINS_PATH = "/Users/julianquick/portfolio_copy/turnstile/experiments/per_turn_judge_v2_full/results.jsonl"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"
NUM_TURNS = 5


def fig_a_wins():
    recs = [json.loads(l) for l in open(WINS_PATH) if json.loads(l)["arm"] == "original"]
    by_cat_turn = defaultdict(lambda: {"n": 0, "unsafe": 0})
    for r in recs:
        key = (r["category"], r["turn"])
        by_cat_turn[key]["n"] += 1
        if r["verdict"].get("unsafe"):
            by_cat_turn[key]["unsafe"] += 1

    cats = sorted(set(c for c, _ in by_cat_turn.keys()))
    # order by final-turn rate
    cats_sorted = sorted(cats, key=lambda c: -by_cat_turn[(c, 4)]["unsafe"]
                                              / max(1, by_cat_turn[(c, 4)]["n"]))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    cmap = plt.get_cmap("tab10")
    turns = list(range(NUM_TURNS))
    pooled = []
    for t in turns:
        n = sum(by_cat_turn[(c, t)]["n"] for c in cats)
        u = sum(by_cat_turn[(c, t)]["unsafe"] for c in cats)
        pooled.append(100 * u / n)

    for i, c in enumerate(cats_sorted):
        pcts = [100 * by_cat_turn[(c, t)]["unsafe"]
                / max(1, by_cat_turn[(c, t)]["n"]) for t in turns]
        n = by_cat_turn[(c, 0)]["n"]
        ax.plot(turns, pcts, marker="o", markersize=5, linewidth=1.6,
                color=cmap(i % 10), alpha=0.85, label=f"{c} (n={n})")
    ax.plot(turns, pooled, marker="s", markersize=6, linewidth=2.4,
            color="black", linestyle="--",
            label=f"pooled (n=200, all wins-side)")

    ax.set_xticks(turns)
    ax.set_xticklabels([f"T{t}" for t in turns])
    ax.set_xlabel("turn $t$")
    ax.set_ylabel("raw ASR (%): turn-$t$ response judged unsafe")
    ax.set_title("Raw per-turn ASR by category, conditioned on conversation eventually breaching\n"
                 "(per_turn_judge_v2_full, 200 wins-side records, original arm)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_xlim(-0.15, NUM_TURNS - 0.85)
    ax.set_ylim(0, 105)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, frameon=False, title="category", title_fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/raw_asr_by_category_wins.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


def fig_b_pool_corridor():
    by_cat = defaultdict(list)
    for run_dir in sorted(glob.glob(f"{POOL_ROOT}/*/")):
        for path in sorted(glob.glob(f"{run_dir}/round_*.pt")):
            blob = torch.load(path, weights_only=False)
            for cat, tob, lab in zip(blob["categories"], blob["turns_of_breach"],
                                     blob["labels"]):
                if not bool(lab):
                    by_cat[cat].append(None)
                else:
                    by_cat[cat].append(int(tob) if tob is not None else None)

    cat_final = {c: sum(1 for t in toks if t is not None) / len(toks)
                 for c, toks in by_cat.items()}
    cats_sorted = sorted(by_cat.keys(), key=lambda c: -cat_final[c])
    turns = list(range(NUM_TURNS))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats_sorted):
        toks = by_cat[c]
        n = len(toks)
        marg = [100 * sum(1 for x in toks if x == t) / n for t in turns]
        cumu = [100 * sum(1 for x in toks if x is not None and x <= t) / n
                for t in turns]
        color = cmap(i % 10)
        ax.fill_between(turns, marg, cumu, color=color, alpha=0.18)
        ax.plot(turns, marg, color=color, linewidth=1.2, linestyle=":",
                alpha=0.9)
        ax.plot(turns, cumu, color=color, linewidth=1.7, marker="o",
                markersize=4.5, label=f"{c} (n={n})", alpha=0.95)

    # custom dotted/solid legend caption
    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0], [0], color="gray", linewidth=1.7, marker="o", markersize=4.5,
               label="upper bound (cumulative)"),
        Line2D([0], [0], color="gray", linewidth=1.2, linestyle=":",
               label="lower bound (marginal)"),
    ]
    leg1 = ax.legend(handles=proxy, loc="upper left", fontsize=8.5,
                     frameon=True, framealpha=0.92)
    ax.add_artist(leg1)

    ax.set_xticks(turns)
    ax.set_xticklabels([f"T{t}" for t in turns])
    ax.set_xlabel("turn $t$")
    ax.set_ylabel("raw ASR (%) corridor")
    ax.set_title("Raw per-turn ASR corridor on full 9,400-conv pool\n"
                 "(marginal $\\leq$ raw $\\leq$ cumulative; upstream judge short-circuits at first breach)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_xlim(-0.15, NUM_TURNS - 0.85)
    ax.set_ylim(0, None)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, frameon=False, title="category", title_fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/raw_asr_by_category_pool_corridor.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_a_wins()
    fig_b_pool_corridor()
