"""Scatter: Stage B harm rating (per-record) vs Stage A harm prior
(per-behavior). One point per Stage B record (n=600), colored by win/loss.

Both axes are Likert 1-5; jittered to avoid pileup.
"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
FIG = f"{ROOT}/figures"


def main():
    # Stage A: behavior -> rating
    stage_a = {}
    for line in open(f"{ROOT}/working/uplift/stage_a_scores.jsonl"):
        r = json.loads(line)
        if r.get("parse_ok"):
            stage_a[r["behavior"]] = r["rating"]

    # Stage B records: pair with Stage A rating
    records = []
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        b = r["behavior"]
        if b not in stage_a:
            continue
        records.append({
            "behavior": b, "category": r["category"],
            "unsafe": r["unsafe"],
            "stage_a": stage_a[b], "stage_b": r["rating"],
        })
    n_w = sum(1 for r in records if r["unsafe"])
    n_l = sum(1 for r in records if not r["unsafe"])
    print(f"Records: {len(records)}  (wins={n_w}, losses={n_l})")

    # correlations
    xs = np.array([r["stage_a"] for r in records])
    ys = np.array([r["stage_b"] for r in records])
    rho_all, p_all = spearmanr(xs, ys)

    wins = [r for r in records if r["unsafe"]]
    xs_w = np.array([r["stage_a"] for r in wins])
    ys_w = np.array([r["stage_b"] for r in wins])
    rho_w, p_w = spearmanr(xs_w, ys_w)

    losses = [r for r in records if not r["unsafe"]]
    xs_l = np.array([r["stage_a"] for r in losses])
    ys_l = np.array([r["stage_b"] for r in losses])
    rho_l, p_l = spearmanr(xs_l, ys_l)

    # plot
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    for grp, color, label in [
        (losses, "#7f7f7f", f"losses (n={n_l})"),
        (wins, "#d62728", f"wins (n={n_w})"),
    ]:
        gx = np.array([r["stage_a"] for r in grp])
        gy = np.array([r["stage_b"] for r in grp])
        gx_j = gx + rng.uniform(-0.18, 0.18, size=len(gx))
        gy_j = gy + rng.uniform(-0.18, 0.18, size=len(gy))
        ax.scatter(gx_j, gy_j, s=22, color=color, edgecolor="black",
                   linewidth=0.3, alpha=0.55, label=label)

    # group means per Stage A level
    for grp, color in [(losses, "#7f7f7f"), (wins, "#d62728")]:
        sa_to_sb = defaultdict(list)
        for r in grp:
            sa_to_sb[r["stage_a"]].append(r["stage_b"])
        sa_levels = sorted(sa_to_sb.keys())
        means = [np.mean(sa_to_sb[s]) for s in sa_levels]
        ax.plot(sa_levels, means, marker="X", markersize=14,
                markerfacecolor=color, markeredgecolor="black",
                markeredgewidth=1.0, linestyle="-", color=color,
                linewidth=1.4, zorder=4, alpha=0.95)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.plot([1, 5], [1, 5], color="gray", linestyle=":", linewidth=0.8,
            alpha=0.6)

    stats_text = (f"All 600:    Spearman ρ = {rho_all:+.2f}, p = {p_all:.2e}\n"
                  f"Wins  n={n_w}:  Spearman ρ = {rho_w:+.2f}, p = {p_w:.2e}\n"
                  f"Losses n={n_l}: Spearman ρ = {rho_l:+.2f}, p = {p_l:.2e}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.93))

    ax.set_xlabel("Stage A harm prior (per-behavior, hypothetical max uplift, "
                  "Likert 1-5)")
    ax.set_ylabel("Stage B harm (per-response, actual uplift, Likert 1-5)")
    ax.set_title("Stage B (per-response) vs Stage A (per-behavior) harm\n"
                 "X = group mean per Stage A level",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.93)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/stage_b_vs_stage_a_scatter.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    print()
    print(f"Spearman correlations:")
    print(f"  all 600:    rho = {rho_all:+.3f}, p = {p_all:.4f}")
    print(f"  wins only:  rho = {rho_w:+.3f}, p = {p_w:.4f}")
    print(f"  losses:     rho = {rho_l:+.3f}, p = {p_l:.4f}")

    # tabulate Stage B mean by Stage A level, both arms
    print(f"\nMean Stage B per Stage A level:")
    print(f"{'Stage A':<10}{'n records':<12}{'n wins':<10}{'mean SB (wins)':<18}"
          f"{'n losses':<10}{'mean SB (loss)':<15}")
    for sa in sorted(set(xs)):
        rs = [r for r in records if r["stage_a"] == sa]
        ws = [r["stage_b"] for r in rs if r["unsafe"]]
        ls = [r["stage_b"] for r in rs if not r["unsafe"]]
        mw = np.mean(ws) if ws else float("nan")
        ml = np.mean(ls) if ls else float("nan")
        print(f"  {sa:<8}  {len(rs):<10}  {len(ws):<8}  {mw:<16.2f}  "
              f"{len(ls):<8}  {ml:<13.2f}")


if __name__ == "__main__":
    main()
