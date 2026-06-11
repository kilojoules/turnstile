"""Extended steering sweep plot at L16 with α ∈ {−1.0, ..., +1.0}, and a
side-by-side L16 vs L24 harm-axis comparison.

Reads the combined judged data from sweep_judged.jsonl (original L16,
α∈{-0.5, ..., +0.75}) + sweep_new_judged.jsonl (L16 ext at ±1.0, ±0.75
plus L24 micro-sweep). Sign convention: +α_c (display) = amplify
compliance; +α_h = amplify harm (no flip on harm).
"""
import json
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
FIG = f"{ROOT}/figures"
COMMON_RANGE = (-1.0, 1.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def boot_mean(vals, n_boot=500, seed=42):
    if not vals:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        means.append(float(np.mean([vals[i] for i in idx])))
    return (float(np.mean(vals)),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def aggregate(subset, axis, layer):
    """axis='comp' or 'harm'; collect cells where the OTHER axis == 0 at the
    specified layer. Returns dict keyed by display alpha (flipped for comp).
    """
    bins = defaultdict(list)
    for r in subset:
        if r.get("layer", 16) != layer:
            continue
        if axis == "comp" and abs(r['alpha_h']) < 1e-9:
            bins[round(-r['alpha_c'], 2)].append(r)  # FLIP for display
        elif axis == "harm" and abs(r['alpha_c']) < 1e-9:
            bins[round(r['alpha_h'], 2)].append(r)
    return bins


def load_all():
    rows = []
    for path in [f"{ROOT}/experiments/steering_v3/sweep_judged.jsonl",
                 f"{ROOT}/experiments/steering_v3/sweep_new_judged.jsonl"]:
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def make_main_plot(rows):
    """Top: ASR | Bottom: meanHarm; left=wins, right=losses; L16 only.
    α now spans ±1.0."""
    wins = [r for r in rows if r.get('prompt_type') == 'win']
    losses = [r for r in rows if r.get('prompt_type') == 'loss']

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex='col',
                              sharey='row')
    titles_top = ["Wins  (n=30/cell)", "Losses  (n=20/cell)"]
    ylabels = ["ASR (Qwen-72B JBB judge)",
               "mean harm Likert (Qwen-72B Stage-B, 1-5)"]

    for col, subset in enumerate([wins, losses]):
        for row, metric in enumerate(["asr", "harm"]):
            ax = axes[row, col]
            for axis, color, label in [
                    ("comp", "#3b6fb0", "α_c (with α_h=0)"),
                    ("harm", "#d65a31", "α_h (with α_c=0)")]:
                bins = aggregate(subset, axis, layer=16)
                xs, ys, ylos, yhis = [], [], [], []
                for alpha in sorted(bins.keys()):
                    if not (COMMON_RANGE[0]-1e-9 <= alpha <= COMMON_RANGE[1]+1e-9):
                        continue
                    rs = bins[alpha]
                    if metric == "asr":
                        comps = [bool(r['judge_compliance_unsafe'])
                                 for r in rs
                                 if r['judge_compliance_unsafe'] is not None]
                        if not comps:
                            continue
                        n_pos = sum(comps); n = len(comps)
                        p = n_pos / n
                        lo, hi = wilson(n_pos, n)
                        xs.append(alpha); ys.append(100 * p)
                        ylos.append(100 * (p - lo))
                        yhis.append(100 * (hi - p))
                    else:
                        vals = [r['judge_harm_likert'] for r in rs
                                if r['judge_harm_likert'] is not None]
                        if not vals:
                            continue
                        m, lo, hi = boot_mean(vals)
                        xs.append(alpha); ys.append(m)
                        ylos.append(m - lo); yhis.append(hi - m)
                ax.errorbar(xs, ys, yerr=[ylos, yhis],
                            marker="o", markersize=6, linewidth=1.6,
                            color=color, capsize=3, label=label)
            ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax.grid(alpha=0.25, linewidth=0.4)
            if row == 0:
                ax.set_title(titles_top[col], fontsize=11)
                ax.set_ylim(0, 105)
                ax.set_ylabel(ylabels[0] if col == 0 else "")
            else:
                ax.set_ylim(0.8, 4.2)
                ax.set_ylabel(ylabels[1] if col == 0 else "")
                ax.set_xlabel(r"$\alpha$")
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=9, frameon=True,
                          framealpha=0.92)

    fig.suptitle(
        "Two-axis steering at L16, extended to α ∈ [−1.0, +1.0]\n"
        "+α = amplify (more compliance / more harm); n=50 prompts (30 wins + 20 losses)",
        fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/steering_sweep_ext.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


def make_layer_compare_plot(rows):
    """Compare harm-direction effect at L16 vs L24 (wins meanHarm only)."""
    wins = [r for r in rows if r.get('prompt_type') == 'win']
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for layer, color, label in [
            (16, "#d65a31", "L16 (compliance peak)"),
            (24, "#7f3fbf", "L24 (harm peak)")]:
        bins = aggregate(wins, "harm", layer=layer)
        xs, ys, ylos, yhis = [], [], [], []
        for alpha in sorted(bins.keys()):
            rs = bins[alpha]
            vals = [r['judge_harm_likert'] for r in rs
                    if r['judge_harm_likert'] is not None]
            if not vals:
                continue
            m, lo, hi = boot_mean(vals)
            xs.append(alpha); ys.append(m)
            ylos.append(m - lo); yhis.append(hi - m)
        ax.errorbar(xs, ys, yerr=[ylos, yhis],
                    marker="o", markersize=7, linewidth=1.8,
                    color=color, capsize=3, label=label)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"$\alpha_h$  (harm-direction steering magnitude)")
    ax.set_ylabel("mean Stage-B harm Likert  (n=30 wins/cell, bootstrap 95% CI)")
    ax.set_title("Harm-direction causal effect: L16 vs L24",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.92)
    ax.set_ylim(0.8, 4.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/steering_L16_vs_L24_harm.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


def main():
    rows = load_all()
    print(f"Loaded total rows: {len(rows)}")
    print(f"  L16 rows: {sum(1 for r in rows if r.get('layer',16)==16)}")
    print(f"  L24 rows: {sum(1 for r in rows if r.get('layer',16)==24)}")
    make_main_plot(rows)
    make_layer_compare_plot(rows)

    # Print summary stats at extreme α
    print("\nL16 wins meanHarm across α_h (after sign flip irrelevant for α_h):")
    wins = [r for r in rows if r.get('prompt_type')=='win']
    bins = aggregate(wins, "harm", layer=16)
    for alpha in sorted(bins.keys()):
        vals = [r['judge_harm_likert'] for r in bins[alpha] if r['judge_harm_likert'] is not None]
        if vals:
            m, lo, hi = boot_mean(vals)
            print(f"  α_h={alpha:+.2f}: n={len(vals)}, mean={m:.2f}, "
                  f"95%CI=[{lo:.2f},{hi:.2f}]")

    print("\nL16 wins meanHarm across α_c (sign-flipped: +=amplify):")
    bins = aggregate(wins, "comp", layer=16)
    for alpha in sorted(bins.keys()):
        vals = [r['judge_harm_likert'] for r in bins[alpha] if r['judge_harm_likert'] is not None]
        if vals:
            m, lo, hi = boot_mean(vals)
            print(f"  α_c={alpha:+.2f}: n={len(vals)}, mean={m:.2f}, "
                  f"95%CI=[{lo:.2f},{hi:.2f}]")

    print("\nL24 wins meanHarm across α_h:")
    bins = aggregate(wins, "harm", layer=24)
    for alpha in sorted(bins.keys()):
        vals = [r['judge_harm_likert'] for r in bins[alpha] if r['judge_harm_likert'] is not None]
        if vals:
            m, lo, hi = boot_mean(vals)
            print(f"  α_h={alpha:+.2f}: n={len(vals)}, mean={m:.2f}, "
                  f"95%CI=[{lo:.2f},{hi:.2f}]")


if __name__ == "__main__":
    main()
