"""Compare LR-direction steering vs mean-diff direction steering at L16.

Two-row grid:
  Row 1: ASR vs α (wins)
  Row 2: harm Likert vs α (wins)
Two columns: LR direction (left), mean-diff direction (right).
Two lines per panel: α_c (sign-flipped: +=amplify compliance) and α_h.

LR data: sweep_judged.jsonl + sweep_new_judged.jsonl (L16 rows only)
Mean-diff data: sweep_meandiff_L16_judged.jsonl
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


def aggregate(rows, axis, flip_alpha_c=True):
    bins = defaultdict(list)
    for r in rows:
        if r.get("layer", 16) != 16:
            continue
        if r.get('prompt_type') != 'win':
            continue
        if axis == "comp" and abs(r['alpha_h']) < 1e-9:
            k = -r['alpha_c'] if flip_alpha_c else r['alpha_c']
            bins[round(k, 2)].append(r)
        elif axis == "harm" and abs(r['alpha_c']) < 1e-9:
            bins[round(r['alpha_h'], 2)].append(r)
    return bins


def main():
    # LR data (combined original + ext)
    lr_rows = []
    for path in [f"{ROOT}/experiments/steering_v3/sweep_judged.jsonl",
                 f"{ROOT}/experiments/steering_v3/sweep_new_judged.jsonl"]:
        for line in open(path):
            lr_rows.append(json.loads(line))
    md_rows = [json.loads(l) for l in open(
        f"{ROOT}/experiments/steering_v3/sweep_meandiff_L16_judged.jsonl")]
    print(f"LR wins (L16): {sum(1 for r in lr_rows if r.get('layer',16)==16 and r.get('prompt_type')=='win')}")
    print(f"Mean-diff wins (L16): {sum(1 for r in md_rows if r.get('layer',16)==16 and r.get('prompt_type')=='win')}")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex='col',
                              sharey='row')
    col_titles = ["LR-direction (predicts-class axis)",
                  "Mean-diff direction (Arditi-style cluster-displacement)"]
    ylabels = ["wins ASR (Qwen-72B JBB judge)",
               "wins mean harm Likert (Qwen-72B Stage-B)"]

    # For mean-diff, we did NOT flip alpha_c sign (its causal effect is
    # untested; the LR-flip was empirical). Show both as-recorded.
    for col, (rows, flip_c) in enumerate([(lr_rows, True), (md_rows, False)]):
        for row, metric in enumerate(["asr", "harm"]):
            ax = axes[row, col]
            for axis, color, label in [
                    ("comp", "#3b6fb0", "α_c (with α_h=0)"),
                    ("harm", "#d65a31", "α_h (with α_c=0)")]:
                bins = aggregate(rows, axis, flip_alpha_c=flip_c)
                xs, ys, ylos, yhis = [], [], [], []
                for alpha in sorted(bins.keys()):
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
                ax.set_title(col_titles[col], fontsize=10.5)
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
        "L16 steering: LR direction vs mean-difference (Arditi-style) direction\n"
        "wins only (n=30/cell); LR α_c sign-flipped (+ = amplify); mean-diff α_c as-recorded",
        fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/steering_meandiff_vs_lr.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    # Print numerics for mean-diff
    print("\nMean-diff wins meanHarm vs α_h:")
    bins = aggregate(md_rows, "harm")
    for alpha in sorted(bins.keys()):
        vals = [r['judge_harm_likert'] for r in bins[alpha]
                if r['judge_harm_likert'] is not None]
        if vals:
            m, lo, hi = boot_mean(vals)
            print(f"  α_h={alpha:+.2f}: n={len(vals)}, mean={m:.2f}, "
                  f"95%CI=[{lo:.2f},{hi:.2f}]")
    print("\nMean-diff wins meanHarm vs α_c (as-recorded):")
    bins = aggregate(md_rows, "comp", flip_alpha_c=False)
    for alpha in sorted(bins.keys()):
        vals = [r['judge_harm_likert'] for r in bins[alpha]
                if r['judge_harm_likert'] is not None]
        if vals:
            m, lo, hi = boot_mean(vals)
            print(f"  α_c={alpha:+.2f}: n={len(vals)}, mean={m:.2f}, "
                  f"95%CI=[{lo:.2f},{hi:.2f}]")
    print("\nMean-diff wins ASR vs α_c (as-recorded):")
    bins = aggregate(md_rows, "comp", flip_alpha_c=False)
    for alpha in sorted(bins.keys()):
        comps = [bool(r['judge_compliance_unsafe']) for r in bins[alpha]
                 if r['judge_compliance_unsafe'] is not None]
        if comps:
            p = sum(comps)/len(comps)
            lo, hi = wilson(sum(comps), len(comps))
            print(f"  α_c={alpha:+.2f}: n={len(comps)}, ASR={100*p:.1f}%, "
                  f"95%CI=[{100*lo:.1f}%, {100*hi:.1f}%]")


if __name__ == "__main__":
    main()
