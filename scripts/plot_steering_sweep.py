"""Plot the two-axis steering sweep: ASR and mean harm Likert vs alpha,
along the compliance axis and the harm axis, separately for wins and losses.

Output: figures/steering_sweep.{pdf,png}
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


def main():
    rows = [json.loads(l) for l in open(
        f"{ROOT}/experiments/steering_v3/sweep_judged.jsonl")]
    print(f"Loaded {len(rows)} judged rows")

    # Sign convention: flip alpha_c so that +alpha_c means "amplify compliance"
    # (matches +alpha_h meaning "amplify harm"). The causal effect on the model
    # is what was originally produced at -alpha_c in the JSONL.
    def aggregate(subset, axis):
        """axis='comp' or 'harm'; collect cells where the OTHER axis == 0.
        Returns dict keyed by the *display* alpha (flipped sign for comp)."""
        bins = defaultdict(list)
        for r in subset:
            if axis == "comp" and abs(r['alpha_h']) < 1e-9:
                bins[round(-r['alpha_c'], 2)].append(r)  # FLIPPED
            elif axis == "harm" and abs(r['alpha_c']) < 1e-9:
                bins[round(r['alpha_h'], 2)].append(r)
        return bins

    wins = [r for r in rows if r.get('prompt_type') == 'win']
    losses = [r for r in rows if r.get('prompt_type') == 'loss']

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex='col',
                              sharey='row')
    titles = [["Wins  (n=30/cell)", "Losses  (n=20/cell)"],
              ["Wins  (n=30/cell)", "Losses  (n=20/cell)"]]
    ylabels = ["ASR (Qwen-72B JBB judge)",
               "mean harm Likert (Qwen-72B Stage-B, 1-5)"]

    # Common range after the α_c sign flip — both curves sampled at the
    # same x positions.
    COMMON_RANGE = (-0.5, 0.5)

    for col, subset in enumerate([wins, losses]):
        for row, metric in enumerate(["asr", "harm"]):
            ax = axes[row, col]
            for ai, (axis, color, label) in enumerate([
                    ("comp", "#3b6fb0", "α_c (with α_h=0)"),
                    ("harm", "#d65a31", "α_h (with α_c=0)")]):
                bins = aggregate(subset, axis)
                xs, ys, ylos, yhis = [], [], [], []
                for alpha in sorted(bins.keys()):
                    if not (COMMON_RANGE[0] - 1e-9 <= alpha
                            <= COMMON_RANGE[1] + 1e-9):
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
                ax.set_title(titles[row][col], fontsize=11)
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
        "Two-axis steering at L16: compliance probe direction vs harm probe direction\n"
        "sign convention: +α = amplify (more compliance / more harm); "
        "α range truncated to common −0.5 to +0.5; "
        "n=50 (30 wins + 20 losses); α·||h||·v at L16, greedy",
        fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/steering_sweep.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
