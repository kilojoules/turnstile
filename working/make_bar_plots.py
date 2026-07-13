"""Per-category bar plots: ASR and compliance-prediction AUC.

Both come from the pooled within-category probe corpus
(experiments/outcome_probe_v1/within_category_cv_pooled.json), 9,400 multi-turn
conversations across 11 source experiments.

  - ASR per category: n_pos / n; error bar = binomial SE = sqrt(p(1-p)/n)
  - Compliance AUC per category: 5-fold CV mean; error bar = CV std

Note on intent AUC: harmful-intent detection is defined cross-corpus
(adversarial vs benign Alpaca), so it has no meaningful per-category
breakdown — the "category" only exists on the adversarial side.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/julianquick/portfolio_copy/turnstile")
DATA = ROOT / "experiments" / "outcome_probe_v1" / "within_category_cv_pooled.json"
OUT = Path(__file__).parent / "bar_plots.png"


def load_per_category():
    blob = json.loads(DATA.read_text())
    rows = []
    for cat, stats in blob["per_category"].items():
        n = stats["n"]
        n_pos = stats["n_pos"]
        p = n_pos / n
        rows.append({
            "category": cat,
            "n": n,
            "n_pos": n_pos,
            "asr": p,
            "asr_se": math.sqrt(p * (1 - p) / n),
            "auc": stats["auc_mean"],
            "auc_std": stats["auc_std"],
        })
    rows.sort(key=lambda r: r["asr"], reverse=True)
    return rows, blob.get("total_convs"), blob.get("total_wins")


def short_label(name: str) -> str:
    aliases = {
        "Harassment/Discrimination": "Harassment /\nDiscrimination",
        "Government decision-making": "Govt.\ndecision-making",
        "Sexual/Adult content": "Sexual /\nAdult",
        "Fraud/Deception": "Fraud /\nDeception",
        "Malware/Hacking": "Malware /\nHacking",
        "Disinformation": "Disinfo",
        "Economic harm": "Economic\nharm",
        "Expert advice": "Expert\nadvice",
        "Physical harm": "Physical\nharm",
        "Privacy": "Privacy",
    }
    return aliases.get(name, name)


def asr_panel(ax, rows, total_convs, total_wins):
    labels = [short_label(r["category"]) for r in rows]
    means = [r["asr"] for r in rows]
    errs = [r["asr_se"] for r in rows]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=3, color="#c0392b", edgecolor="black", linewidth=0.6)
    pooled = total_wins / total_convs
    ax.axhline(pooled, color="black", linestyle="--", linewidth=0.8,
               label=f"pooled ASR = {pooled:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ASR (wins / conversations)")
    ax.set_title(f"Attack Success Rate by Category\n(n = {total_convs:,} convs; bars: binomial SE)")
    ax.set_ylim(0, max(m + e for m, e in zip(means, errs)) * 1.15)
    ax.legend(loc="upper right", fontsize=8)
    for xi, m, e, r in zip(x, means, errs, rows):
        ax.text(xi, m + e + 0.005, f"{m:.2f}\n(n={r['n']})", ha="center", fontsize=7)


def auc_panel(ax, rows):
    labels = [short_label(r["category"]) for r in rows]
    means = [r["auc"] for r in rows]
    errs = [r["auc_std"] for r in rows]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=3, color="#2ca02c", edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    macro_mean = float(np.mean(means))
    ax.axhline(macro_mean, color="black", linestyle=":", linewidth=0.8,
               label=f"macro mean = {macro_mean:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Compliance-prediction AUC")
    ax.set_title("Compliance Prediction by Category\n(L16, turn 0; bars: 5-fold CV std)")
    ax.set_ylim(0.45, 0.85)
    ax.legend(loc="upper right", fontsize=8)
    for xi, m, e in zip(x, means, errs):
        ax.text(xi, m + e + 0.005, f"{m:.2f}", ha="center", fontsize=7)


def main():
    rows, total_convs, total_wins = load_per_category()
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    asr_panel(axes[0], rows, total_convs, total_wins)
    auc_panel(axes[1], rows)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print()
    print(f"{'category':32s}  {'ASR':>6s}  {'SE':>6s}    {'AUC':>6s}  {'std':>6s}    n")
    for r in rows:
        print(f"{r['category']:32s}  {r['asr']:6.3f}  {r['asr_se']:6.3f}    "
              f"{r['auc']:6.3f}  {r['auc_std']:6.3f}    {r['n']}")


if __name__ == "__main__":
    main()
