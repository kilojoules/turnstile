"""Plot ASR over training rounds for the 11 DPO/self-play runs that
contributed hidden states to the pooled probe analysis.

Reads each run's `metrics.jsonl` and writes
`figures/dpo_training_progress.{pdf,png}`.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile/experiments"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"

# (run, display_label, family, color, linestyle)
RUNS = [
    ("authority_dpo",       "authority",        "variant", "#1f77b4", "-"),
    ("incrementalism_dpo",  "incrementalism",   "variant", "#2ca02c", "-"),
    ("reward_dpo",          "reward",           "variant", "#9467bd", "-"),
    ("urgency_dpo",         "urgency",          "variant", "#17becf", "-"),
    ("stealth_jbb_v1",      "stealth-jbb",      "variant", "#bcbd22", "-"),

    ("stealth_s42",         "stealth s42",      "stealth", "#d62728", "-"),
    ("stealth_hard_s456",   "stealth-hard s456","stealth", "#d62728", "--"),

    ("control_s42",         "control s42",      "control", "#7f7f7f", "-"),
    ("control_hard_s456",   "control-hard s456","control", "#7f7f7f", "--"),

    ("frozen_v1",           "frozen victim",    "ablation","#8c564b", "-"),
    ("urgency_v1",          "urgency-v1",       "ablation","#e377c2", "-"),
]


def load_asr(run):
    path = f"{ROOT}/{run}/metrics.jsonl"
    rounds, asrs = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rounds.append(r["round"])
            asrs.append(100 * r["asr"])
    return rounds, asrs


def main():
    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    for run, label, family, color, ls in RUNS:
        rounds, asrs = load_asr(run)
        ax.plot(rounds, asrs, marker="o", markersize=3.5, linewidth=1.4,
                color=color, linestyle=ls, label=label, alpha=0.9)

    ax.set_xlabel("DPO self-play round")
    ax.set_ylabel("ASR on JailbreakBench (%, pivot-only judge)")
    ax.set_title("Adversary training progress across the 11 runs feeding the probe pool",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0, None)

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, frameon=False, title="run", title_fontsize=9)

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/dpo_training_progress.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
