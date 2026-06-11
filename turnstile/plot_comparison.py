"""Cross-matchup comparison charts for Turnstile experiments.

Generates:
  1. Multi-matchup ASR curves (all matchups on one plot)
  2. Category vulnerability heatmap (matchup × JBB category)
  3. Breach turn heatmap per matchup
  4. Judge agreement analysis (Guard vs 70B)
  5. Summary statistics table

Usage:
  python -m turnstile.plot_comparison --experiment-dirs experiments/frozen_*
  python -m turnstile.plot_comparison  # auto-discovers frozen_* experiments
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "images")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(exp_dir):
    path = os.path.join(exp_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_rounds(exp_dir):
    rounds_dir = os.path.join(exp_dir, "rounds")
    if not os.path.isdir(rounds_dir):
        return []
    data = []
    for fname in sorted(os.listdir(rounds_dir)):
        if fname.startswith("round_") and fname.endswith(".jsonl"):
            r = int(fname.split("_")[1].split(".")[0])
            with open(os.path.join(rounds_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec["round"] = r
                        data.append(rec)
    return data


def discover_experiments(output_dir):
    """Find all frozen_* experiment directories."""
    exps = {}
    for name in sorted(os.listdir(output_dir)):
        if name.startswith("frozen_") and name != "frozen_v1":
            metrics = load_metrics(os.path.join(output_dir, name))
            if metrics:
                exps[name] = os.path.join(output_dir, name)
    return exps


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_asr_curves(experiments, output_name="matchup_comparison"):
    """Multi-matchup ASR curves on one plot."""
    import matplotlib.pyplot as plt

    colors = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, exp_dir) in enumerate(experiments.items()):
        metrics = load_metrics(exp_dir)
        if not metrics:
            continue
        rounds = [m["round"] for m in metrics]
        asrs = [m["asr"] * 100 for m in metrics]
        label = name.replace("frozen_", "").replace("v", " vs ")
        c = colors[i % len(colors)]
        ax.plot(rounds, asrs, "o-", color=c, linewidth=2, markersize=5,
                label=label)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("ASR by Adversary–Victim Matchup (Dual Judge, JBB)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_DIR, f"asr_comparison_{output_name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_category_heatmap(experiments, output_name="matchup_comparison"):
    """Category × matchup vulnerability heatmap."""
    import matplotlib.pyplot as plt

    # Collect per-category ASR for each matchup
    all_categories = set()
    matchup_cat_asr = {}

    for name, exp_dir in experiments.items():
        data = load_all_rounds(exp_dir)
        if not data:
            continue

        cat_total = Counter()
        cat_wins = Counter()
        for d in data:
            cat = d.get("category", "Unknown")
            all_categories.add(cat)
            cat_total[cat] += 1
            if d.get("unsafe"):
                cat_wins[cat] += 1

        matchup_cat_asr[name] = {
            cat: cat_wins[cat] / cat_total[cat] * 100
            if cat_total[cat] > 0 else 0
            for cat in all_categories
        }

    if not matchup_cat_asr:
        return

    categories = sorted(all_categories)
    matchup_names = list(matchup_cat_asr.keys())
    labels = [n.replace("frozen_", "").replace("v", " vs ")
              for n in matchup_names]

    matrix = np.array([
        [matchup_cat_asr[m].get(c, 0) for c in categories]
        for m in matchup_names
    ])

    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 0.9),
                                    max(4, len(matchup_names) * 1.2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(matchup_names)))
    ax.set_yticklabels(labels, fontsize=11)

    # Annotate cells
    for i in range(len(matchup_names)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = "white" if val > 40 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Category Vulnerability by Matchup (% ASR)",
                 fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="ASR (%)", shrink=0.8)

    fig.tight_layout()
    path = os.path.join(IMAGE_DIR, f"category_heatmap_{output_name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_breach_turns(experiments, output_name="matchup_comparison"):
    """Breach turn distribution per matchup."""
    import matplotlib.pyplot as plt

    matchup_data = {}
    for name, exp_dir in experiments.items():
        data = load_all_rounds(exp_dir)
        wins = [d for d in data if d.get("unsafe")]
        if wins:
            turns = [d.get("turn_of_breach", 0) or 0 for d in wins]
            matchup_data[name] = turns

    if not matchup_data:
        return

    fig, axes = plt.subplots(1, len(matchup_data),
                             figsize=(4 * len(matchup_data), 4),
                             sharey=True)
    if len(matchup_data) == 1:
        axes = [axes]

    colors = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b"]

    for i, (name, turns) in enumerate(matchup_data.items()):
        ax = axes[i]
        label = name.replace("frozen_", "").replace("v", " vs ")
        max_turn = max(turns) if turns else 4
        bins = range(0, max_turn + 2)
        ax.hist(turns, bins=bins, color=colors[i % len(colors)],
                edgecolor="white", alpha=0.8, align="left")
        ax.set_xlabel("Turn of Breach", fontsize=10)
        if i == 0:
            ax.set_ylabel("Count", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Turn-of-Breach Distribution by Matchup",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(IMAGE_DIR, f"breach_turns_{output_name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_judge_agreement(experiments, output_name="matchup_comparison"):
    """Judge agreement rates across matchups."""
    import matplotlib.pyplot as plt

    matchup_stats = {}
    for name, exp_dir in experiments.items():
        data = load_all_rounds(exp_dir)
        if not data or "guard_unsafe" not in data[0]:
            continue  # no dual-judge data

        total = len(data)
        guard_unsafe = sum(1 for d in data if d.get("guard_unsafe"))
        together_unsafe = sum(1 for d in data if d.get("together_unsafe"))
        both_unsafe = sum(1 for d in data if d.get("unsafe"))
        disagree = sum(1 for d in data if d.get("disagreement"))

        matchup_stats[name] = {
            "total": total,
            "guard_rate": guard_unsafe / total * 100,
            "together_rate": together_unsafe / total * 100,
            "both_rate": both_unsafe / total * 100,
            "disagree_rate": disagree / total * 100,
        }

    if not matchup_stats:
        print("  No dual-judge data found — skipping agreement plot")
        return

    labels = [n.replace("frozen_", "").replace("v", " vs ")
              for n in matchup_stats]
    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))

    guard = [matchup_stats[n]["guard_rate"] for n in matchup_stats]
    together = [matchup_stats[n]["together_rate"] for n in matchup_stats]
    both = [matchup_stats[n]["both_rate"] for n in matchup_stats]
    disagree = [matchup_stats[n]["disagree_rate"] for n in matchup_stats]

    ax.bar(x - 1.5 * w, guard, w, color="#ef4444", label="Guard only",
           edgecolor="white")
    ax.bar(x - 0.5 * w, together, w, color="#3b82f6", label="70B only",
           edgecolor="white")
    ax.bar(x + 0.5 * w, both, w, color="#8b5cf6", label="Both agree",
           edgecolor="white")
    ax.bar(x + 1.5 * w, disagree, w, color="#f59e0b", label="Disagree",
           edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% of conversations", fontsize=11)
    ax.set_title("Dual-Judge Agreement by Matchup",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(IMAGE_DIR, f"judge_agreement_{output_name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(experiments):
    """Print summary statistics."""
    print(f"\n{'Name':<20} | {'Rounds':<7} | {'Total':<7} | {'Wins':<6} | "
          f"{'Mean ASR':<10} | {'Peak ASR':<10} | {'Final ASR':<10}")
    print("-" * 85)

    for name, exp_dir in experiments.items():
        metrics = load_metrics(exp_dir)
        data = load_all_rounds(exp_dir)
        if not metrics:
            continue
        total = sum(m["candidates"] for m in metrics)
        wins = sum(m["wins"] for m in metrics)
        asrs = [m["asr"] for m in metrics]
        label = name.replace("frozen_", "").replace("v", " vs ")
        print(f"{label:<20} | {len(metrics):<7} | {total:<7} | {wins:<6} | "
              f"{sum(asrs)/len(asrs):.1%}{'':>5} | {max(asrs):.1%}{'':>5} | "
              f"{asrs[-1]:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-matchup comparison charts"
    )
    parser.add_argument("--experiment-dirs", nargs="*", type=str, default=None,
                        help="Experiment directories (auto-discovers if omitted)")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--output-name", type=str, default="matchup_comparison")
    args = parser.parse_args()

    if args.experiment_dirs:
        experiments = {}
        for d in args.experiment_dirs:
            name = os.path.basename(d.rstrip("/"))
            experiments[name] = d
    else:
        experiments = discover_experiments(args.output_dir)

    if not experiments:
        print("No experiments found.")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments: "
          f"{list(experiments.keys())}")

    try:
        import matplotlib  # noqa
        print("\n=== Generating plots ===")
        plot_asr_curves(experiments, args.output_name)
        plot_category_heatmap(experiments, args.output_name)
        plot_breach_turns(experiments, args.output_name)
        plot_judge_agreement(experiments, args.output_name)
    except ImportError:
        print("matplotlib not available — skipping plots")

    print_summary_table(experiments)


if __name__ == "__main__":
    main()
