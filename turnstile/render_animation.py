"""Render MP4 animations of Turnstile multi-turn experiments.

Shows adversary and victim alternating turns in speech bubbles,
with a live ASR curve and breach-turn indicator.

Usage:
  python -m turnstile.render_animation --experiment-dir experiments/frozen_1Bv8B
  python -m turnstile.render_animation --all  # all frozen_* experiments
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BG_COLOR = "#0f0f1a"
PANEL_COLOR = "#1a1a2e"
ADV_COLOR = "#ef4444"
VIC_COLOR = "#3b82f6"
JUDGE_SAFE = "#22c55e"
JUDGE_UNSAFE = "#ef4444"
TEXT_COLOR = "#e2e8f0"
DIM_TEXT = "#64748b"
ASR_LINE = "#ef4444"
ASR_FILL = "#ef444433"
BUBBLE_ADV = "#2a1a1a"
BUBBLE_VIC = "#1a1a2a"

FPS = 30
ROUND_SECONDS = 5.0
INTRO_SECONDS = 2.0
OUTRO_SECONDS = 3.0
FRAMES_PER_ROUND = int(FPS * ROUND_SECONDS)
INTRO_FRAMES = int(FPS * INTRO_SECONDS)
OUTRO_FRAMES = int(FPS * OUTRO_SECONDS)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "images")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment(exp_dir, max_rounds=30):
    """Load round data and metrics."""
    rounds_data = []
    for r in range(max_rounds):
        path = os.path.join(exp_dir, "rounds", f"round_{r}.jsonl")
        entries = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        if entries:
            rounds_data.append(entries)

    metrics = []
    mpath = os.path.join(exp_dir, "metrics.jsonl")
    if os.path.exists(mpath):
        with open(mpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))

    return rounds_data, metrics


def pick_showcase(entries, max_items=2):
    """Pick representative conversations: prioritize wins, then losses."""
    wins = [e for e in entries if e.get("unsafe")]
    losses = [e for e in entries if not e.get("unsafe")]
    picked = wins[:max_items]
    if len(picked) < max_items:
        picked += losses[:max_items - len(picked)]
    return picked


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _wrap(text, width=50):
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width))[:200]


def _draw_conversation(ax, conversation, turn_of_breach, progress):
    """Draw multi-turn conversation with progressive reveal."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(PANEL_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])

    if not conversation:
        return

    # Show turns progressively
    n_msgs = len(conversation)
    visible = max(1, int(progress * n_msgs + 0.5))

    y = 0.95
    for i, msg in enumerate(conversation[:visible]):
        if i >= 6:  # max 6 messages shown
            break

        role = msg.get("role", "user")
        text = _wrap(msg.get("content", ""), width=55)
        is_adv = (role == "user")
        color = ADV_COLOR if is_adv else VIC_COLOR
        bg = BUBBLE_ADV if is_adv else BUBBLE_VIC
        x = 0.05 if is_adv else 0.50
        label = "ADV" if is_adv else "VIC"

        # Breach indicator
        turn_idx = i // 2  # each turn = user + assistant
        is_breach = (turn_of_breach is not None
                     and role == "assistant"
                     and turn_idx == turn_of_breach)

        lines = text.split("\n")
        box_height = len(lines) * 0.04 + 0.02
        rect = plt.Rectangle((x - 0.02, y - box_height - 0.01), 0.47,
                              box_height + 0.02, facecolor=bg,
                              edgecolor=color if is_breach else "none",
                              linewidth=2 if is_breach else 0,
                              alpha=0.8, transform=ax.transAxes,
                              clip_on=True)
        ax.add_patch(rect)

        ax.text(x, y, f"[{label}] {text}", fontsize=6, color=TEXT_COLOR,
                va="top", transform=ax.transAxes, fontfamily="monospace")

        if is_breach:
            ax.text(x + 0.35, y, "BREACH", fontsize=7, color=JUDGE_UNSAFE,
                    fontweight="bold", va="top", transform=ax.transAxes)

        y -= box_height + 0.03


def _draw_asr_chart(ax, metrics, current_round, progress):
    """Draw live ASR curve."""
    ax.set_facecolor(PANEL_COLOR)
    ax.set_xlim(-0.5, max(len(metrics), 10) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Round", fontsize=9, color=DIM_TEXT)
    ax.set_ylabel("ASR %", fontsize=9, color=DIM_TEXT)
    ax.tick_params(colors=DIM_TEXT, labelsize=8)
    ax.grid(True, alpha=0.15, color=DIM_TEXT)

    if not metrics:
        return

    # Draw up to current round
    n = min(current_round + 1, len(metrics))
    rounds = [m["round"] for m in metrics[:n]]
    asrs = [m["asr"] * 100 for m in metrics[:n]]

    if len(rounds) > 1:
        ax.fill_between(rounds, 0, asrs, color=ASR_FILL)
    ax.plot(rounds, asrs, "o-", color=ASR_LINE, linewidth=2, markersize=4)

    # Current value label
    if asrs:
        ax.text(rounds[-1] + 0.3, asrs[-1], f"{asrs[-1]:.0f}%",
                fontsize=9, color=ASR_LINE, fontweight="bold", va="center")


def render_experiment(exp_dir, output_path=None):
    """Render animation for one experiment."""
    rounds_data, metrics = load_experiment(exp_dir)
    if not rounds_data:
        print(f"  No data in {exp_dir}")
        return

    exp_name = os.path.basename(exp_dir)
    matchup_label = exp_name.replace("frozen_", "").replace("v", " vs ")

    if output_path is None:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        output_path = os.path.join(IMAGE_DIR, f"{exp_name}_animation.mp4")

    n_rounds = len(rounds_data)
    total_frames = INTRO_FRAMES + n_rounds * FRAMES_PER_ROUND + OUTRO_FRAMES

    # Pre-pick showcase conversations
    showcases = [pick_showcase(rd) for rd in rounds_data]

    # Setup figure
    fig = plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax_conv = fig.add_subplot(gs[0])
    ax_asr = fig.add_subplot(gs[1])

    fig.suptitle(f"Turnstile: {matchup_label} (Frozen Victim, Dual Judge)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.97)

    def update(frame):
        ax_conv.clear()
        ax_asr.clear()

        if frame < INTRO_FRAMES:
            # Intro
            ax_conv.set_facecolor(PANEL_COLOR)
            ax_conv.set_xticks([])
            ax_conv.set_yticks([])
            ax_conv.text(0.5, 0.5, f"{matchup_label}\nMulti-Turn Adversarial Red-Teaming",
                         ha="center", va="center", fontsize=16,
                         color=TEXT_COLOR, transform=ax_conv.transAxes,
                         fontweight="bold")
            _draw_asr_chart(ax_asr, metrics, -1, 0)
            return

        frame -= INTRO_FRAMES

        if frame >= n_rounds * FRAMES_PER_ROUND:
            # Outro
            ax_conv.set_facecolor(PANEL_COLOR)
            ax_conv.set_xticks([])
            ax_conv.set_yticks([])

            total_attacks = sum(len(rd) for rd in rounds_data)
            total_wins = sum(
                sum(1 for e in rd if e.get("unsafe")) for rd in rounds_data
            )
            overall_asr = total_wins / total_attacks * 100 if total_attacks else 0
            peak_asr = max(m["asr"] * 100 for m in metrics) if metrics else 0

            ax_conv.text(0.5, 0.6, "EXPERIMENT COMPLETE",
                         ha="center", va="center", fontsize=18,
                         color=TEXT_COLOR, fontweight="bold",
                         transform=ax_conv.transAxes)
            ax_conv.text(0.5, 0.4,
                         f"Rounds: {n_rounds}  |  "
                         f"Attacks: {total_attacks}  |  "
                         f"Wins: {total_wins}\n"
                         f"Overall ASR: {overall_asr:.1f}%  |  "
                         f"Peak ASR: {peak_asr:.1f}%",
                         ha="center", va="center", fontsize=12,
                         color=DIM_TEXT, transform=ax_conv.transAxes)

            _draw_asr_chart(ax_asr, metrics, n_rounds, 1.0)
            return

        # Normal round
        r = frame // FRAMES_PER_ROUND
        frame_in_round = frame % FRAMES_PER_ROUND
        progress = frame_in_round / FRAMES_PER_ROUND

        # Header
        n_wins = sum(1 for e in rounds_data[r] if e.get("unsafe"))
        n_total = len(rounds_data[r])
        asr = metrics[r]["asr"] * 100 if r < len(metrics) else 0

        ax_conv.set_facecolor(PANEL_COLOR)
        ax_conv.set_xticks([])
        ax_conv.set_yticks([])
        ax_conv.text(0.02, 0.98,
                     f"Round {r}  |  {n_wins}/{n_total} wins  |  "
                     f"ASR: {asr:.0f}%",
                     fontsize=10, color=TEXT_COLOR, va="top",
                     transform=ax_conv.transAxes, fontweight="bold")

        # Show conversation
        if showcases[r]:
            entry = showcases[r][0]
            conv = entry.get("conversation", [])
            tob = entry.get("turn_of_breach")
            goal = entry.get("goal", entry.get("behavior", ""))
            verdict = "UNSAFE" if entry.get("unsafe") else "SAFE"
            v_color = JUDGE_UNSAFE if entry.get("unsafe") else JUDGE_SAFE

            ax_conv.text(0.02, 0.90,
                         f"Goal: {_wrap(goal, 80)}",
                         fontsize=7, color=DIM_TEXT, va="top",
                         transform=ax_conv.transAxes)
            ax_conv.text(0.85, 0.90, verdict, fontsize=10, color=v_color,
                         fontweight="bold", va="top",
                         transform=ax_conv.transAxes)

            _draw_conversation(ax_conv, conv, tob, progress)

        _draw_asr_chart(ax_asr, metrics, r, progress)

    print(f"  Rendering {total_frames} frames ({n_rounds} rounds)...")
    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=1000 // FPS, blit=False)

    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)
    ani.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render MP4 animations of Turnstile experiments"
    )
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Single experiment directory")
    parser.add_argument("--all", action="store_true",
                        help="Render all frozen_* experiments")
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    os.makedirs(IMAGE_DIR, exist_ok=True)

    if args.experiment_dir:
        render_experiment(args.experiment_dir)
    elif args.all:
        for name in sorted(os.listdir(args.output_dir)):
            exp_dir = os.path.join(args.output_dir, name)
            if (name.startswith("frozen_") and name != "frozen_v1"
                    and os.path.isdir(exp_dir)):
                metrics_path = os.path.join(exp_dir, "metrics.jsonl")
                if os.path.exists(metrics_path):
                    print(f"\n=== {name} ===")
                    render_experiment(exp_dir)
    else:
        print("Specify --experiment-dir or --all")


if __name__ == "__main__":
    main()
