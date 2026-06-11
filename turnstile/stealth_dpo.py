"""Probe-aware DPO pair construction for stealth adversary training.

Standard DPO pairs wins vs losses for the same goal. Probe-aware DPO
adds a second preference axis: for the same goal, a stealthy win
(low probe score) is preferred over a detected win (high probe score).

Pair types (ordered by informativeness):
  1. stealthy_win vs detected_win  (same goal, same turn depth)
  2. any_win vs loss               (same goal, same turn depth)

This extracts signal from EVERY conversation — losses teach what not
to do, and detected wins teach what to do *differently*.

Usage from stealth_loop:
    pairs = build_probe_aware_pairs(round_files)
    save_pairs(pairs, pairs_file)
    train_dpo(model_id, pairs_file, adapter_path, ...)
"""

import json
import os
import random
from collections import defaultdict

from turnstile.dpo import _victim_to_adversary


def build_probe_aware_pairs(round_files, per_turn=True, probe_threshold=0.5):
    """Build DPO pairs using probe scores as preference signal.

    Each round file must have records with:
      - conversation: victim-perspective messages
      - unsafe: bool
      - goal: str
      - probe_score: float (0=undetected, 1=detected)

    Args:
        round_files: Paths to round JSONL files.
        per_turn: Split multi-turn conversations into per-turn prefixes.
        probe_threshold: Score below which a win is considered "stealthy."

    Returns list of {
        "chosen": messages, "rejected": messages,
        "goal": str, "pair_type": str
    }.
    """
    convs_by_goal = defaultdict(lambda: {
        "stealthy_wins": [], "detected_wins": [], "losses": [],
    })

    for f in round_files:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                goal = r.get("goal", "")
                if not goal or not r.get("conversation"):
                    continue

                adv_msgs = _victim_to_adversary(r["conversation"], goal)
                probe_score = r.get("probe_score", 0.5)

                if per_turn:
                    asst_indices = [
                        i for i, m in enumerate(adv_msgs)
                        if m["role"] == "assistant"
                    ]
                    prefixes = [adv_msgs[:ai + 1] for ai in asst_indices]
                else:
                    prefixes = [adv_msgs]

                if r.get("unsafe", False):
                    bucket = ("stealthy_wins" if probe_score < probe_threshold
                              else "detected_wins")
                    for p in prefixes:
                        convs_by_goal[goal][bucket].append(p)
                else:
                    for p in prefixes:
                        convs_by_goal[goal]["losses"].append(p)

    pairs = []
    for goal, data in convs_by_goal.items():
        sw = data["stealthy_wins"]
        dw = data["detected_wins"]
        losses = data["losses"]

        # Type 1: stealthy win > detected win (same turn depth)
        for chosen in sw:
            for rejected in dw:
                if len(chosen) == len(rejected):
                    pairs.append({
                        "chosen": chosen,
                        "rejected": rejected,
                        "goal": goal,
                        "pair_type": "stealth_vs_detected",
                    })

        # Type 2: any win > loss (same turn depth)
        all_wins = sw + dw
        for chosen in all_wins:
            for rejected in losses:
                if len(chosen) == len(rejected):
                    pairs.append({
                        "chosen": chosen,
                        "rejected": rejected,
                        "goal": goal,
                        "pair_type": "win_vs_loss",
                    })

    random.shuffle(pairs)
    return pairs


def save_pairs(pairs, path):
    """Write pairs to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


def summarize_pairs(pairs):
    """Print a summary of pair types."""
    from collections import Counter
    types = Counter(p["pair_type"] for p in pairs)
    print(f"   [Probe-DPO] {len(pairs)} pairs total:")
    for t, n in types.most_common():
        print(f"     {t}: {n}")
