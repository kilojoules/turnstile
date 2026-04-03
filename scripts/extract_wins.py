"""Extract verified wins from round JSONL files into training format.

Converts victim-perspective conversations (user=adversary, assistant=victim)
into adversary-perspective training data (system=goal, user=victim feedback,
assistant=adversary attacks).

Usage:
    python scripts/extract_wins.py \
        --rounds-dir results_3090_rounds \
        --output-dir data/base_adversary
"""

import argparse
import json
import os
import glob


def victim_to_adversary_perspective(conversation, goal):
    """Convert victim-perspective conversation to adversary training format.

    Victim perspective:
        [{"role": "user", "content": attack}, {"role": "assistant", "content": response}, ...]

    Adversary perspective:
        [{"role": "system", "content": "Goal: ..."},
         {"role": "user", "content": "Begin the conversation."},
         {"role": "assistant", "content": attack_1},
         {"role": "user", "content": "Target responded: response_1\\n\\nGenerate next message."},
         {"role": "assistant", "content": attack_2},
         ...]
    """
    adv_messages = [
        {"role": "system", "content": f"Goal: {goal}"},
        {"role": "user", "content": "Begin the conversation."},
    ]

    for i, msg in enumerate(conversation):
        if msg["role"] == "user":
            # This is an adversary attack
            adv_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "assistant":
            # This is a victim response — becomes adversary's "user" input
            adv_messages.append({
                "role": "user",
                "content": f"Target responded: {msg['content']}\n\nGenerate next message.",
            })

    return adv_messages


def extract_wins(rounds_dir, output_dir):
    """Extract all verified wins from round JSONL files."""
    os.makedirs(output_dir, exist_ok=True)

    round_files = sorted(glob.glob(os.path.join(rounds_dir, "round_*.jsonl")))
    if not round_files:
        print(f"No round files found in {rounds_dir}")
        return

    all_wins = []
    for round_file in round_files:
        round_name = os.path.basename(round_file).replace(".jsonl", "")
        round_wins = []

        with open(round_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not record.get("unsafe", False):
                    continue

                goal = record.get("goal", "")
                conversation = record.get("conversation", [])
                if not conversation or not goal:
                    continue

                adv_messages = victim_to_adversary_perspective(conversation, goal)
                # Verify we have at least one assistant turn
                n_assistant = sum(1 for m in adv_messages if m["role"] == "assistant")
                if n_assistant == 0:
                    continue

                round_wins.append({"messages": adv_messages})

        # Save per-round wins
        if round_wins:
            wins_file = os.path.join(output_dir, f"{round_name}_wins.jsonl")
            with open(wins_file, "w") as f:
                for item in round_wins:
                    f.write(json.dumps(item) + "\n")

        all_wins.extend(round_wins)
        print(f"  {round_name}: {round_wins.__len__()} wins")

    # Save combined training file
    train_file = os.path.join(output_dir, "train.jsonl")
    with open(train_file, "w") as f:
        for item in all_wins:
            f.write(json.dumps(item) + "\n")

    print(f"\nTotal: {len(all_wins)} verified wins -> {train_file}")
    return all_wins


def merge_win_sources(*dirs, output_path):
    """Merge wins from multiple source directories into a single train.jsonl."""
    all_wins = []
    for d in dirs:
        pattern = os.path.join(d, "round_*_wins.jsonl")
        for f in sorted(glob.glob(pattern)):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_wins.append(json.loads(line))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in all_wins:
            f.write(json.dumps(item) + "\n")

    print(f"Merged {len(all_wins)} wins -> {output_path}")
    return all_wins


def main():
    parser = argparse.ArgumentParser(description="Extract verified wins into training format")
    parser.add_argument("--rounds-dir", type=str, required=True,
                        help="Directory with round_*.jsonl files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for wins JSONL files")
    parser.add_argument("--merge-with", type=str, nargs="*", default=[],
                        help="Additional win directories to merge with")
    parser.add_argument("--merge-output", type=str, default=None,
                        help="Output path for merged train.jsonl")
    args = parser.parse_args()

    wins = extract_wins(args.rounds_dir, args.output_dir)

    if args.merge_with and args.merge_output:
        merge_win_sources(args.output_dir, *args.merge_with,
                          output_path=args.merge_output)


if __name__ == "__main__":
    main()
