"""Collect per-turn hidden states from saved round data.

Replays conversations through the frozen victim model and extracts
hidden states at each turn. Used when the main loop was run with
--no-hidden-states and we need to backfill for probe/T-SAE analysis.

Usage:
  python -m turnstile.collect_hidden_states \
      --experiment-dir experiments/frozen_v1
"""

import argparse
import json
import os

import torch

from turnstile.model_utils import load_model, extract_hidden_states, unload_model


def collect(experiment_dir, victim_model="meta-llama/Llama-3.1-8B-Instruct",
            layer_idx=None, max_length=256):
    """Extract per-turn hidden states from saved round conversations."""
    rounds_dir = os.path.join(experiment_dir, "rounds")
    hs_dir = os.path.join(experiment_dir, "hidden_states")
    os.makedirs(hs_dir, exist_ok=True)

    round_files = sorted(
        f for f in os.listdir(rounds_dir)
        if f.startswith("round_") and f.endswith(".jsonl")
    )
    if not round_files:
        print("No round data found")
        return

    print(f"Loading victim model: {victim_model}")
    model, tok = load_model(victim_model)

    for fname in round_files:
        round_num = int(fname.split("_")[1].split(".")[0])
        hs_path = os.path.join(hs_dir, f"round_{round_num}.pt")
        if os.path.exists(hs_path):
            print(f"  Round {round_num}: already collected, skipping")
            continue

        # Load round data
        with open(os.path.join(rounds_dir, fname)) as f:
            records = [json.loads(l) for l in f if l.strip()]

        print(f"  Round {round_num}: {len(records)} conversations")

        all_hidden_states = []
        labels = []
        turns_of_breach = []

        for i, rec in enumerate(records):
            conv = rec["conversation"]
            # Extract hidden state at each turn (before victim response)
            turn_states = []
            for t in range(len(conv) // 2):
                # Build victim prompt up to this turn's adversary message
                prefix = conv[:t * 2 + 1]  # up to and including adversary msg
                vic_prompt = tok.apply_chat_template(
                    prefix, tokenize=False, add_generation_prompt=True
                )
                try:
                    h = extract_hidden_states(
                        model, tok, [vic_prompt],
                        layer_idx=layer_idx, max_length=max_length,
                    )
                    turn_states.append(h.squeeze(0))
                except RuntimeError:
                    # OOM on long prompts — truncate this conversation
                    break

            if turn_states:
                all_hidden_states.append(torch.stack(turn_states))
            else:
                all_hidden_states.append(None)

            labels.append(rec["unsafe"])
            turns_of_breach.append(rec.get("turn_of_breach"))

        # Filter out None entries
        valid_hs = [hs for hs in all_hidden_states if hs is not None]
        valid_labels = [l for hs, l in zip(all_hidden_states, labels)
                        if hs is not None]
        valid_tob = [t for hs, t in zip(all_hidden_states, turns_of_breach)
                     if hs is not None]

        data = {
            "hidden_states": valid_hs,
            "labels": torch.tensor(valid_labels, dtype=torch.bool),
            "turns_of_breach": valid_tob,
            "round": round_num,
        }
        torch.save(data, hs_path)
        print(f"    Saved {len(valid_hs)} conversations to {hs_path}")

    unload_model(model, tok)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect per-turn hidden states from round data"
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    collect(args.experiment_dir, args.victim_model, args.layer, args.max_length)
