"""Collect per-turn victim hidden states for all JBB-labeled conversations.

Loads the victim model once, replays each conversation turn-by-turn,
and extracts the residual-stream hidden state at the last token before
each victim response. Saves with JBB labels for probe training.

Usage:
    python scripts/collect_hidden_states.py \
        --rejudge-files results/rejudge_jbb.jsonl results/rejudge_jbb_remaining.jsonl \
        --output-dir results/hidden_states_jbb
"""

import argparse
import json
import os

import torch
from turnstile.model_utils import load_model, extract_hidden_states, unload_model


def collect(rejudge_files, output_dir, victim_model="meta-llama/Llama-3.1-8B-Instruct",
            layer_idx=None, max_length=512):
    """Collect hidden states for all conversations in rejudge files."""
    os.makedirs(output_dir, exist_ok=True)

    # Load all conversations
    conversations = []
    for f in rejudge_files:
        if not os.path.exists(f):
            print(f"  Skipping {f} (not found)")
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    conversations.append(json.loads(line))

    print(f"Total conversations: {len(conversations)}")
    n_unsafe = sum(1 for c in conversations if c.get("jbb_unsafe", False))
    print(f"JBB unsafe: {n_unsafe} ({n_unsafe/len(conversations):.1%})")

    # Load victim
    print(f"\nLoading victim: {victim_model}")
    model, tok = load_model(victim_model)

    n_layers = model.config.num_hidden_layers
    if layer_idx is None:
        layer_idx = n_layers // 2
    print(f"Extracting from layer {layer_idx}/{n_layers}")

    # Collect hidden states per conversation
    all_hidden_states = []
    all_labels = []
    all_goals = []
    all_categories = []

    for i, conv_record in enumerate(conversations):
        conversation = conv_record.get("conversation", [])
        label = conv_record.get("jbb_unsafe", False)
        goal = conv_record.get("goal", "")
        category = conv_record.get("category", "")

        if not conversation:
            continue

        # Replay conversation turn by turn, collecting hidden states
        # before each victim response
        turn_hidden_states = []
        num_turns = len(conversation) // 2  # user/assistant pairs

        for t in range(num_turns):
            # Prefix up to and including the user message for this turn
            asst_idx = t * 2 + 1
            if asst_idx > len(conversation):
                break
            prefix = conversation[:t * 2 + 1]  # up to user message

            vic_prompt = tok.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True
            )

            try:
                h = extract_hidden_states(
                    model, tok, [vic_prompt],
                    layer_idx=layer_idx, max_length=max_length,
                )
                turn_hidden_states.append(h.squeeze(0))
            except RuntimeError:
                # OOM on long conversations
                break

        if turn_hidden_states:
            hs_tensor = torch.stack(turn_hidden_states)  # (num_turns, d_model)
            all_hidden_states.append(hs_tensor)
            all_labels.append(label)
            all_goals.append(goal)
            all_categories.append(category)

        if (i + 1) % 50 == 0:
            n_pos = sum(all_labels)
            print(f"  Processed {i+1}/{len(conversations)} "
                  f"({n_pos} unsafe so far)")

    unload_model(model, tok)

    # Save
    output_path = os.path.join(output_dir, "hidden_states_jbb.pt")
    torch.save({
        "hidden_states": all_hidden_states,
        "labels": torch.tensor(all_labels, dtype=torch.bool),
        "goals": all_goals,
        "categories": all_categories,
        "layer_idx": layer_idx,
        "victim_model": victim_model,
    }, output_path)

    print(f"\nSaved {len(all_hidden_states)} conversations to {output_path}")
    print(f"  Unsafe: {sum(all_labels)}, Safe: {len(all_labels) - sum(all_labels)}")
    print(f"  Hidden state shape per conv: (num_turns, {all_hidden_states[0].shape[-1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejudge-files", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/hidden_states_jbb")
    parser.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    collect(args.rejudge_files, args.output_dir, args.victim_model, args.layer)


if __name__ == "__main__":
    main()
