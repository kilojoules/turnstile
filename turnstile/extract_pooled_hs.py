"""Extract per-turn hidden states at all layers for pooled experiments.

For each conversation in each source experiment's rounds/*.jsonl, replay
through the victim model and extract hidden states at turns 0-4 at layers
[0,4,8,12,16,20,24,28,31].

Output: experiments/pooled_hs/{source}/round_N.pt with per-layer tensors.

Usage:
  python -m turnstile.extract_pooled_hs \
      --sources stealth_s42 control_s42 frozen_v1 \
      --output experiments/pooled_hs
"""
import argparse
import json
import os

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from turnstile.model_utils import load_model, unload_model


LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


@torch.inference_mode()
def extract_all_layers(model, tokenizer, text, max_length=512):
    """Extract hidden states at all LAYERS, last non-padding token."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length).to(model.device)
    out = model(**inputs, output_hidden_states=True)
    seq_len = int(inputs["attention_mask"].sum())
    return {L: out.hidden_states[L][0, seq_len - 1, :].float().cpu()
            for L in LAYERS}


def process_source(model, tokenizer, src_name, rounds_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    round_files = sorted(f for f in os.listdir(rounds_dir)
                         if f.startswith("round_") and f.endswith(".jsonl"))
    print(f"\n=== {src_name}: {len(round_files)} rounds ===")

    for rf in round_files:
        rnum = int(rf.replace("round_", "").replace(".jsonl", ""))
        out_pt = os.path.join(out_dir, f"round_{rnum}.pt")
        if os.path.exists(out_pt):
            print(f"  skip round_{rnum} (cached)")
            continue

        rows = [json.loads(line) for line in open(os.path.join(rounds_dir, rf))]
        # per-layer, per-turn, per-conv: list of tensors
        per_layer = {L: [] for L in LAYERS}
        labels = []
        turns_of_breach = []
        categories = []
        goals = []

        for i, row in enumerate(rows):
            msgs = row.get("conversation", [])
            if not msgs:
                continue
            labels.append(int(bool(row.get("unsafe"))))
            turns_of_breach.append(row.get("turn_of_breach"))
            categories.append(row.get("category", "unknown"))
            goals.append(row.get("goal", ""))

            # Extract per-turn: build prompt up to each user message
            per_turn_layer = {L: [] for L in LAYERS}
            accumulated = []
            turn_idx = 0
            for m in msgs:
                accumulated.append(m)
                if m["role"] == "user" and turn_idx < 5:
                    prompt = tokenizer.apply_chat_template(
                        accumulated, tokenize=False, add_generation_prompt=True
                    )
                    try:
                        hs_by_layer = extract_all_layers(model, tokenizer, prompt)
                        for L in LAYERS:
                            per_turn_layer[L].append(hs_by_layer[L])
                    except RuntimeError:
                        # OOM or truncation issue — pad with zeros
                        for L in LAYERS:
                            per_turn_layer[L].append(torch.zeros(4096))
                    turn_idx += 1

            # Pad to 5 turns with zeros
            while turn_idx < 5:
                for L in LAYERS:
                    per_turn_layer[L].append(torch.zeros(4096))
                turn_idx += 1

            for L in LAYERS:
                per_layer[L].append(torch.stack(per_turn_layer[L]))

            if (i + 1) % 25 == 0:
                print(f"  round_{rnum}: {i+1}/{len(rows)} convs")

        # Save
        out = {
            "layers": LAYERS,
            "labels": torch.tensor(labels, dtype=torch.bool),
            "turns_of_breach": turns_of_breach,
            "categories": categories,
            "goals": goals,
            "round": rnum,
            # per layer: (n_convs, 5, 4096)
            "hidden_states_by_layer": {L: torch.stack(per_layer[L]) for L in LAYERS},
        }
        torch.save(out, out_pt)
        print(f"  saved round_{rnum} ({len(labels)} convs)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True,
                   help="Experiment dir names under experiments/")
    p.add_argument("--output", type=str, default="experiments/pooled_hs")
    p.add_argument("--victim-model", type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct")
    args = p.parse_args()

    print(f"Loading victim: {args.victim_model}")
    model, tokenizer = load_model(args.victim_model)

    for src in args.sources:
        rounds_dir = f"experiments/{src}/rounds"
        out_dir = os.path.join(args.output, src)
        if not os.path.isdir(rounds_dir):
            print(f"[SKIP] {src}: no rounds dir")
            continue
        process_source(model, tokenizer, src, rounds_dir, out_dir)

    unload_model(model, tokenizer)
    print("\nEXTRACTION COMPLETE")


if __name__ == "__main__":
    main()
