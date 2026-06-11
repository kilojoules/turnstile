"""Replay rewritten conversations through the frozen victim and extract
matched-turn hidden states.

Input: the JSONL produced by ``intent_rewrite.py``. Each record has the
original conversation, the breach turn ``t*``, and a list of
``{turn, original, rewrite}`` user-turn rewrites for turns ``t*..n_turns-1``.

For each record we replay the conversation deterministically:
  1. Take turns ``0..t*-1`` of the original (user + assistant) verbatim.
  2. At each turn ``t >= t*`` substitute the rewritten user message and
     generate the assistant response with temperature 0, max 300 tokens.
  3. Extract the L16 hidden state at the last token of the user-message prefix
     for *both* the original conversation and the rewritten conversation,
     so we have matched-turn paired hidden states.

Output: a single ``.pt`` file containing a list of records, each with::

  {
    "source": str,
    "turn_of_breach": int,
    "original_conversation": [...],     # the unsafe one
    "translated_conversation": [...],   # benign, same prelude
    "original_hs": tensor[num_turns, 4096],
    "translated_hs": tensor[num_turns, 4096],
    "matched_turn_hs": list of (t, original_h, translated_h)  # only for t >= t*
  }

Also re-judges the translated conversation end-to-end and stores the verdict.

Usage (on a GPU box, ~20 GB VRAM enough with 4-bit quant):
  python -m turnstile.intent_replay \
      --rewrites-jsonl experiments/intent_obliteration_paired/rewrites.jsonl \
      --output experiments/intent_obliteration_paired/replay.pt \
      --victim-model meta-llama/Llama-3.1-8B-Instruct \
      --together-key ~/.together
"""

import argparse
import json
import os

import torch

from turnstile.model_utils import (
    load_model, generate_text, extract_hidden_states, unload_model,
)


def hs_at_user_prefix(model, tok, conv_msgs, layer_idx=None,
                     max_length=2048):
    """Hidden state at last token of the prompt that ends with the latest
    user message (i.e., what the victim sees right before generating)."""
    prompt = tok.apply_chat_template(
        conv_msgs, tokenize=False, add_generation_prompt=True,
    )
    return extract_hidden_states(
        model, tok, [prompt], layer_idx=layer_idx, max_length=max_length,
    ).squeeze(0)


def replay_conversation(model, tok, original_conv, rewrites, tob,
                        gen_max_tokens=300, layer_idx=None):
    """Replay: keep turns 0..tob-1 verbatim, substitute rewritten user turns
    at tob..end and generate assistant turns deterministically.

    Returns (translated_conversation, original_hs_list, translated_hs_list).
    """
    n_user_turns = sum(1 for m in original_conv if m["role"] == "user")
    rewrite_by_turn = {r["turn"]: r["rewrite"] for r in rewrites}

    # Build translated conversation incrementally.
    translated = list(original_conv[:2 * tob])  # turns 0..tob-1 verbatim

    # Hidden states from original (extract at each user-prefix end).
    original_hs = []
    for t in range(n_user_turns):
        prefix = original_conv[: 2 * t + 1]  # up to and including user msg
        h = hs_at_user_prefix(model, tok, prefix, layer_idx=layer_idx)
        original_hs.append(h)

    # Hidden states from translated (generate assistant turns on the fly).
    translated_hs = []
    for t in range(n_user_turns):
        if t < tob:
            # identical prefix as original; recompute deterministic HS
            prefix = translated[: 2 * t + 1]
            h = hs_at_user_prefix(model, tok, prefix, layer_idx=layer_idx)
            translated_hs.append(h)
        else:
            # substitute rewritten user message
            if t not in rewrite_by_turn:
                # rewriter didn't cover this turn (shouldn't happen for full pipelines)
                break
            translated.append({"role": "user", "content": rewrite_by_turn[t]})
            prefix = translated[:]
            h = hs_at_user_prefix(model, tok, prefix, layer_idx=layer_idx)
            translated_hs.append(h)
            # generate assistant response (T=0)
            prompt = tok.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True,
            )
            try:
                resp = generate_text(model, tok, prompt,
                                     max_tokens=gen_max_tokens, temperature=0.0)
            except torch.cuda.OutOfMemoryError as e:
                # Specific handler: prompt too long; truncate this conversation
                # and continue with subsequent ones. Other errors must propagate.
                print(f"    [OOM, truncating conversation: {e}]")
                torch.cuda.empty_cache()
                break
            translated.append({"role": "assistant", "content": resp.strip()})

    return translated, original_hs, translated_hs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rewrites-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--victim-model",
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layer-idx", type=int, default=None,
                   help="Default None -> n_layers // 2 (= L16 for Llama-3.1-8B)")
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    print(f"Loading rewrites from {args.rewrites_jsonl}...")
    rewrites_data = []
    with open(args.rewrites_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rewrites_data.append(json.loads(line))
    if args.max_records:
        rewrites_data = rewrites_data[:args.max_records]
    print(f"Loaded {len(rewrites_data)} rewritten conversations")

    print(f"Loading victim {args.victim_model}...")
    model, tok = load_model(args.victim_model)

    # Resumability: load existing output if any.
    existing = []
    if os.path.exists(args.output):
        existing = torch.load(args.output, weights_only=False)
        seen = {r["source"] for r in existing}
        print(f"Resuming -- {len(existing)} records already in {args.output}")
    else:
        seen = set()

    out_records = list(existing)
    for i, rec in enumerate(rewrites_data):
        if rec["source"] in seen:
            continue
        try:
            translated, orig_hs, trans_hs = replay_conversation(
                model, tok,
                original_conv=rec["conversation"],
                rewrites=rec["rewrites"],
                tob=rec["turn_of_breach"],
                layer_idx=args.layer_idx,
            )
        except torch.cuda.OutOfMemoryError as e:
            # Per-record OOM is recoverable: the next record may have a
            # smaller prompt. Other exceptions (config bugs, missing keys,
            # tokenizer issues) must propagate so we notice them.
            print(f"  [skip:OOM] {rec['source']}: {e}")
            torch.cuda.empty_cache()
            continue
        except (KeyError, IndexError) as e:
            # Malformed input record (missing 'rewrites' or 'conversation' fields).
            # Skip and log; the rewriter step is responsible for producing
            # well-formed records.
            print(f"  [skip:malformed] {rec['source']}: {e}")
            continue

        out = {
            "source": rec["source"],
            "experiment": rec.get("experiment"),
            "round": rec.get("round"),
            "index": rec.get("index"),
            "goal": rec.get("goal"),
            "behavior": rec.get("behavior"),
            "category": rec.get("category"),
            "turn_of_breach": rec["turn_of_breach"],
            "original_conversation": rec["conversation"],
            "translated_conversation": translated,
            "rewrites": rec["rewrites"],
            "original_hs": torch.stack(orig_hs) if orig_hs else None,
            "translated_hs": torch.stack(trans_hs) if trans_hs else None,
        }
        out_records.append(out)
        # Save periodically for resumability.
        if len(out_records) % 10 == 0:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            torch.save(out_records, args.output)
            print(f"  [progress] {len(out_records)} replays saved to {args.output}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(out_records, args.output)
    print(f"\nFinal: {len(out_records)} records saved to {args.output}")
    unload_model(model, tok)


if __name__ == "__main__":
    main()
