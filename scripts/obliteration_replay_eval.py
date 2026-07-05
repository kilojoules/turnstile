"""Experiment 2: does obliterating the linear compliance direction reduce ASR?

The compliance_obliteration_v1 LoRA was trained (SFT + probe-AUC penalty) to
drive the L16 linear compliance signal to chance while preserving behavior on
the training trajectories. Its README lists the decisive followup as un-run:
"Eval the adapter on a held-out attack corpus: does ASR / Stage-B Likert shift?"

This runs that followup. Matched to how the adapter was trained (flat per-turn
examples with teacher-forced history), for each selected victim turn we:
  - take the true conversation history up to that turn,
  - regenerate the victim's response with the BASE model and with the
    OBLITERATED model (base + LoRA), identical context, greedy,
  - judge each regenerated response for compliance (+ harm).

ASR_base vs ASR_obliterated on identical attacker context answers whether the
scrubbed linear direction was causal for the behavior or merely epiphenomenal.

Output rows carry {prompt_id, method in {base, obliterated}, alpha=0.0, goal,
category, response, turn, source, round} for scripts/judge_postresponse_sweep.py.
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 300


def load_base(device):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map=device, torch_dtype=torch.bfloat16)
    model.eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, history):
    ids = tok.apply_chat_template(history, return_tensors="pt",
                                  add_generation_prompt=True).to(model.device)
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def select_turns(convs, n, wins_only=True):
    """Pick the last victim turn of each (win) conversation with its history."""
    picked = []
    for c in convs:
        if wins_only and not c.get("unsafe"):
            continue
        conv = c.get("conversation", [])
        # index of last assistant turn
        last_a = max((i for i, t in enumerate(conv) if t["role"] == "assistant"), default=-1)
        if last_a <= 0:
            continue
        history = conv[:last_a]                       # ends on a user (attacker) msg
        if not history or history[-1]["role"] != "user":
            continue
        picked.append({
            "goal": c.get("goal", ""), "category": c.get("category"),
            "behavior": c.get("behavior"), "source": c.get("source"),
            "round": c.get("round"), "idx": c.get("idx"),
            "turn": last_a, "history": history,
        })
        if len(picked) >= n:
            break
    return picked


def run_model(model, tok, turns, method, fout, done):
    for i, t in enumerate(turns):
        key = (i, method)
        if key in done:
            continue
        resp = generate(model, tok, t["history"])
        fout.write(json.dumps({
            "prompt_id": i, "method": method, "alpha": 0.0,
            "goal": t["goal"], "category": t["category"], "behavior": t["behavior"],
            "source": t["source"], "round": t["round"], "turn": t["turn"],
            "response": resp,
        }) + "\n")
        fout.flush()
    print(f"  done method={method}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default="data/all_conversations_jbb.json")
    ap.add_argument("--adapter", default="experiments/compliance_obliteration_v1/adapter")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    convs = json.load(open(args.convs))
    turns = select_turns(convs, args.n)
    print(f"Selected {len(turns)} win-turns for replay", flush=True)

    done = set()
    if os.path.exists(args.output):
        for l in open(args.output):
            try:
                r = json.loads(l); done.add((r["prompt_id"], r["method"]))
            except Exception:
                pass
        print(f"Resuming: {len(done)} cells done", flush=True)

    print("Loading base victim...", flush=True)
    model, tok = load_base(device)

    with open(args.output, "a") as fout:
        run_model(model, tok, turns, "base", fout, done)
        print("Attaching obliteration LoRA...", flush=True)
        obl = PeftModel.from_pretrained(model, args.adapter)
        obl.eval()
        run_model(obl, tok, turns, "obliterated", fout, done)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
