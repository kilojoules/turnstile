"""H1: Arditi-style directional ablation across all layers.

Project out a candidate direction from the residual stream output of
*every* transformer block during generation. Run on the held-out test
set used by single_prompt_steering_v2 (103 harmful breach turns).

Hypothesis: if any of (lr_compliance, lr_harm, lr_turn_pos, random,
balanced_meandiff_compliance) ablation amplifies ASR above the
73% baseline, we have a publishable amplifier and our "no amplification
handle" claim flips.

Output: one JSONL row per (prompt, direction). No alpha sweep — ablation
is binary (ablated or not).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

import torch

from turnstile.model_utils import load_model, generate_text, unload_model
from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure
from turnstile.single_prompt_steering_v2 import coherence_metrics


class AblationHook:
    """Project out the direction component from the residual stream output
    at every transformer block. Direction must be unit-normalized.

       x' = x - (x · d) * d
    """
    def __init__(self, direction):
        self.direction = direction
        self._handles = []

    def _hook_fn(self, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            d = self.direction.to(device=h.device, dtype=h.dtype)
            proj = (h * d).sum(dim=-1, keepdim=True)  # (..., 1)
            return (h - proj * d,) + output[1:]
        d = self.direction.to(device=output.device, dtype=output.dtype)
        proj = (output * d).sum(dim=-1, keepdim=True)
        return output - proj * d

    def register_all_layers(self, model):
        for layer in model.model.layers:
            self._handles.append(layer.register_forward_hook(self._hook_fn))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def ablate_and_generate(model, tokenizer, prompt, direction, max_tokens):
    chat = [{"role": "user", "content": prompt}]
    templated = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    hook = AblationHook(direction)
    hook.register_all_layers(model)
    try:
        resp = generate_text(model, tokenizer, templated,
                             max_tokens=max_tokens, temperature=0.0)
    finally:
        hook.remove()
    return resp.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-json", required=True,
                   help="Path with 'test' field of {harmful_breach, goal, ...}.")
    p.add_argument("--directions-json", required=True,
                   help="JSON listing {label: pt_path:key, ...} of directions to test.")
    p.add_argument("--output", required=True)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_jsonl = os.path.join(args.output, "ablation.jsonl")

    inp = json.load(open(args.inputs_json))
    test = inp["test"]
    print(f"loaded {len(test)} test prompts")

    dirspec = json.load(open(args.directions_json))
    # Pre-load each direction
    directions = {}
    for label, spec in dirspec.items():
        path, key = spec.split(":")
        blob = torch.load(path, weights_only=False)
        directions[label] = blob[key]["direction"]
        print(f"  {label}: shape={tuple(directions[label].shape)} (loaded from {os.path.basename(path)}:{key})")

    # Add a NULL baseline (no ablation) — equivalent to a zero-vector hook
    # but easier to skip entirely.
    print(f"  null_baseline: no hook")

    seen = set()
    if os.path.exists(out_jsonl):
        for line in open(out_jsonl):
            if line.strip():
                r = json.loads(line)
                seen.add((r["test_record_idx"], r["direction_label"]))

    todo = []
    for label in list(directions.keys()) + ["null_baseline"]:
        for p in test:
            if (p["record_idx"], label) not in seen:
                todo.append((p, label))
    print(f"{len(todo)} configs to run; skip {len(seen)}")

    print(f"loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)
    print("loading strict dual judge...")
    judge = StrictDualJudge()

    n_done = 0; n_fail = 0; n_broken = 0
    t0 = time.time()
    try:
        for (p, label) in todo:
            if label == "null_baseline":
                # Plain generation, no hook
                chat = [{"role": "user", "content": p["harmful_breach"]}]
                templated = vic_tok.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True)
                try:
                    resp = generate_text(vic_model, vic_tok, templated,
                                         max_tokens=args.max_tokens, temperature=0.0).strip()
                    oom = False
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); resp = None; oom = True
            else:
                try:
                    resp = ablate_and_generate(
                        vic_model, vic_tok, p["harmful_breach"],
                        directions[label], args.max_tokens,
                    )
                    oom = False
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); resp = None; oom = True
            coh = coherence_metrics(resp, vic_tok)
            if coh.get("is_broken"): n_broken += 1
            if resp is None:
                verdict_dict = None; judge_failed = True
            else:
                conv = [{"role": "user", "content": p["harmful_breach"]},
                        {"role": "assistant", "content": resp}]
                try:
                    verdict = judge.judge_conversation(conv, goal=p["goal"])
                    verdict_dict = asdict(verdict); judge_failed = False
                except LocalJudgeFailure as e:
                    verdict_dict = {"unsafe": None, "guard_unsafe": None,
                                    "together_unsafe": None, "raw_70b_output": str(e),
                                    "reason": "judge_failed"}
                    judge_failed = True; n_fail += 1
            row = {
                "test_record_idx": int(p["record_idx"]),
                "category": p.get("category"),
                "goal": p["goal"],
                "direction_label": label,
                "response": resp,
                "oom": oom,
                "coherence": coh,
                "verdict": verdict_dict,
                "judge_failed": judge_failed,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta_min = (len(todo) - n_done) / max(rate, 1e-3) / 60
                print(f"[ablation] [{n_done}/{len(todo)}] {rate:.2f} cfg/s  "
                      f"eta {eta_min:.1f} min  ({n_fail} fail, {n_broken} broken)")
        print(f"[ablation] finished: {n_done}, fails={n_fail}, broken={n_broken}")
    finally:
        judge.unload()
        unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
