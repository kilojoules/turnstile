"""Layer-sweep steering orchestrator.

For each (prompt, method, layer, alpha) cell, generate a steered response.
One model load; loops over all cells. Resumable.

Inputs:
  --directions-dir : directory with v_{method}_L{layer}.pt for all (method, layer)
  --metadata : JSON with per_layer.h_norm_median[layer]
  --prompts : JSON list (will subset to first N wins)
  --output : JSONL (resumable)
  --layers : comma-sep list
  --methods : comma-sep list (one of: lr_comp, md_comp, lr_harm, md_harm, random)
  --alphas : comma-sep list (excluding 0; baseline run once per prompt)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch

from turnstile.model_utils import load_model, generate_text, unload_model


MAX_NEW_TOKENS = 350


class SingleDirectionHook:
    def __init__(self, direction, alpha, h_norm):
        self.v = direction
        self.alpha = float(alpha)
        self.h_norm = float(h_norm)
        self._delta = None
        self._handle = None

    def _ensure_delta(self, dtype, device):
        if (self._delta is None or self._delta.dtype != dtype
                or self._delta.device != device):
            self._delta = (self.alpha * self.h_norm * self.v).to(
                dtype=dtype, device=device)

    def _hook_fn(self, module, inputs, output):
        if self.alpha == 0.0:
            return output
        if isinstance(output, tuple):
            h = output[0]
            self._ensure_delta(h.dtype, h.device)
            return (h + self._delta,) + output[1:]
        else:
            self._ensure_delta(output.dtype, output.device)
            return output + self._delta

    def register(self, model, layer_idx):
        self._handle = model.model.layers[layer_idx].register_forward_hook(
            self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def steer_and_generate(model, tokenizer, prompt, v, alpha, h_norm, layer,
                       max_tokens=MAX_NEW_TOKENS):
    chat = prompt["chat_history"] + \
           [{"role": "user", "content": prompt["last_user_message"]}]
    templated = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True)
    if alpha == 0.0 or v is None:
        return generate_text(model, tokenizer, templated,
                             max_tokens=max_tokens, temperature=0.0)
    hook = SingleDirectionHook(v, alpha, h_norm)
    hook.register(model, layer)
    try:
        resp = generate_text(model, tokenizer, templated,
                             max_tokens=max_tokens, temperature=0.0)
    finally:
        hook.remove()
    return resp


def coherence(text):
    toks = text.split()
    if not toks:
        return {"max_repeat": 0, "token_unique_ratio": 0.0,
                "n_tokens": 0, "n_chars": len(text)}
    max_rep = 1; cur = 1; prev = toks[0]
    for t in toks[1:]:
        if t == prev: cur += 1; max_rep = max(max_rep, cur)
        else: cur = 1
        prev = t
    return {"max_repeat": max_rep,
            "token_unique_ratio": len(set(toks))/len(toks),
            "n_tokens": len(toks), "n_chars": len(text)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directions-dir", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", default="0,4,8,12,16,20,24,28,31")
    p.add_argument("--methods", default="lr_comp,md_comp,lr_harm,md_harm,random")
    p.add_argument("--alphas", default="-1.0,-0.5,0.5,1.0")
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--victim-model",
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--prompt-type-filter", default="win",
                   help="Only run prompts of this type")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    methods = args.methods.split(",")
    alphas = [float(x) for x in args.alphas.split(",")]
    meta = json.load(open(args.metadata))
    prompts_all = json.load(open(args.prompts))
    prompts = [p for p in prompts_all
               if p.get("prompt_type") == args.prompt_type_filter
               ][:args.n_prompts]
    print(f"Prompts: {len(prompts)} ({args.prompt_type_filter})", flush=True)
    print(f"Layers: {layers}", flush=True)
    print(f"Methods: {methods}", flush=True)
    print(f"Alphas: {alphas}", flush=True)
    print(f"Total cells per prompt: 1 baseline + "
          f"{len(layers)*len(methods)*len(alphas)} = "
          f"{1 + len(layers)*len(methods)*len(alphas)}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    vecs = {}
    for L in layers:
        for m in methods:
            path = f"{args.directions_dir}/v_{m}_L{L}.pt"
            v = torch.load(path, weights_only=False)
            vecs[(m, L)] = v.float()
    print(f"Loaded {len(vecs)} direction vectors", flush=True)

    seen = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                seen.add((r["prompt_id"], r.get("method", "baseline"),
                          int(r.get("layer", -1)), round(float(r["alpha"]), 4)))
    print(f"Resumed: {len(seen)} cells already done", flush=True)

    print(f"\nLoading {args.victim_model}...", flush=True)
    model, tokenizer = load_model(args.victim_model)

    n_total_per_prompt = 1 + len(layers) * len(methods) * len(alphas)
    n_total = n_total_per_prompt * len(prompts)
    n_done = 0
    t_start = time.time()
    with open(args.output, "a") as fout:
        for prompt in prompts:
            pid = prompt["prompt_id"]
            # Baseline (one per prompt)
            key = (pid, "baseline", -1, 0.0)
            if key not in seen:
                t0 = time.time()
                resp = steer_and_generate(model, tokenizer, prompt,
                                          None, 0.0, 0.0, 0,
                                          args.max_tokens)
                row = {
                    "prompt_id": pid,
                    "behavior": prompt.get("behavior"),
                    "category": prompt.get("category"),
                    "goal": prompt.get("goal"),
                    "prompt_type": prompt.get("prompt_type"),
                    "method": "baseline",
                    "layer": -1,
                    "alpha": 0.0,
                    "h_norm": 0.0,
                    "response": resp,
                    "coherence": coherence(resp),
                    "elapsed": time.time() - t0,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                fout.write(json.dumps(row) + "\n"); fout.flush()
                n_done += 1
            else:
                n_done += 1

            for m in methods:
                for L in layers:
                    h_norm = float(meta["per_layer"][f"L{L}"]["h_norm_median"])
                    v = vecs[(m, L)]
                    for alpha in alphas:
                        key = (pid, m, L, round(alpha, 4))
                        if key in seen:
                            n_done += 1; continue
                        t0 = time.time()
                        resp = steer_and_generate(
                            model, tokenizer, prompt, v, alpha, h_norm, L,
                            args.max_tokens)
                        row = {
                            "prompt_id": pid,
                            "behavior": prompt.get("behavior"),
                            "category": prompt.get("category"),
                            "goal": prompt.get("goal"),
                            "prompt_type": prompt.get("prompt_type"),
                            "method": m,
                            "layer": L,
                            "alpha": alpha,
                            "h_norm": h_norm,
                            "response": resp,
                            "coherence": coherence(resp),
                            "elapsed": time.time() - t0,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                        fout.write(json.dumps(row) + "\n"); fout.flush()
                        n_done += 1
                        elapsed = time.time() - t_start
                        rate = n_done / max(1e-6, elapsed)
                        eta = (n_total - n_done) / max(1e-6, rate)
                        if n_done % 20 == 0 or n_done == n_total:
                            print(f"  [{n_done}/{n_total}] {m:<10} L{L:<2} "
                                  f"α={alpha:+.2f} pid={pid[:24]:<24}  "
                                  f"{time.time()-t0:.1f}s  ETA {eta/60:.1f}min",
                                  flush=True)

    unload_model(model, tokenizer)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
