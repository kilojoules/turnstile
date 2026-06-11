"""Two-axis steering experiment: compliance probe direction + harm probe
direction, both at L16.

Hooks into Llama-3.1-8B-Instruct at L16 and adds
   alpha_c * h_norm * v_comp + alpha_h * h_norm * v_harm
to the residual stream at every position during generation.

Usage:
  # pilot: 5 prompts, 5 single-direction conditions
  python -m turnstile.two_axis_steering \\
      --mode pilot \\
      --directions /workspace/turnstile/experiments/steering_v3 \\
      --prompts /workspace/turnstile/data/steering_prompts.json \\
      --output /workspace/turnstile/experiments/steering_v3/pilot.jsonl

  # main: 50 prompts, full 3x3
  python -m turnstile.two_axis_steering \\
      --mode main \\
      --alpha-comp 0.5 --alpha-harm 0.5 \\
      --directions /workspace/turnstile/experiments/steering_v3 \\
      --prompts /workspace/turnstile/data/steering_prompts.json \\
      --output /workspace/turnstile/experiments/steering_v3/main.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch

from turnstile.model_utils import load_model, generate_text, unload_model


LAYER = 16
MAX_NEW_TOKENS = 350


class TwoAxisSteeringHook:
    """Adds (alpha_c * h_norm * v_c) + (alpha_h * h_norm * v_h) to residual."""

    def __init__(self, v_comp, alpha_c, v_harm, alpha_h, h_norm):
        # all should be torch.float32 unit vectors
        self.v_c = v_comp
        self.alpha_c = float(alpha_c)
        self.v_h = v_harm
        self.alpha_h = float(alpha_h)
        self.h_norm = float(h_norm)
        self._handle = None
        # precompute the per-position shift (constant across positions)
        self._delta = None

    def _ensure_delta(self, dtype, device):
        if self._delta is None or self._delta.dtype != dtype or \
                self._delta.device != device:
            delta = (self.alpha_c * self.h_norm * self.v_c
                     + self.alpha_h * self.h_norm * self.v_h)
            self._delta = delta.to(dtype=dtype, device=device)

    def _hook_fn(self, module, inputs, output):
        # output may be a tuple (hidden_states, ...) or a tensor
        if self.alpha_c == 0.0 and self.alpha_h == 0.0:
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


def calibrate_h_norm(model, tokenizer, prompts, layer, device="cuda"):
    """Measure ||h_L{layer}|| at the last token of each prompt; return median."""
    norms = []
    for p in prompts:
        templated = tokenizer.apply_chat_template(
            p["chat_history"] + [{"role": "user", "content": p["last_user_message"]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(templated, return_tensors="pt",
                           truncation=True, max_length=2048).to(device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer + 1][0, -1, :]
        norms.append(float(h.norm().item()))
        del out
    norms_t = torch.tensor(norms)
    return float(norms_t.median().item()), float(norms_t.mean().item()), \
           float(norms_t.std().item())


def steer_and_generate(model, tokenizer, prompt, v_comp, alpha_c,
                       v_harm, alpha_h, h_norm, layer, max_new_tokens):
    """Generate a single response with two-axis steering applied at `layer`."""
    chat = prompt["chat_history"] + \
           [{"role": "user", "content": prompt["last_user_message"]}]
    templated = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )

    hook = TwoAxisSteeringHook(v_comp, alpha_c, v_harm, alpha_h, h_norm)
    hook.register(model, layer)
    try:
        resp = generate_text(model, tokenizer, templated,
                             max_tokens=max_new_tokens, temperature=0.0)
    finally:
        hook.remove()
    return resp


def coherence_metrics(text):
    """Cheap diagnostics: max consecutive token repeat, char-uniqueness."""
    toks = text.split()
    if not toks:
        return {"max_repeat": 0, "token_unique_ratio": 0.0, "n_tokens": 0,
                "n_chars": len(text)}
    max_rep, cur, prev = 1, 1, toks[0]
    for t in toks[1:]:
        if t == prev:
            cur += 1; max_rep = max(max_rep, cur)
        else:
            cur = 1
        prev = t
    return {"max_repeat": max_rep,
            "token_unique_ratio": len(set(toks)) / len(toks),
            "n_tokens": len(toks),
            "n_chars": len(text)}


def load_directions(d_dir, layer):
    v_c = torch.load(f"{d_dir}/v_comp_L{layer}.pt", weights_only=False)
    v_h = torch.load(f"{d_dir}/v_harm_L{layer}.pt", weights_only=False)
    meta = json.load(open(f"{d_dir}/steering_directions_meta.json"))
    return v_c.float(), v_h.float(), meta


def load_seen_keys(path):
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                seen.add((r["prompt_id"], round(r["alpha_c"], 4),
                          round(r["alpha_h"], 4)))
    return seen


def run_grid(model, tokenizer, prompts, v_c, v_h, h_norm, alphas_c, alphas_h,
             output_path, max_new_tokens=MAX_NEW_TOKENS, layer=LAYER):
    """Run all (prompt, alpha_c, alpha_h) cells; resumable via output JSONL."""
    seen = load_seen_keys(output_path)
    n_total = len(prompts) * len(alphas_c) * len(alphas_h)
    n_done = 0
    n_skip = 0
    t_start = time.time()
    print(f"\nRunning {n_total} (prompt, alpha_c, alpha_h) cells "
          f"({len(prompts)} prompts × {len(alphas_c)*len(alphas_h)} alpha-cells)",
          flush=True)
    with open(output_path, "a") as fout:
        for prompt in prompts:
            pid = prompt["prompt_id"]
            for ac in alphas_c:
                for ah in alphas_h:
                    key = (pid, round(ac, 4), round(ah, 4))
                    if key in seen:
                        n_skip += 1; n_done += 1
                        continue
                    t0 = time.time()
                    resp = steer_and_generate(
                        model, tokenizer, prompt, v_c, ac, v_h, ah,
                        h_norm, layer, max_new_tokens)
                    coh = coherence_metrics(resp)
                    row = {
                        "prompt_id": pid,
                        "behavior": prompt.get("behavior"),
                        "category": prompt.get("category"),
                        "goal": prompt.get("goal"),
                        "original_unsafe": prompt.get("original_unsafe"),
                        "original_turn_of_breach": prompt.get(
                            "original_turn_of_breach"),
                        "alpha_c": ac, "alpha_h": ah,
                        "layer": layer, "h_norm": h_norm,
                        "response": resp,
                        "coherence": coh,
                        "elapsed_sec": time.time() - t0,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    n_done += 1
                    elapsed = time.time() - t_start
                    rate = n_done / max(1e-6, elapsed)
                    eta = (n_total - n_done) / max(1e-6, rate)
                    if n_done % 10 == 0 or n_done == n_total:
                        print(f"  [{n_done}/{n_total}] (skip={n_skip}) "
                              f"prompt={pid} a_c={ac:+.2f} a_h={ah:+.2f}  "
                              f"{time.time()-t0:.1f}s/gen  ETA {eta/60:.1f}min",
                              flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["pilot", "main", "sweep", "calibrate"],
                   required=True)
    p.add_argument("--directions", required=True,
                   help="Dir with v_comp_L16.pt, v_harm_L16.pt, meta.json")
    p.add_argument("--prompts", required=True,
                   help="JSON list of prompts (see data/steering_prompts.json)")
    p.add_argument("--output", required=True, help="JSONL output (resumable)")
    p.add_argument("--alpha-comp", type=float, default=0.5,
                   help="alpha magnitude for compliance axis (main mode)")
    p.add_argument("--alpha-harm", type=float, default=0.5,
                   help="alpha magnitude for harm axis (main mode)")
    p.add_argument("--alphas-comp", type=str, default=None,
                   help="comma-sep list of alpha_c values for sweep mode")
    p.add_argument("--alphas-harm", type=str, default=None,
                   help="comma-sep list of alpha_h values for sweep mode")
    p.add_argument("--n-prompts-pilot", type=int, default=5)
    p.add_argument("--n-prompts-main", type=int, default=50)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--h-norm", type=float, default=None,
                   help="Override empirical |h_L16| (default: use meta median)")
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prompts = json.load(open(args.prompts))
    print(f"Loaded {len(prompts)} prompts from {args.prompts}", flush=True)

    v_c, v_h, meta = load_directions(args.directions, args.layer)
    print(f"Loaded directions: cos(v_c, v_h) = {meta['cos_v_comp_v_harm']:+.4f}, "
          f"||h_L{args.layer}|| meta median = {meta['h_L16_norm_median']:.2f}",
          flush=True)

    print(f"\nLoading victim: {args.victim_model}", flush=True)
    model, tokenizer = load_model(args.victim_model)

    # Calibration
    if args.mode == "calibrate":
        med, mean, std = calibrate_h_norm(
            model, tokenizer, prompts[:20], args.layer)
        print(f"\nCalibration on {min(20, len(prompts))} prompts at L{args.layer}:")
        print(f"  median = {med:.3f}")
        print(f"  mean   = {mean:.3f}")
        print(f"  std    = {std:.3f}")
        unload_model(model, tokenizer)
        return

    # Decide h_norm
    if args.h_norm is not None:
        h_norm = args.h_norm
    else:
        h_norm = float(meta["h_L16_norm_median"])
    print(f"\nUsing ||h_L{args.layer}|| = {h_norm:.3f}", flush=True)

    if args.mode == "pilot":
        # 5 single-direction conditions: 0/0, +/0, -/0, 0/+, 0/-
        # over a small prompt subset
        prompts_sub = prompts[:args.n_prompts_pilot]
        alphas_c_set = sorted({-args.alpha_comp, 0.0, args.alpha_comp})
        alphas_h_set = sorted({-args.alpha_harm, 0.0, args.alpha_harm})
        # for pilot, just sweep each axis solo: (ac in {±a*}, ah=0) + (ac=0, ah in {±a*})
        # i.e., 5 cells: (+,0), (-,0), (0,+), (0,-), (0,0)
        cells = [(args.alpha_comp, 0.0), (-args.alpha_comp, 0.0),
                 (0.0, args.alpha_harm), (0.0, -args.alpha_harm),
                 (0.0, 0.0)]
        # do this as a degenerate "grid" — just iterate
        seen = load_seen_keys(args.output)
        with open(args.output, "a") as fout:
            for prompt in prompts_sub:
                for ac, ah in cells:
                    key = (prompt["prompt_id"], round(ac, 4), round(ah, 4))
                    if key in seen:
                        continue
                    t0 = time.time()
                    resp = steer_and_generate(
                        model, tokenizer, prompt, v_c, ac, v_h, ah,
                        h_norm, args.layer, args.max_new_tokens)
                    row = {
                        "prompt_id": prompt["prompt_id"],
                        "behavior": prompt.get("behavior"),
                        "category": prompt.get("category"),
                        "goal": prompt.get("goal"),
                        "alpha_c": ac, "alpha_h": ah,
                        "layer": args.layer, "h_norm": h_norm,
                        "response": resp,
                        "coherence": coherence_metrics(resp),
                        "elapsed_sec": time.time() - t0,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    print(f"  pilot prompt={prompt['prompt_id']} "
                          f"a_c={ac:+.2f} a_h={ah:+.2f}: "
                          f"{time.time()-t0:.1f}s, "
                          f"len={len(resp)}", flush=True)
    elif args.mode == "main":
        prompts_sub = prompts[:args.n_prompts_main]
        alphas_c = sorted({-args.alpha_comp, 0.0, args.alpha_comp})
        alphas_h = sorted({-args.alpha_harm, 0.0, args.alpha_harm})
        run_grid(model, tokenizer, prompts_sub, v_c, v_h, h_norm,
                 alphas_c, alphas_h, args.output,
                 max_new_tokens=args.max_new_tokens, layer=args.layer)
    elif args.mode == "sweep":
        # Single-axis sweeps: union of (a_c, 0) cells and (0, a_h) cells.
        # Baseline (0, 0) appears once.
        prompts_sub = prompts[:args.n_prompts_main]
        alphas_c = sorted({float(x) for x in args.alphas_comp.split(",")})
        alphas_h = sorted({float(x) for x in args.alphas_harm.split(",")})
        cells = set()
        for ac in alphas_c:
            cells.add((round(ac, 4), 0.0))
        for ah in alphas_h:
            cells.add((0.0, round(ah, 4)))
        cells = sorted(cells)
        print(f"Sweep: {len(cells)} cells × {len(prompts_sub)} prompts = "
              f"{len(cells)*len(prompts_sub)} generations", flush=True)
        seen = load_seen_keys(args.output)
        n_total = len(cells) * len(prompts_sub)
        n_done = 0
        t_start = time.time()
        with open(args.output, "a") as fout:
            for prompt in prompts_sub:
                for ac, ah in cells:
                    key = (prompt["prompt_id"], round(ac, 4), round(ah, 4))
                    n_done += 1
                    if key in seen:
                        continue
                    t0 = time.time()
                    resp = steer_and_generate(
                        model, tokenizer, prompt, v_c, ac, v_h, ah,
                        h_norm, args.layer, args.max_new_tokens)
                    row = {
                        "prompt_id": prompt["prompt_id"],
                        "behavior": prompt.get("behavior"),
                        "category": prompt.get("category"),
                        "goal": prompt.get("goal"),
                        "prompt_type": prompt.get("prompt_type"),
                        "original_unsafe": prompt.get("original_unsafe"),
                        "original_turn_of_breach": prompt.get(
                            "original_turn_of_breach"),
                        "alpha_c": ac, "alpha_h": ah,
                        "layer": args.layer, "h_norm": h_norm,
                        "response": resp,
                        "coherence": coherence_metrics(resp),
                        "elapsed_sec": time.time() - t0,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    fout.write(json.dumps(row) + "\n"); fout.flush()
                    elapsed = time.time() - t_start
                    rate = max(1, n_done) / max(1e-6, elapsed)
                    eta = (n_total - n_done) / max(1e-6, rate)
                    print(f"  [{n_done}/{n_total}] prompt={prompt['prompt_id']:<24} "
                          f"a_c={ac:+.2f} a_h={ah:+.2f}  "
                          f"{time.time()-t0:.1f}s  ETA {eta/60:.1f}min",
                          flush=True)

    unload_model(model, tokenizer)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
