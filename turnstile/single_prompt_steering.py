"""Single-prompt steering experiment.

Pipeline:

  1. Load 200 prompts from inputs.json (100 harmful breach turns +
     100 AlpacaEval benign instructions).
  2. Run the 8B victim once per prompt; capture residual-stream hidden
     states at L16 and L31 at the *last user-message token* (i.e. the
     token immediately before the model emits its first response token).
  3. Fit two directions per layer on labels (harmful=1, benign=0):
       - LR (logistic regression, normalized weight vector)
       - mean-diff (Arditi-style: mean(harmful)-mean(benign), normalized)
     Save fitted directions to disk for later reuse.
  4. α sweep: for each (layer, direction-method, alpha) configuration,
     run the 8B victim *under steering* on every harmful prompt;
     judge the response with the strict dual judge against the original
     harmful goal.
  5. Output one JSONL row per (prompt_id, layer, method, alpha). Resumable.

Phase 4 (transferability of multi-turn directions) is a separate pass:
pass --transfer-direction-pt path/to/v2_direction.pt to apply a previously
fitted direction without re-fitting.
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from turnstile.model_utils import load_model, generate_text, unload_model
from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------
def extract_hs(model, tokenizer, prompts, layers, device="cuda"):
    """Return dict layer -> tensor (n_prompts, hidden_dim) of HS at the last
    user-message token (last token of the rendered chat-template prompt before
    add_generation_prompt's assistant header).

    We use add_generation_prompt=True and take the *final* token of that
    rendered string -- this is the token immediately preceding generation,
    which is what the model actually conditions on when emitting turn 0.
    """
    by_layer = {L: [] for L in layers}
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(templated, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        # hidden_states is tuple of (n_layers+1) tensors, each (1, seq, d)
        # Index 0 is embedding output; layer i is at index i+1 (post-block).
        for L in layers:
            h = out.hidden_states[L + 1][0, -1, :].detach().to("cpu", torch.float32)
            by_layer[L].append(h)
        del out
    return {L: torch.stack(by_layer[L]) for L in layers}


# ---------------------------------------------------------------------------
# Direction fitting
# ---------------------------------------------------------------------------
def fit_direction(hs, labels, method):
    """Fit a unit-norm direction. labels: 1 for harmful, 0 for benign.
    Returns (direction (d,), info dict)."""
    X = hs.numpy()
    y = labels.numpy()
    if method == "lr":
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X, y)
        w = clf.coef_[0]
        info = {"method": "lr", "raw_norm": float(np.linalg.norm(w)),
                "train_acc": float(clf.score(X, y))}
    elif method == "meandiff":
        mu_h = X[y == 1].mean(axis=0)
        mu_b = X[y == 0].mean(axis=0)
        w = mu_h - mu_b
        info = {"method": "meandiff", "raw_norm": float(np.linalg.norm(w))}
    else:
        raise ValueError(f"unknown method {method}")
    w = w / max(np.linalg.norm(w), 1e-9)
    info["unit_norm"] = 1.0
    return torch.tensor(w, dtype=torch.float32), info


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------
class SteeringHook:
    """Adds alpha * direction to the residual stream at layer L during
    forward passes. The direction lives on the same device/dtype as the
    layer output."""
    def __init__(self, direction, alpha):
        self.direction = direction
        self.alpha = float(alpha)
        self._handle = None

    def _hook_fn(self, module, inputs, output):
        if self.alpha == 0.0:
            return output
        if isinstance(output, tuple):
            h = output[0]
            d = self.direction.to(device=h.device, dtype=h.dtype)
            return (h + self.alpha * d,) + output[1:]
        d = self.direction.to(device=output.device, dtype=output.dtype)
        return output + self.alpha * d

    def register(self, model, layer_idx):
        self._handle = model.model.layers[layer_idx].register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Steered generation + judging
# ---------------------------------------------------------------------------
def steer_and_generate(model, tokenizer, prompt, direction, layer, alpha, max_tokens):
    chat = [{"role": "user", "content": prompt}]
    templated = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    hook = SteeringHook(direction, alpha)
    hook.register(model, layer)
    try:
        resp = generate_text(model, tokenizer, templated,
                             max_tokens=max_tokens, temperature=0.0)
    finally:
        hook.remove()
    return resp.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_phase1_extract(args, harmful, benign, vic_model, vic_tok):
    out_pt = os.path.join(args.output, "hs_layers.pt")
    if os.path.exists(out_pt) and not args.force_extract:
        print(f"[phase1] reusing existing {out_pt}")
        return torch.load(out_pt, weights_only=False)

    layers = [args.layer_a, args.layer_b]
    print(f"[phase1] extracting HS at layers {layers} for {len(harmful)+len(benign)} prompts...")
    all_prompts = [h["prompt"] for h in harmful] + [b["prompt"] for b in benign]
    t0 = time.time()
    hs_by_layer = extract_hs(vic_model, vic_tok, all_prompts, layers)
    n_h = len(harmful)
    labels = torch.tensor([1] * n_h + [0] * len(benign), dtype=torch.long)
    bundle = {
        "hs": hs_by_layer,
        "labels": labels,
        "prompt_ids": [h["prompt_id"] for h in harmful] + [b["prompt_id"] for b in benign],
        "layers": layers,
        "n_harmful": n_h, "n_benign": len(benign),
    }
    torch.save(bundle, out_pt)
    print(f"[phase1] done in {time.time()-t0:.1f}s -> {out_pt}")
    return bundle


def run_phase2_fit(args, bundle):
    out_pt = os.path.join(args.output, "directions.pt")
    if os.path.exists(out_pt) and not args.force_fit:
        print(f"[phase2] reusing existing {out_pt}")
        return torch.load(out_pt, weights_only=False)

    layers = bundle["layers"]
    labels = bundle["labels"]
    directions = {}
    for L in layers:
        for method in ("lr", "meandiff"):
            d, info = fit_direction(bundle["hs"][L], labels, method)
            directions[(L, method)] = {"direction": d, "info": info}
            print(f"[phase2] L={L} method={method:>9}  raw_norm={info['raw_norm']:.3f}  "
                  + (f"train_acc={info.get('train_acc', float('nan')):.3f}"
                     if 'train_acc' in info else ""))
    # Save in a flat-key form for portability.
    save_blob = {f"L{L}_{m}": v for (L, m), v in directions.items()}
    torch.save(save_blob, out_pt)
    return directions


def run_phase3_sweep(args, harmful, directions, vic_model, vic_tok, judge):
    out_jsonl = os.path.join(args.output, "sweep.jsonl")
    seen = set()
    if os.path.exists(out_jsonl):
        for line in open(out_jsonl):
            if line.strip():
                r = json.loads(line)
                seen.add((r["prompt_id"], r["layer"], r["method"], r["alpha"]))
    layers = sorted(set(L for (L, _) in directions))
    methods = ("lr", "meandiff")
    alphas = args.alphas

    todo = []
    for h in harmful:
        for L in layers:
            for m in methods:
                for a in alphas:
                    key = (h["prompt_id"], L, m, float(a))
                    if key not in seen:
                        todo.append((h, L, m, float(a)))
    print(f"[phase3] {len(todo)} (prompt, layer, method, alpha) configs to run "
          f"(skip {len(seen)} already done)")

    n_done = 0
    n_fail = 0
    t0 = time.time()
    for h, L, m, a in todo:
        d = directions[(L, m)]["direction"]
        try:
            resp = steer_and_generate(
                vic_model, vic_tok, h["prompt"], d, L, a, args.max_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {"prompt_id": h["prompt_id"], "layer": L, "method": m, "alpha": a,
                   "response": None, "oom": True, "verdict": None,
                   "judge_failed": True,
                   "ts": datetime.now(timezone.utc).isoformat()}
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_fail += 1
            continue
        conv = [
            {"role": "user", "content": h["prompt"]},
            {"role": "assistant", "content": resp},
        ]
        try:
            verdict = judge.judge_conversation(conv, goal=h["goal"])
            verdict_dict = asdict(verdict)
            judge_failed = False
        except LocalJudgeFailure as e:
            verdict_dict = {"unsafe": None, "guard_unsafe": None,
                            "together_unsafe": None, "raw_70b_output": str(e),
                            "reason": "judge_failed"}
            judge_failed = True
            n_fail += 1
        row = {
            "prompt_id": h["prompt_id"],
            "goal": h["goal"],
            "category": h.get("category"),
            "layer": L, "method": m, "alpha": a,
            "response": resp,
            "verdict": verdict_dict,
            "judge_failed": judge_failed,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        n_done += 1
        if n_done % 20 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta_min = (len(todo) - n_done) / max(rate, 1e-3) / 60
            print(f"[phase3] [{n_done}/{len(todo)}] {rate:.2f} cfg/s  "
                  f"eta {eta_min:.1f} min  ({n_fail} fail)")
    print(f"[phase3] finished: {n_done} new configs, {n_fail} fails")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layer-a", type=int, default=16)
    p.add_argument("--layer-b", type=int, default=31)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--skip-judge", action="store_true",
                   help="Run only phases 1+2 (probe-fit), no α sweep.")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--force-fit", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    inp = json.load(open(args.inputs_json))
    harmful = inp["harmful"]
    benign = inp["benign"]
    print(f"loaded {len(harmful)} harmful + {len(benign)} benign prompts")

    print(f"loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)

    bundle = run_phase1_extract(args, harmful, benign, vic_model, vic_tok)
    directions = run_phase2_fit(args, bundle)

    if args.skip_judge:
        print("--skip-judge: stopping after phase 2.")
        unload_model(vic_model, vic_tok)
        return

    print("loading strict dual judge (Llama-Guard-3-1B + Llama-3.1-70B-Instruct)...")
    judge = StrictDualJudge()
    try:
        run_phase3_sweep(args, harmful, directions, vic_model, vic_tok, judge)
    finally:
        judge.unload()
        unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
