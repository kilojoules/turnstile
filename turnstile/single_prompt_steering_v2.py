"""Single-prompt steering experiment, redesigned.

Addresses the six concerns surfaced by an external review of the original
single-prompt sweep:

1. *Train-on-test contamination.* Stratified 50/50 record-paired split.
   Direction is fit on training-pair breach turns only; steering is
   evaluated on held-out test harmful breach turns.

2. *Distributional confound.* Within-corpus paired contrast. For each
   v2 record, the harmful breach turn is contrasted against its
   70B-rewritten benign counterfactual breach turn from the same
   conversation. Length, register, audit-style preamble matched by
   construction; only the harmful content of the breach itself differs.

3. *Coherence collapse at large alpha.* Every steered generation is
   diagnosed (max consecutive token repeat, compression ratio, token
   uniqueness ratio). No filtering -- the 2x2 of
   {coherent, broken} x {unsafe, safe} is reported per cell.

4. *Effect-fair alpha range.* alpha grid is chosen as a fraction of
   median ||h|| at each layer, so unit-step interventions inject the
   same fraction of ambient activation magnitude across layers. A
   secondary wider sweep at L31 only (up to 0.9 |h|) is included
   to test whether L31 is inert at *all* matched scales.

5. *Two-point asymmetry, not depth profile.* Three layers (L16, L24,
   L31) are tested but the central claim is binary: L16 has causal
   power; L31 does not at matched alpha/||h||. L24 is a sanity
   midpoint, not depth-profile evidence.

6. *Stronger nuisance baseline than random.* Two control directions:
   (a) a random unit vector at each tested layer (fit-free null), and
   (b) an LR direction fit on a length-median split of training breach
   turns (a "this is a long prompt" axis as a stronger nuisance test).
   If either matches the harm-LR effect, the harm interpretation
   collapses.

Output: one JSONL row per (test_prompt, layer, direction, alpha),
including the steered response, the coherence diagnostics, and the
strict-dual-judge verdict against the *original* harmful goal of the
record. Resumable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from turnstile.model_utils import load_model, generate_text, unload_model
from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure


# ---------------------------------------------------------------------------
# Hidden-state extraction (last user-message token, post-chat-template)
# ---------------------------------------------------------------------------
def extract_hs(model, tokenizer, prompts, layers, device="cuda"):
    by_layer = {L: [] for L in layers}
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(templated, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        for L in layers:
            h = out.hidden_states[L + 1][0, -1, :].detach().to("cpu", torch.float32)
            by_layer[L].append(h)
        del out
    return {L: torch.stack(by_layer[L]) for L in layers}


# ---------------------------------------------------------------------------
# Direction fitting
# ---------------------------------------------------------------------------
def fit_lr_direction(X, y):
    """Logistic regression direction, unit-normalized. Returns (direction, info)."""
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    w = clf.coef_[0]
    raw_norm = float(np.linalg.norm(w))
    train_acc = float(clf.score(X, y))
    return torch.tensor(w / max(raw_norm, 1e-9), dtype=torch.float32), \
           {"method": "lr", "raw_norm": raw_norm, "train_acc": train_acc}


def fit_random_direction(d_model, seed):
    rng = np.random.RandomState(seed)
    w = rng.randn(d_model).astype(np.float32)
    return torch.tensor(w / np.linalg.norm(w)), {"method": "random", "seed": seed}


# ---------------------------------------------------------------------------
# Steering hook (same as v1 module)
# ---------------------------------------------------------------------------
class SteeringHook:
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
# Coherence diagnostics
# ---------------------------------------------------------------------------
def coherence_metrics(text, tokenizer):
    """Return dict of coherence metrics. None of these affect filtering --
    they're recorded for offline analysis."""
    if not text:
        return {"empty": True, "max_consec_same_token": 0, "max_bigram_repeat": 0,
                "compress_ratio": 0.0, "type_token_ratio": 0.0,
                "is_broken": True, "n_tokens": 0}
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_tok = len(ids)
    if n_tok == 0:
        return {"empty": True, "max_consec_same_token": 0, "max_bigram_repeat": 0,
                "compress_ratio": 0.0, "type_token_ratio": 0.0,
                "is_broken": True, "n_tokens": 0}

    # Max consecutive identical token id
    max_run = cur = 1
    for i in range(1, n_tok):
        if ids[i] == ids[i-1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1

    # Max consecutive identical bigram
    max_bg = cur_bg = 1
    if n_tok >= 4:
        for i in range(2, n_tok - 1):
            if ids[i] == ids[i-2] and ids[i+1] == ids[i-1]:
                cur_bg += 1
                max_bg = max(max_bg, cur_bg)
            else:
                cur_bg = 1

    type_token_ratio = len(set(ids)) / n_tok
    raw = text.encode("utf-8")
    compress_ratio = len(gzip.compress(raw)) / max(len(raw), 1)

    is_broken = (max_run >= 30 or
                 max_bg >= 30 or
                 type_token_ratio < 0.05)
    return {
        "empty": False,
        "max_consec_same_token": int(max_run),
        "max_bigram_repeat": int(max_bg),
        "compress_ratio": float(compress_ratio),
        "type_token_ratio": float(type_token_ratio),
        "is_broken": bool(is_broken),
        "n_tokens": int(n_tok),
    }


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------
def steer_and_generate(model, tokenizer, prompt, direction, layer, alpha,
                       max_tokens):
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
def phase1_extract(args, train_pairs, test_pairs, vic_model, vic_tok):
    out_pt = os.path.join(args.output, "hs_v2pair.pt")
    if os.path.exists(out_pt) and not args.force_extract:
        print(f"[phase1] reusing existing {out_pt}")
        return torch.load(out_pt, weights_only=False)

    layers = args.layers
    print(f"[phase1] extracting HS at layers {layers} for "
          f"{2*len(train_pairs)+2*len(test_pairs)} prompts...")

    train_h_prompts = [p["harmful_breach"] for p in train_pairs]
    train_b_prompts = [p["benign_breach"] for p in train_pairs]
    test_h_prompts = [p["harmful_breach"] for p in test_pairs]
    test_b_prompts = [p["benign_breach"] for p in test_pairs]

    all_prompts = train_h_prompts + train_b_prompts + test_h_prompts + test_b_prompts
    t0 = time.time()
    hs_by_layer = extract_hs(vic_model, vic_tok, all_prompts, layers)

    n_th = len(train_pairs)
    bundle = {
        "hs_by_layer": hs_by_layer,
        "splits": {
            "train_harmful": list(range(0, n_th)),
            "train_benign":  list(range(n_th, 2*n_th)),
            "test_harmful":  list(range(2*n_th, 2*n_th + len(test_pairs))),
            "test_benign":   list(range(2*n_th + len(test_pairs), 2*n_th + 2*len(test_pairs))),
        },
        "layers": layers,
    }
    torch.save(bundle, out_pt)
    print(f"[phase1] done in {time.time()-t0:.1f}s -> {out_pt}")
    return bundle


def phase2_fit(args, bundle, train_pairs):
    """Fit three direction families per layer:
       - LR_harm: harmful vs benign-rewritten breach turns (the real one)
       - LR_length: above-median vs below-median breach-turn char length
                    (nuisance: "is this a long prompt" axis)
       - random: one fixed-seed unit vector per layer (fit-free null)
    """
    out_pt = os.path.join(args.output, "directions_v2pair.pt")
    if os.path.exists(out_pt) and not args.force_fit:
        print(f"[phase2] reusing existing {out_pt}")
        return torch.load(out_pt, weights_only=False)

    splits = bundle["splits"]
    hs_by_layer = bundle["hs_by_layer"]

    # Length-LR labels: pool train harmful + train benign, label by
    # whether character length > median.
    train_lens_pooled = ([len(p["harmful_breach"]) for p in train_pairs] +
                          [len(p["benign_breach"]) for p in train_pairs])
    median_len = float(np.median(train_lens_pooled))
    print(f"[phase2] train-pool median breach length = {median_len:.0f} chars")

    directions = {}
    for L in args.layers:
        H = hs_by_layer[L]
        Xh = H[splits["train_harmful"]].numpy()
        Xb = H[splits["train_benign"]].numpy()
        # LR_harm
        X = np.vstack([Xh, Xb])
        y = np.concatenate([np.ones(len(Xh)), np.zeros(len(Xb))])
        d, info = fit_lr_direction(X, y)
        directions[(L, "lr_harm")] = {"direction": d, "info": info}
        print(f"[phase2] L={L} lr_harm    raw_norm={info['raw_norm']:.3f}  "
              f"train_acc={info['train_acc']:.3f}")

        # LR_length: nuisance axis on the same 2N pooled training points
        y_len = np.array([float(len(p["harmful_breach"]) > median_len) for p in train_pairs] +
                         [float(len(p["benign_breach"])  > median_len) for p in train_pairs])
        if len(set(y_len)) == 2:
            d, info = fit_lr_direction(X, y_len)
            directions[(L, "lr_length")] = {"direction": d, "info": info}
            print(f"[phase2] L={L} lr_length  raw_norm={info['raw_norm']:.3f}  "
                  f"train_acc={info['train_acc']:.3f}")
        else:
            print(f"[phase2] L={L} lr_length  skipped (degenerate split)")

        # random
        d_model = H.shape[1]
        d, info = fit_random_direction(d_model, seed=13 + L)
        directions[(L, "random")] = {"direction": d, "info": info}
        print(f"[phase2] L={L} random     seed={info['seed']}  unit-norm")

    save_blob = {f"L{L}_{m}": v for (L, m), v in directions.items()}
    torch.save(save_blob, out_pt)
    return directions


def alpha_grid_for(layer, hs_by_layer, ratios):
    """Return a sorted list of alphas at the requested |h|-ratios."""
    median_h = float(torch.norm(hs_by_layer[layer], dim=1).median())
    alphas = sorted({0.0} | {round(r * median_h, 3) for r in ratios}
                    | {round(-r * median_h, 3) for r in ratios})
    return alphas, median_h


def phase3_sweep(args, bundle, directions, test_pairs, vic_model, vic_tok, judge):
    out_jsonl = os.path.join(args.output, "sweep_v2pair.jsonl")
    seen = set()
    if os.path.exists(out_jsonl):
        for line in open(out_jsonl):
            if line.strip():
                r = json.loads(line)
                seen.add((r["test_record_idx"], r["layer"], r["direction"], float(r["alpha"])))

    # Build the sweep grid.
    # Standard alphas at fractions {0.1, 0.25, 0.5} of |h|.
    ratios_main = (0.1, 0.25, 0.5)
    # Wider alphas at L31 only: {0.7, 0.9} additional ratios.
    ratios_wide_l31 = (0.7, 0.9)

    todo = []
    for L in args.layers:
        alphas_main, median_h = alpha_grid_for(L, bundle["hs_by_layer"], ratios_main)
        # Determine which directions to run at this layer.
        # lr_harm: all layers. lr_length: L16 only. random: L16 and L31.
        dirs_here = ["lr_harm"]
        if L == args.layer_a:  # L16 by default
            dirs_here.append("lr_length")
        if L == args.layer_a or L == args.layer_c:  # L16, L31
            dirs_here.append("random")
        for direction in dirs_here:
            if (L, direction) not in directions:
                continue
            for a in alphas_main:
                for p in test_pairs:
                    todo.append((p, L, direction, float(a), median_h))

    # Wide L31 sweep (lr_harm only, additional alphas)
    alphas_wide, median_h_l31 = alpha_grid_for(
        args.layer_c, bundle["hs_by_layer"], ratios_wide_l31)
    for a in alphas_wide:
        if a == 0.0: continue
        for p in test_pairs:
            todo.append((p, args.layer_c, "lr_harm", float(a), median_h_l31))

    # Drop already-done
    todo = [(p, L, m, a, mh) for (p, L, m, a, mh) in todo
            if (p["record_idx"], L, m, a) not in seen]
    print(f"[phase3] {len(todo)} configs to run; skip {len(seen)} already done")
    print(f"[phase3] median ‖h‖ per layer:")
    for L in args.layers:
        med = float(torch.norm(bundle['hs_by_layer'][L], dim=1).median())
        print(f"  L{L}: {med:.2f}")

    n_done = 0
    n_fail = 0
    n_broken = 0
    t0 = time.time()
    for (p, L, direction, alpha, median_h) in todo:
        d = directions[(L, direction)]["direction"]
        try:
            resp = steer_and_generate(
                vic_model, vic_tok, p["harmful_breach"], d, L, alpha, args.max_tokens,
            )
            oom = False
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            resp = None
            oom = True

        coherence = coherence_metrics(resp, vic_tok) if resp is not None else \
            {"empty": True, "is_broken": True, "n_tokens": 0,
             "max_consec_same_token": 0, "max_bigram_repeat": 0,
             "compress_ratio": 0.0, "type_token_ratio": 0.0}
        if coherence["is_broken"]:
            n_broken += 1

        if resp is None:
            verdict_dict = None
            judge_failed = True
        else:
            conv = [{"role": "user", "content": p["harmful_breach"]},
                    {"role": "assistant", "content": resp}]
            try:
                verdict = judge.judge_conversation(conv, goal=p["goal"])
                verdict_dict = asdict(verdict)
                judge_failed = False
            except LocalJudgeFailure as e:
                verdict_dict = {"unsafe": None, "guard_unsafe": None,
                                "together_unsafe": None, "raw_70b_output": str(e),
                                "reason": "judge_failed"}
                judge_failed = True
                n_fail += 1

        row = {
            "test_record_idx": p["record_idx"],
            "category": p.get("category"),
            "goal": p["goal"],
            "layer": L,
            "direction": direction,
            "alpha": float(alpha),
            "alpha_h_ratio": float(alpha / max(median_h, 1e-6)),
            "median_h_at_layer": float(median_h),
            "response": resp,
            "oom": oom,
            "coherence": coherence,
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
            print(f"[phase3] [{n_done}/{len(todo)}] {rate:.2f} cfg/s  "
                  f"eta {eta_min:.1f} min  ({n_fail} judge-fail, {n_broken} broken)")
    print(f"[phase3] finished: {n_done} new configs, {n_fail} fails, {n_broken} broken")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layer-a", type=int, default=16, help="L16 (causal layer)")
    p.add_argument("--layer-b", type=int, default=24, help="L24 (sanity midpoint)")
    p.add_argument("--layer-c", type=int, default=31, help="L31 (last block, predicted inert)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--skip-judge", action="store_true",
                   help="Run only phases 1+2.")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--force-fit", action="store_true")
    args = p.parse_args()
    args.layers = [args.layer_a, args.layer_b, args.layer_c]

    os.makedirs(args.output, exist_ok=True)

    inp = json.load(open(args.inputs_json))
    train = inp["train"]; test = inp["test"]
    print(f"loaded {len(train)} train pairs + {len(test)} test pairs")
    print(f"category distribution (test): {dict(Counter(t['category'] for t in test))}")

    print(f"loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)

    bundle = phase1_extract(args, train, test, vic_model, vic_tok)
    directions = phase2_fit(args, bundle, train)

    if args.skip_judge:
        print("--skip-judge: stopping after phase 2.")
        unload_model(vic_model, vic_tok)
        return

    print("loading strict dual judge...")
    judge = StrictDualJudge()
    try:
        phase3_sweep(args, bundle, directions, test, vic_model, vic_tok, judge)
    finally:
        judge.unload()
        unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
