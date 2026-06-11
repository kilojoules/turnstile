"""Causal steering ablation on the paired-obliteration dataset.

Tests whether the L16 compliance direction (the same direction whose linear
probe reaches AUC 0.751 in the obliteration paper) causally controls the
victim's compliance behavior at the breach turn.

Protocol:
  - 5-fold stratified CV by turn-of-breach. For fold k, fit a logistic-
    regression compliance direction at L16 on the training records' original
    conversations (label 0 = refuse turns t<t*, label 1 = comply turn t=t*),
    then apply steering on the held-out records only.

  - At the breach turn t* of each held-out record, replay through the victim
    on both arms (original + obliterated). At the assistant generation step,
    register a forward hook on transformer block 16 that adds alpha * v
    (where v is the unit-normalized probe coefficient) to the residual
    stream. Generate the assistant response (T=0).

  - Score every (record, arm, alpha) generation with the strict dual judge
    (Llama-Guard-3-1B + Llama-3.1-70B-Instruct, no silent fallback).

Sweep alpha in {-8, -4, -2, 0, +2, +4, +8} by default. The headline is the
compliance-rate change between alpha=-A and alpha=+A on the obliterated arm
(where baseline compliance is ~0%) and on the original arm (baseline ~100%).

Usage (on a GPU box):
  python -m turnstile.clamp_obliterated \\
      --replay-pt experiments/intent_obliteration_paired/replay_judged.pt \\
      --output experiments/clamp_obliterated_v1 \\
      --victim-model meta-llama/Llama-3.1-8B-Instruct \\
      --layer-idx 16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from turnstile.model_utils import (
    load_model, generate_text, unload_model, extract_hidden_states,
)
from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure


# ---------------------------------------------------------------------------
# Steering hook (matches turnstile/clamp_probe_direction.py)
# ---------------------------------------------------------------------------
class SteeringHook:
    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction.detach().clone()
        self.alpha = float(alpha)
        self._handle = None

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        d = self.direction.to(hs.device, hs.dtype)
        hs = hs + self.alpha * d
        if is_tuple:
            return (hs,) + output[1:]
        return hs

    def register(self, model, layer_idx: int):
        layer = model.model.layers[layer_idx]
        self._handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Compliance direction fitting (same labelling as obliteration paper sec 3.5)
# ---------------------------------------------------------------------------
def collect_pooled_examples(records, hs_key: str = "original_hs"):
    """Per-record-per-turn examples on the ORIGINAL arm:
       label 0 = refuse (t<t*), label 1 = comply (t==t*).

    ``hs_key`` selects which per-record hidden-state tensor to pull from --
    "original_hs" is the cached L16 tensor in replay_judged.pt; pass a
    different key when extra layers have been added by the caller.
    """
    Xs, ys, ridx = [], [], []
    for i, rec in enumerate(records):
        tob = rec["turn_of_breach"]
        hs = rec[hs_key]  # [num_turns, 4096]
        n_turns = hs.shape[0]
        for t in range(n_turns):
            if t < tob:
                Xs.append(hs[t].numpy()); ys.append(0); ridx.append(i)
            elif t == tob:
                Xs.append(hs[t].numpy()); ys.append(1); ridx.append(i)
    if not Xs:
        raise ValueError("No examples extracted -- empty dataset?")
    return np.stack(Xs), np.array(ys), np.array(ridx)


def fit_direction(records, train_record_ids, seed: int = 0,
                  hs_key: str = "original_hs"):
    """Fit logistic regression on the subset of records given by
    ``train_record_ids``. Returns (unit_direction, raw_coef, intercept)."""
    train_records = [records[i] for i in train_record_ids]
    X, y, _ = collect_pooled_examples(train_records, hs_key=hs_key)
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float32)
    direction = torch.from_numpy(w)
    norm = direction.norm().item()
    direction = direction / direction.norm()
    return direction, float(norm), float(clf.intercept_[0])


# ---------------------------------------------------------------------------
# Hidden-state extraction at an arbitrary layer (for non-default probe layers)
# ---------------------------------------------------------------------------
def hs_at_user_prefix(model, tok, conv_msgs, layer_idx, max_length=2048):
    """Hidden state at the last token of the prompt that ends with the latest
    user message (matches the convention used in turnstile/intent_replay.py)."""
    prompt = tok.apply_chat_template(
        conv_msgs, tokenize=False, add_generation_prompt=True,
    )
    return extract_hidden_states(
        model, tok, [prompt], layer_idx=layer_idx, max_length=max_length,
    ).squeeze(0)


def extract_records_hs_at_layer(records, vic_model, vic_tok, layer_idx,
                                hs_key, cache_path=None):
    """For each record, recompute per-turn user-prefix hidden states at
    ``layer_idx`` on the ORIGINAL conversation and store under ``hs_key``.

    Resumable via ``cache_path``: if present, populates from that file and
    skips records already covered. Records are mutated in place.
    """
    cached = {}
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        print(f"  loaded {len(cached)} cached HS extractions from {cache_path}")

    print(f"  extracting per-turn user-prefix HS at layer={layer_idx} for "
          f"{len(records)} records (key={hs_key})...")
    t0 = time.time()
    for i, rec in enumerate(records):
        if rec["source"] in cached:
            rec[hs_key] = cached[rec["source"]]
            continue
        n_turns = rec["original_hs"].shape[0]
        per_turn = []
        for t in range(n_turns):
            prefix = rec["original_conversation"][: 2 * t + 1]
            h = hs_at_user_prefix(vic_model, vic_tok, prefix,
                                  layer_idx=layer_idx)
            per_turn.append(h.cpu().float())
        rec[hs_key] = torch.stack(per_turn)
        cached[rec["source"]] = rec[hs_key]
        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(records)}] elapsed {elapsed/60:.1f} min")
            if cache_path:
                torch.save(cached, cache_path)
    if cache_path:
        torch.save(cached, cache_path)
    print(f"  HS extraction done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Replay with steering at the pivot turn
# ---------------------------------------------------------------------------
def replay_pivot_with_steering(vic_model, vic_tok, conversation, tob,
                               hook=None, layer_idx: int = 16,
                               max_tokens: int = 300):
    """Build the prompt for turns 0..t* (prelude + user msg at t*), generate
    the assistant response under the steering hook, return the response."""
    prefix = conversation[: 2 * tob + 1]
    if not prefix or prefix[-1]["role"] != "user":
        raise ValueError(f"Unexpected prefix (no trailing user turn): "
                         f"len={len(prefix)} last={prefix[-1]['role'] if prefix else None}")
    prompt = vic_tok.apply_chat_template(
        prefix, tokenize=False, add_generation_prompt=True,
    )
    if hook is not None:
        hook.register(vic_model, layer_idx)
    try:
        resp = generate_text(
            vic_model, vic_tok, prompt,
            max_tokens=max_tokens, temperature=0.0,
        )
    finally:
        if hook is not None:
            hook.remove()
    return resp.strip()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(args):
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.replay_pt}...")
    records = torch.load(args.replay_pt, weights_only=False)
    print(f"  {len(records)} records loaded")

    if args.max_records:
        records = records[: args.max_records]
        print(f"  truncated to first {len(records)} records")

    probe_layer = args.probe_layer
    steer_layer = args.steer_layer

    # Decide which per-record HS tensor to fit on. The cached "original_hs" is
    # already at L16; for any other probe layer we extract fresh and stash it
    # under "original_hs_l{N}".
    cached_layer = 16  # convention used by replay_judged.pt
    if probe_layer == cached_layer:
        hs_key = "original_hs"
    else:
        hs_key = f"original_hs_l{probe_layer}"

    print(f"\nProbe layer = {probe_layer}  "
          f"({'cached' if probe_layer == cached_layer else 'fresh extract'})")
    print(f"Steer layer = {steer_layer}")

    # We need the victim loaded before we can extract HS at a non-cached layer,
    # so load it now (needed regardless for the sweep).
    print(f"\nLoading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)

    if probe_layer != cached_layer:
        print(f"\n=== Phase 0: extract per-turn user-prefix HS at L{probe_layer} ===")
        cache_path = os.path.join(args.output, f"hs_l{probe_layer}_cache.pt")
        extract_records_hs_at_layer(
            records, vic_model, vic_tok,
            layer_idx=probe_layer, hs_key=hs_key, cache_path=cache_path,
        )

    # Stratified 5-fold CV by turn_of_breach
    n = len(records)
    tobs = np.array([r["turn_of_breach"] for r in records])
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)
    folds = list(skf.split(np.zeros(n), tobs))

    print(f"\n=== Phase 1: per-fold compliance direction at L{probe_layer} ===")
    fold_directions = []
    for k, (train_idx, test_idx) in enumerate(folds):
        direction, raw_norm, intercept = fit_direction(
            records, train_idx, seed=args.seed + k, hs_key=hs_key,
        )
        torch.save({
            "direction": direction,
            "raw_coef_norm": raw_norm,
            "intercept": intercept,
            "train_record_ids": train_idx.tolist(),
            "test_record_ids": test_idx.tolist(),
            "probe_layer": probe_layer,
            "steer_layer": steer_layer,
        }, os.path.join(args.output, f"direction_fold{k}.pt"))
        fold_directions.append((direction, train_idx, test_idx))
        print(f"  fold {k}: train={len(train_idx)}  test={len(test_idx)}  "
              f"raw||w||={raw_norm:.3f}  intercept={intercept:+.3f}")

    print(f"\nLoading strict dual judge "
          f"(Llama-Guard-3-1B + Llama-3.1-70B-Instruct)...")
    judge = StrictDualJudge()

    alphas = list(args.alphas)
    arms = ["original", "obliterated"]
    print(f"\n=== Phase 2: sweep alpha={alphas}  arms={arms} ===")
    print(f"  steer_layer={steer_layer}  max_tokens={args.max_tokens}")

    results_path = os.path.join(args.output, "results.jsonl")
    seen = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                seen.add((row["source"], row["arm"], row["alpha"]))
        print(f"  resuming: {len(seen)} (record,arm,alpha) tuples already done")

    n_planned = sum(len(test_idx) for _, _, test_idx in fold_directions) \
                * len(alphas) * len(arms)
    n_done_total = len(seen)

    t_start = time.time()
    for k, (direction, _train_idx, test_idx) in enumerate(fold_directions):
        for ridx in test_idx:
            rec = records[ridx]
            tob = rec["turn_of_breach"]
            for arm in arms:
                conv = (rec["original_conversation"] if arm == "original"
                        else rec["translated_conversation"])
                for alpha in alphas:
                    key = (rec["source"], arm, float(alpha))
                    if key in seen:
                        continue
                    hook = SteeringHook(direction, alpha) if alpha != 0 else None
                    try:
                        resp = replay_pivot_with_steering(
                            vic_model, vic_tok, conv, tob,
                            hook=hook, layer_idx=steer_layer,
                            max_tokens=args.max_tokens,
                        )
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"  [OOM] {rec['source']} arm={arm} α={alpha}: {e}")
                        torch.cuda.empty_cache()
                        if hook is not None:
                            hook.remove()
                        continue
                    judged_conv = list(conv[: 2 * tob + 1]) \
                                  + [{"role": "assistant", "content": resp}]
                    try:
                        verdict = judge.judge_conversation(
                            judged_conv, goal=rec["goal"],
                        )
                        verdict_dict = asdict(verdict)
                        judge_failed = False
                    except LocalJudgeFailure as e:
                        print(f"  [judge fail] {rec['source']} arm={arm} α={alpha}: {e}")
                        verdict_dict = {
                            "unsafe": None, "guard_unsafe": None,
                            "together_unsafe": None,
                            "raw_70b_output": str(e),
                            "reason": "judge_failed",
                        }
                        judge_failed = True
                    row = {
                        "source": rec["source"],
                        "fold": k,
                        "arm": arm,
                        "alpha": float(alpha),
                        "probe_layer": probe_layer,
                        "steer_layer": steer_layer,
                        "turn_of_breach": tob,
                        "goal": rec["goal"],
                        "behavior": rec.get("behavior"),
                        "category": rec.get("category"),
                        "response": resp,
                        "verdict": verdict_dict,
                        "judge_failed": judge_failed,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    with open(results_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    n_done_total += 1
                    if n_done_total % 25 == 0:
                        elapsed = time.time() - t_start
                        rate = max(n_done_total - len(seen), 1) / max(elapsed, 1)
                        remaining = n_planned - n_done_total
                        eta_min = remaining / max(rate, 1e-3) / 60.0
                        print(f"  [{n_done_total}/{n_planned}]  "
                              f"{rate:.2f} item/s  eta {eta_min:.1f} min")

    print(f"\nFinished sweep. Total rows: {n_done_total} (planned {n_planned}).")
    aggregate_and_print(args.output, alphas, arms)

    judge.unload()
    unload_model(vic_model, vic_tok)


def aggregate_and_print(output_dir, alphas, arms):
    rows = []
    path = os.path.join(output_dir, "results.jsonl")
    if not os.path.exists(path):
        print(f"  [aggregate] no results.jsonl at {path}")
        return
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    print(f"\n=== Aggregate compliance rates ===")
    print(f"{'arm':>12s}  {'alpha':>6s}  {'comply%':>8s}  "
          f"{'n_unsafe':>8s}  {'n_judged':>8s}  {'n_judge_fail':>12s}")
    print("-" * 64)
    summary = {}
    for arm in arms:
        for alpha in alphas:
            sub = [r for r in rows if r["arm"] == arm
                   and abs(r["alpha"] - float(alpha)) < 1e-9]
            n_total = len(sub)
            n_fail = sum(1 for r in sub if r["judge_failed"])
            judged = [r for r in sub if not r["judge_failed"]]
            n_unsafe = sum(1 for r in judged if r["verdict"]["unsafe"])
            n_judged = len(judged)
            rate = n_unsafe / max(n_judged, 1)
            summary[f"{arm}/alpha={alpha}"] = {
                "arm": arm, "alpha": alpha,
                "n_total": n_total, "n_judged": n_judged,
                "n_unsafe": n_unsafe, "rate": rate,
                "n_judge_fail": n_fail,
            }
            print(f"{arm:>12s}  {alpha:>+6.1f}  {100*rate:>7.1f}%  "
                  f"{n_unsafe:>8d}  {n_judged:>8d}  {n_fail:>12d}")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_dir}/summary.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-pt", required=True,
                   help="path to replay_judged.pt")
    p.add_argument("--output", required=True,
                   help="output directory")
    p.add_argument("--victim-model",
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--probe-layer", type=int, default=16,
                   help="layer at which to fit the compliance probe direction")
    p.add_argument("--steer-layer", type=int, default=None,
                   help="transformer block index for steering hook "
                        "(default: same as --probe-layer)")
    p.add_argument("--layer-idx", type=int, default=None,
                   help="(deprecated) sets --probe-layer and --steer-layer "
                        "to this value")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.layer_idx is not None:
        args.probe_layer = args.layer_idx
        args.steer_layer = args.layer_idx
    if args.steer_layer is None:
        args.steer_layer = args.probe_layer
    run_sweep(args)


if __name__ == "__main__":
    main()
