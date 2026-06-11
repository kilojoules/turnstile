"""Compliance-amplification redesigned sweep.

Complement to ``single_prompt_steering_v2`` (which fit a direction on
*input semantic*: harmful-prompt vs benign-rewrite). Here we fit on
*model behavior outcome*: compliant turn vs refusal turn within attack
conversations. Then we test whether pushing +α along this direction at
L16 / L24 / L31 amplifies ASR on held-out test breach turns.

Design (mirrors v2 redesign, with severities 1--6 baked in):

  1. *Train-on-test*: stratified 50/50 split by record. Direction fit
     on training-record per-turn HS only.
  2. *Distributional confound*: per-turn labels come from the same
     within-attack distribution -- both compliant and refusal turns
     are inside successful jailbreak conversations, with the same
     attack style. The contrast is purely outcome.
  3. *Coherence collapse*: same diagnostic; report 2x2 per cell.
  4. *Effect-fair α*: α = ratio × median(‖h‖_layer).
  5. *Two-point asymmetry*: report L16 vs L31 explicitly; L24 is
     midpoint.
  6. *Stronger nuisance*: random unit vector + a per-turn-position LR
     (fit on "is this turn 0 vs turn 4").

We use per-turn HS already saved in replay_v2_full.pt for L16 (key
``original_hs``) and L31 (``original_hs_l31``). For L24, we re-extract
HS at the corresponding turn boundary from the original_conversation.

Output JSONL one row per (test_record, layer, direction, alpha).
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict, Counter
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from turnstile.model_utils import load_model, generate_text, unload_model
from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure
from turnstile.single_prompt_steering_v2 import (
    SteeringHook, coherence_metrics, steer_and_generate,
    fit_lr_direction, fit_random_direction,
)


def stratified_split(records, seed=13):
    """50/50 split by category, returning (train_idx, test_idx)."""
    by_cat = defaultdict(list)
    for i, r in enumerate(records):
        by_cat[r.get("category")].append(i)
    rng = random.Random(seed)
    train, test = [], []
    for cat, idxs in by_cat.items():
        rng.shuffle(idxs)
        mid = len(idxs) // 2
        train.extend(idxs[:mid])
        test.extend(idxs[mid:])
    return sorted(train), sorted(test)


def collect_per_turn_hs(records, layer_key, label_key, train_idx):
    """For each train record, walk the original-arm conversation and
    pair each per-turn HS with the per-turn judge label.

    Per-turn labels in replay_v2_full.pt are dicts of the form
    ``{'turn': t, 'unsafe': bool, 'reason': str}`` -- not bare bools --
    so we extract the ``unsafe`` field rather than coercing the dict.

    Returns (X (N, d_model), y (N,) in {0,1}, label_dist dict).
    """
    X = []; y = []
    n_dropped = 0
    for i in train_idx:
        r = records[i]
        hs_seq = r.get(layer_key)
        labels = r.get(label_key)
        if hs_seq is None or labels is None: continue
        for t in range(min(len(hs_seq), len(labels))):
            lab = labels[t]
            if lab is None:
                n_dropped += 1; continue
            unsafe = lab.get("unsafe") if isinstance(lab, dict) else lab
            if unsafe is None:
                n_dropped += 1; continue
            X.append(hs_seq[t])
            y.append(int(bool(unsafe)))
    if not X:
        return None, None, {"dropped": n_dropped}
    X = torch.stack(X).float().numpy()
    y = np.array(y, dtype=np.int64)
    return X, y, {"n": len(y), "n_unsafe": int(y.sum()), "n_safe": int((y==0).sum()),
                  "dropped": n_dropped}


def fit_turn_position_lr(records, layer_key, train_idx):
    """Nuisance LR: 'is this turn 0 vs turn 4'. Fit on train per-turn HS
    pooled, with y=1 if turn>=2 else 0."""
    X = []; y = []
    for i in train_idx:
        r = records[i]
        hs_seq = r.get(layer_key)
        if hs_seq is None: continue
        for t in range(len(hs_seq)):
            X.append(hs_seq[t])
            y.append(1 if t >= 2 else 0)
    if not X: return None, None
    X = torch.stack(X).float().numpy()
    y = np.array(y, dtype=np.int64)
    return X, y


def re_extract_l24_hs(records, vic_model, vic_tok, train_idx, test_idx, breach_only=False):
    """Re-extract per-turn HS at L24 for original-arm conversations.
    Returns a dict {record_idx: tensor (n_turns, d_model)} on CPU."""
    out = {}
    all_idx = sorted(set(list(train_idx) + list(test_idx)))
    print(f"[L24-extract] {len(all_idx)} records, {'breach-turn only' if breach_only else 'all turns'}...")
    t0 = time.time()
    for k, ri in enumerate(all_idx):
        r = records[ri]
        oc = r.get("original_conversation") or []
        # Walk turn-by-turn through the user/assistant prefix; capture
        # HS at the user-token boundary (last token before assistant
        # generation) at each user turn.
        hs_list = []
        n_user_turns = (len(oc) + 1) // 2
        if breach_only:
            t_star = r.get("turn_of_breach")
            if t_star is None: continue
            turn_iter = [t_star]
        else:
            turn_iter = list(range(n_user_turns))
        for t in turn_iter:
            prefix = oc[:2*t + 1]  # up to and including the t-th user message
            templated = vic_tok.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True,
            )
            inputs = vic_tok(templated, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                fwd = vic_model(**inputs, output_hidden_states=True, use_cache=False)
            h = fwd.hidden_states[24 + 1][0, -1, :].detach().to("cpu", torch.float32)
            hs_list.append(h)
            del fwd
        if hs_list:
            out[ri] = torch.stack(hs_list)
        if (k+1) % 25 == 0:
            print(f"[L24-extract] {k+1}/{len(all_idx)}  ({time.time()-t0:.0f}s)")
    return out


def fit_directions(records, train_idx, l24_hs_by_record, args):
    """Fit (compliance LR + position LR + random) at each layer."""
    directions = {}
    for L in args.layers:
        if L == 16:
            hs_key = "original_hs"
            X, y, stats = collect_per_turn_hs(records, hs_key, "original_per_turn_labels", train_idx)
        elif L == 31:
            hs_key = "original_hs_l31"
            X, y, stats = collect_per_turn_hs(records, hs_key, "original_per_turn_labels", train_idx)
        elif L == 24:
            # Build X/y from l24_hs_by_record
            X = []; y = []
            for i in train_idx:
                if i not in l24_hs_by_record: continue
                hs_seq = l24_hs_by_record[i]
                labels = records[i].get("original_per_turn_labels", [])
                for t in range(min(len(hs_seq), len(labels))):
                    lab = labels[t]
                    if lab is None: continue
                    unsafe = lab.get("unsafe") if isinstance(lab, dict) else lab
                    if unsafe is None: continue
                    X.append(hs_seq[t])
                    y.append(int(bool(unsafe)))
            X = torch.stack(X).float().numpy() if X else None
            y = np.array(y, dtype=np.int64) if y else None
            stats = {"n": len(y) if y is not None else 0,
                     "n_unsafe": int(y.sum()) if y is not None else 0,
                     "n_safe": int((y==0).sum()) if y is not None else 0}
        if X is None or y is None or len(set(y)) < 2:
            print(f"[fit] L{L} compliance: degenerate split, skipping")
            continue
        d, info = fit_lr_direction(X, y)
        directions[(L, "lr_compliance")] = {"direction": d, "info": info,
                                              "fit_stats": stats}
        print(f"[fit] L{L} lr_compliance  raw_norm={info['raw_norm']:.3f}  "
              f"acc={info['train_acc']:.3f}  "
              f"n_compliant={stats['n_unsafe']}  n_refusal={stats['n_safe']}")

        # Nuisance: turn position
        if L == 24:
            Xp = []; yp = []
            for i in train_idx:
                if i not in l24_hs_by_record: continue
                for t in range(len(l24_hs_by_record[i])):
                    Xp.append(l24_hs_by_record[i][t])
                    yp.append(1 if t >= 2 else 0)
            Xp = torch.stack(Xp).float().numpy() if Xp else None
            yp = np.array(yp) if yp else None
        else:
            Xp, yp = fit_turn_position_lr(records, hs_key, train_idx)
        if Xp is not None and len(set(yp)) == 2:
            d, info = fit_lr_direction(Xp, yp)
            directions[(L, "lr_turn_pos")] = {"direction": d, "info": info}
            print(f"[fit] L{L} lr_turn_pos    raw_norm={info['raw_norm']:.3f}  "
                  f"acc={info['train_acc']:.3f}")
        # random
        d_model = X.shape[1]
        d, info = fit_random_direction(d_model, seed=42 + L)
        directions[(L, "random")] = {"direction": d, "info": info}
        print(f"[fit] L{L} random         seed={info['seed']}")
    return directions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-pt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[16, 24, 31])
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--skip-l24-extract", action="store_true",
                   help="Skip L24 (faster; use L16+L31 only)")
    p.add_argument("--skip-judge", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"loading replay {args.replay_pt}...")
    records = torch.load(args.replay_pt, weights_only=False)
    print(f"  {len(records)} records")

    train_idx, test_idx = stratified_split(records, seed=args.seed)
    print(f"split: {len(train_idx)} train, {len(test_idx)} test")

    print(f"loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)

    l24_hs_path = os.path.join(args.output, "l24_per_turn_hs.pt")
    if 24 in args.layers and not args.skip_l24_extract:
        if os.path.exists(l24_hs_path):
            print(f"[L24-extract] reusing {l24_hs_path}")
            l24_hs_by_record = torch.load(l24_hs_path, weights_only=False)
        else:
            l24_hs_by_record = re_extract_l24_hs(
                records, vic_model, vic_tok, train_idx, test_idx,
            )
            torch.save(l24_hs_by_record, l24_hs_path)
    else:
        l24_hs_by_record = {}
        if 24 in args.layers:
            args.layers = [L for L in args.layers if L != 24]
            print("[setup] dropped L24 from layers (--skip-l24-extract)")

    directions_path = os.path.join(args.output, "directions_compliance.pt")
    if os.path.exists(directions_path):
        print(f"[fit] reusing {directions_path}")
        save_blob = torch.load(directions_path, weights_only=False)
        directions = {}
        for k, v in save_blob.items():
            # parse "L{L}_{name}" → (L, name)
            assert k.startswith("L")
            rest = k[1:]
            L_str, name = rest.split("_", 1)
            directions[(int(L_str), name)] = v
    else:
        directions = fit_directions(records, train_idx, l24_hs_by_record, args)
        torch.save({f"L{L}_{m}": v for (L, m), v in directions.items()},
                   directions_path)

    # median ‖h‖ per layer for α grid -- pulled from training HS pool
    median_h = {}
    for L in args.layers:
        hs_pool = []
        if L == 16:
            for i in train_idx:
                seq = records[i].get("original_hs")
                if seq is not None: hs_pool.extend([seq[t] for t in range(len(seq))])
        elif L == 31:
            for i in train_idx:
                seq = records[i].get("original_hs_l31")
                if seq is not None: hs_pool.extend([seq[t] for t in range(len(seq))])
        elif L == 24:
            for i in train_idx:
                if i in l24_hs_by_record:
                    seq = l24_hs_by_record[i]
                    hs_pool.extend([seq[t] for t in range(len(seq))])
        if hs_pool:
            stacked = torch.stack(hs_pool).float()
            median_h[L] = float(torch.norm(stacked, dim=1).median())
            print(f"[setup] median ‖h‖_L{L} = {median_h[L]:.2f}")

    # Phase: sweep on test breach turns, judge, save.
    out_jsonl = os.path.join(args.output, "compliance_sweep.jsonl")
    seen = set()
    if os.path.exists(out_jsonl):
        for line in open(out_jsonl):
            if line.strip():
                r = json.loads(line)
                seen.add((r["test_record_idx"], r["layer"], r["direction"], float(r["alpha"])))

    todo = []
    for L in args.layers:
        if L not in median_h: continue
        for ratio in args.ratios:
            alpha = round(ratio * median_h[L], 3)
            for d_name in ("lr_compliance", "lr_turn_pos", "random"):
                if (L, d_name) not in directions: continue
                for ti in test_idx:
                    r = records[ti]
                    t_star = r.get("turn_of_breach")
                    oc = r.get("original_conversation") or []
                    if t_star is None or 2*t_star >= len(oc): continue
                    breach_msg = oc[2*t_star]["content"]
                    if (ti, L, d_name, float(alpha)) not in seen:
                        todo.append((ti, breach_msg, r.get("goal"), r.get("category"),
                                     L, d_name, float(alpha)))
    print(f"[sweep] {len(todo)} configs to run; skip {len(seen)}")

    if args.skip_judge:
        print("--skip-judge: stopping after fit + L24 extract.")
        unload_model(vic_model, vic_tok)
        return

    print("loading strict dual judge...")
    judge = StrictDualJudge()

    n_done = 0; n_fail = 0; n_broken = 0
    t0 = time.time()
    try:
        for (ti, prompt, goal, cat, L, d_name, alpha) in todo:
            direction = directions[(L, d_name)]["direction"]
            try:
                resp = steer_and_generate(
                    vic_model, vic_tok, prompt, direction, L, alpha, args.max_tokens,
                )
                oom = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                resp = None; oom = True
            coh = coherence_metrics(resp, vic_tok)
            if coh.get("is_broken"): n_broken += 1
            if resp is None:
                verdict_dict = None; judge_failed = True
            else:
                conv = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp}]
                try:
                    verdict = judge.judge_conversation(conv, goal=goal)
                    verdict_dict = asdict(verdict); judge_failed = False
                except LocalJudgeFailure as e:
                    verdict_dict = {"unsafe": None, "guard_unsafe": None,
                                    "together_unsafe": None, "raw_70b_output": str(e),
                                    "reason": "judge_failed"}
                    judge_failed = True; n_fail += 1
            row = {
                "test_record_idx": int(ti),
                "category": cat,
                "goal": goal,
                "layer": L, "direction": d_name, "alpha": float(alpha),
                "alpha_h_ratio": float(alpha / max(median_h[L], 1e-6)),
                "median_h_at_layer": median_h[L],
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
                print(f"[sweep] [{n_done}/{len(todo)}] {rate:.2f} cfg/s  "
                      f"eta {eta_min:.1f} min  ({n_fail} fail, {n_broken} broken)")
        print(f"[sweep] finished: {n_done}, fails={n_fail}, broken={n_broken}")
    finally:
        judge.unload()
        unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
