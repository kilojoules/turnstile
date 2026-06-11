"""Flexible additive-steering sweep with externally-loaded directions.

Used for H2 (low-baseline raw goal prompts) and H4 (re-fit
position-balanced compliance direction). Reuses single_prompt_steering_v2's
α-sweep loop, but takes pre-fit directions from a .pt file rather than
fitting them in-process.

Inputs:
  --inputs-json    : {test: [{harmful_breach, goal, record_idx, ...}, ...]}
  --directions-pt  : torch.save dict mapping label -> {direction, info}
  --layer          : steer layer
  --alphas         : alpha grid (raw, NOT ratio of ‖h‖; you supply absolute α)
  --median-h       : median ‖h‖ at this layer (for the row's alpha_h_ratio col)
  --label          : direction label inside the .pt to use
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
from turnstile.single_prompt_steering_v2 import (
    SteeringHook, coherence_metrics, steer_and_generate,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-json", required=True)
    p.add_argument("--directions-pt", required=True)
    p.add_argument("--label", required=True,
                   help="key inside directions-pt (e.g. L16_lr_compliance)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--median-h", type=float, required=True,
                   help="Median ||h|| at this layer (used to fill alpha_h_ratio column).")
    p.add_argument("--alphas", type=float, nargs="+", required=True,
                   help="Absolute α values to sweep.")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    inp = json.load(open(args.inputs_json))
    test = inp["test"]
    print(f"loaded {len(test)} test prompts")

    blob = torch.load(args.directions_pt, weights_only=False)
    if args.label not in blob:
        raise SystemExit(f"label {args.label!r} not in {list(blob.keys())}")
    direction = blob[args.label]["direction"]
    info = blob[args.label].get("info", {})
    print(f"direction {args.label}: shape={tuple(direction.shape)} info={info}")
    print(f"layer={args.layer} median_h={args.median_h:.2f} alphas={args.alphas}")

    seen = set()
    if os.path.exists(args.out_jsonl):
        for line in open(args.out_jsonl):
            if line.strip():
                r = json.loads(line)
                seen.add((r["test_record_idx"], r["alpha"]))

    todo = []
    for a in args.alphas:
        for p in test:
            if (p["record_idx"], float(a)) not in seen:
                todo.append((p, float(a)))
    print(f"{len(todo)} configs to run; skip {len(seen)}")

    print(f"loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)
    print("loading strict dual judge...")
    judge = StrictDualJudge()

    n_done = 0; n_fail = 0; n_broken = 0
    t0 = time.time()
    try:
        for (p, alpha) in todo:
            try:
                resp = steer_and_generate(
                    vic_model, vic_tok, p["harmful_breach"],
                    direction, args.layer, alpha, args.max_tokens,
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
                "layer": args.layer,
                "direction_label": args.label,
                "alpha": float(alpha),
                "alpha_h_ratio": float(alpha / max(args.median_h, 1e-6)),
                "median_h_at_layer": args.median_h,
                "response": resp,
                "oom": oom,
                "coherence": coh,
                "verdict": verdict_dict,
                "judge_failed": judge_failed,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(args.out_jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta_min = (len(todo) - n_done) / max(rate, 1e-3) / 60
                print(f"[flex] [{n_done}/{len(todo)}] {rate:.2f} cfg/s  "
                      f"eta {eta_min:.1f} min  ({n_fail} fail, {n_broken} broken)")
        print(f"[flex] finished: {n_done}, fails={n_fail}, broken={n_broken}")
    finally:
        judge.unload()
        unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
