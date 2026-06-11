"""H3 follow-up: re-run the L31 mean-diff α sweep at much larger α to test
whether the "L31 inertness" finding is an artifact of unit-normalization.

Background. In the main single-prompt sweep we use unit-normalized
directions and α∈[−8,+8]. At L16 the residual stream norm is ~10, so
α=8 = 82% of ambient ‖h‖. At L31 ‖h‖ is ~146, so α=8 is only 5.5% of
ambient -- in Arditi's natural raw-mean-diff units this corresponds to
α≈0.11, i.e. ~10× *under* the natural class-separation scale.

This module re-runs (L31, mean-diff) only, at a wider α grid that matches
Arditi's natural scale (α=1 in raw-magnitude units = α≈73 in our
unit-normalized parameterization).
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

import torch

from turnstile.model_utils import load_model, generate_text, unload_model
from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure
from turnstile.single_prompt_steering import SteeringHook, steer_and_generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-json", required=True)
    p.add_argument("--directions-pt", required=True,
                   help="Path to directions.pt from the original single-prompt run.")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-145.0, -72.0, -36.0, -18.0, 0.0, 18.0, 36.0, 72.0, 145.0])
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    inp = json.load(open(args.inputs_json))
    harmful = inp["harmful"]

    blob = torch.load(args.directions_pt, weights_only=False)
    direction = blob["L31_meandiff"]["direction"]
    print(f"[setup] direction L31_meandiff loaded; shape={direction.shape},"
          f" raw_norm (from fit info)={blob['L31_meandiff']['info']['raw_norm']:.2f}")
    print(f"[setup] alpha grid: {args.alphas}")

    seen = set()
    if os.path.exists(args.out_jsonl):
        for line in open(args.out_jsonl):
            if line.strip():
                r = json.loads(line)
                seen.add((r["prompt_id"], float(r["alpha"])))
    todo = [(h, a) for h in harmful for a in args.alphas
            if (h["prompt_id"], float(a)) not in seen]
    print(f"[setup] {len(todo)} (prompt, alpha) configs to run; skip {len(seen)} already done")

    print(f"[setup] loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)
    print("[setup] loading strict dual judge...")
    judge = StrictDualJudge()

    n_done = 0
    n_fail = 0
    t0 = time.time()
    try:
        for h, a in todo:
            try:
                resp = steer_and_generate(
                    vic_model, vic_tok, h["prompt"], direction, 31, a, args.max_tokens,
                )
                oom = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                resp = None
                oom = True
            if resp is None:
                row = {"prompt_id": h["prompt_id"], "goal": h["goal"],
                       "category": h.get("category"),
                       "layer": 31, "method": "meandiff", "alpha": float(a),
                       "response": None, "oom": True, "verdict": None,
                       "judge_failed": True,
                       "ts": datetime.now(timezone.utc).isoformat()}
                with open(args.out_jsonl, "a") as f:
                    f.write(json.dumps(row) + "\n")
                n_fail += 1
                continue
            conv = [{"role": "user", "content": h["prompt"]},
                    {"role": "assistant", "content": resp}]
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
                "layer": 31,
                "method": "meandiff",
                "alpha": float(a),
                "response": resp,
                "verdict": verdict_dict,
                "judge_failed": judge_failed,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(args.out_jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            if n_done % 20 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta_min = (len(todo) - n_done) / max(rate, 1e-3) / 60
                print(f"[run] [{n_done}/{len(todo)}] {rate:.2f} cfg/s  "
                      f"eta {eta_min:.1f} min  ({n_fail} fail)")
        print(f"[run] finished: {n_done} new configs, {n_fail} fails")
    finally:
        judge.unload()
        unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
