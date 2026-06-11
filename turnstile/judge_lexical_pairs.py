"""Run-then-judge filter for the lexical swap dataset and 10 hand-edited
gold pairs.

For each pair we actually invoke the 8B victim on (buildup_preserved +
edited_breach_user_turn), capture its real response, and judge that
response against the *original harmful goal* via the strict dual judge's
``judge_conversation``. We do this at two sampling temperatures:
  - T=0.0 (deterministic; clean causal contrast)
  - T=0.7 (matches the v2 sweep settings; honest noisy estimate)

This replaces an earlier broken version that fed a hardcoded refusal as
the assistant turn, which made the 70B compliance judge always say safe.

Two outputs:
  - judged_lexical_swaps.jsonl: one row per swap pair with verdicts at
    T=0.0 and T=0.7, plus the raw victim responses.
  - judged_gold_pairs.jsonl: one row per gold pair, each with verdicts
    at T=0.0 and T=0.7 for *both* the harmful and benign variants.
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


TEMPS = (0.0, 0.7)


def _generate_victim(vic_model, vic_tok, conv, temperature, max_tokens):
    prompt = vic_tok.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True,
    )
    return generate_text(
        vic_model, vic_tok, prompt,
        max_tokens=max_tokens, temperature=temperature,
    ).strip()


def _judge_one(judge, conv, goal):
    try:
        verdict = judge.judge_conversation(conv, goal=goal)
        return asdict(verdict), False
    except LocalJudgeFailure as e:
        return ({"unsafe": None, "guard_unsafe": None,
                 "together_unsafe": None, "raw_70b_output": str(e),
                 "reason": "judge_failed"}, True)


def _run_then_judge(vic_model, vic_tok, judge, conv_prefix, goal, max_tokens):
    """Run victim + judge at both temperatures. Returns
    {"t00": {"response": str, "verdict": dict, "judge_failed": bool}, "t07": {...}}."""
    out = {}
    for temp in TEMPS:
        key = "t00" if temp == 0.0 else "t07"
        try:
            resp = _generate_victim(vic_model, vic_tok, conv_prefix, temp, max_tokens)
            oom = False
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            resp = None
            oom = True
        if resp is None:
            out[key] = {"response": None, "oom": True,
                        "verdict": None, "judge_failed": True}
            continue
        full_conv = list(conv_prefix) + [{"role": "assistant", "content": resp}]
        verdict_dict, judge_failed = _judge_one(judge, full_conv, goal)
        out[key] = {"response": resp, "oom": oom,
                    "verdict": verdict_dict, "judge_failed": judge_failed}
    return out


def judge_lexical_swaps(vic_model, vic_tok, judge, in_path, out_path, max_tokens):
    print(f"\n=== Judging lexical swap pairs ({in_path}) ===")
    d = json.load(open(in_path))
    records = d["records"]
    seen = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                seen.add(json.loads(line)["source_id"])

    todo = [r for r in records
            if r.get("edited_breach_user_turn") is not None
            and r["source_id"] not in seen]
    print(f"  {len(records)} total; {len(seen)} already done; {len(todo)} todo")

    n_done = 0
    t0 = time.time()
    for rec in todo:
        conv_prefix = list(rec["buildup_preserved"])
        conv_prefix.append({"role": "user",
                            "content": rec["edited_breach_user_turn"]})
        verdicts = _run_then_judge(
            vic_model, vic_tok, judge, conv_prefix, rec["goal"], max_tokens,
        )
        row = {
            "source_id": rec["source_id"],
            "category": rec["category"],
            "goal": rec["goal"],
            "experiment": rec["experiment"],
            "swap_status": rec["swap_status"],
            "n_swaps": rec["n_swaps"],
            "char_delta": rec.get("char_delta"),
            "harmful_breach_user_turn": rec["harmful_breach_user_turn"],
            "edited_breach_user_turn": rec["edited_breach_user_turn"],
            "verdicts": verdicts,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta_min = (len(todo) - n_done) / max(rate, 1e-3) / 60
            print(f"  [{n_done}/{len(todo)}] {rate:.2f} pair/s  eta {eta_min:.1f} min")
    print(f"  finished: {n_done} new pairs")


def judge_gold_pairs(vic_model, vic_tok, judge, in_path, out_path, max_tokens):
    print(f"\n=== Judging gold-standard pairs ({in_path}) ===")
    d = json.load(open(in_path))
    pairs = d["pairs"]
    seen = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                seen.add(json.loads(line)["pair_id"])

    for pair in pairs:
        if pair["pair_id"] in seen:
            continue
        result = {
            "pair_id": pair["pair_id"],
            "source_id": pair["source_id"],
            "category": pair["category"],
            "goal": pair["goal"],
            "minimal_edit_suitability": pair["minimal_edit_suitability"],
            "n_words_changed": pair["n_words_changed"],
            "edits": pair["edits"],
            "notes": pair["notes"],
            "verdicts_by_variant": {},
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        for variant in ("harmful", "benign"):
            v = pair[variant]
            conv_prefix = []
            if v.get("prelude_user_turn"):
                conv_prefix.append({"role": "user",
                                    "content": v["prelude_user_turn"]})
                # No real assistant response in the gold pair payload, so
                # generate one here too. Keep it deterministic.
                pre_prompt = vic_tok.apply_chat_template(
                    conv_prefix, tokenize=False, add_generation_prompt=True,
                )
                pre_resp = generate_text(
                    vic_model, vic_tok, pre_prompt,
                    max_tokens=max_tokens, temperature=0.0,
                ).strip()
                conv_prefix.append({"role": "assistant", "content": pre_resp})
            conv_prefix.append({"role": "user",
                                "content": v["breach_user_turn"]})
            verdicts = _run_then_judge(
                vic_model, vic_tok, judge, conv_prefix, pair["goal"], max_tokens,
            )
            result["verdicts_by_variant"][variant] = verdicts
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        h00 = result["verdicts_by_variant"]["harmful"]["t00"]["verdict"]
        b00 = result["verdicts_by_variant"]["benign"]["t00"]["verdict"]
        h07 = result["verdicts_by_variant"]["harmful"]["t07"]["verdict"]
        b07 = result["verdicts_by_variant"]["benign"]["t07"]["verdict"]
        print(f"  pair {pair['pair_id']:>2}: "
              f"T=0 harmful={h00 and h00.get('unsafe')} benign={b00 and b00.get('unsafe')}  "
              f"T=.7 harmful={h07 and h07.get('unsafe')} benign={b07 and b07.get('unsafe')}  "
              f"({pair['minimal_edit_suitability']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lexical-swaps-json", required=True)
    p.add_argument("--gold-pairs-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--victim-model",
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)

    print("Loading strict dual judge (Llama-Guard-3-1B + Llama-3.1-70B-Instruct)...")
    judge = StrictDualJudge()

    try:
        judge_gold_pairs(
            vic_model, vic_tok, judge, args.gold_pairs_json,
            os.path.join(args.out_dir, "judged_gold_pairs.jsonl"),
            args.max_tokens,
        )
        judge_lexical_swaps(
            vic_model, vic_tok, judge, args.lexical_swaps_json,
            os.path.join(args.out_dir, "judged_lexical_swaps.jsonl"),
            args.max_tokens,
        )
    finally:
        judge.unload()
        unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
