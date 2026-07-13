#!/usr/bin/env python
"""ASR + CE eval for compliance_obliteration_v1 LoRA, local judges only.

For each per-turn example in the trainer's held-out (source, round) split:
  - generate base assistant response given history
  - generate LoRA assistant response given the same history
  - compute NLL/token of the CORPUS assistant response under base and LoRA
  - judge each generation with turnstile.strict_dual_judge.StrictDualJudge
    (LOCAL Llama-Guard-3-1B + Llama-3.1-70B-Instruct, 4-bit BnB; NO together
    api call, per project policy)
  - append a JSONL row (resumable)

Drops rows where the 70B judge raises LocalJudgeFailure (logged loudly per
the project's no-silent-fallback rule).

Runs on a single A100 80GB. Loads all three models concurrently (~45GB).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turnstile.model_utils import _compute_assistant_spans, load_model, _BNB_CONFIG
from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure


POOL_SOURCES = [
    "selfplay_jbb_v1", "stealth_s42", "stealth_hard_s456",
    "control_s42", "control_hard_s456",
    "authority_dpo", "reward_dpo",
    "incrementalism_dpo", "urgency_dpo",
    "frozen_v1", "urgency_v1", "stealth_jbb_v1",
]
VICTIM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LABELS_REL = "experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl"


def load_examples(root, sources, labels_path):
    """Per-turn examples + goal/behavior/category metadata."""
    labels = {}
    with open(labels_path) as f:
        for line in f:
            r = json.loads(line)
            labels[(r["source"], r["round"], r["idx"])] = r.get("per_turn_unsafe") or []

    out = []
    for src in sources:
        rd = Path(root) / "experiments" / src / "rounds"
        if not rd.is_dir():
            continue
        for rf in sorted(rd.glob("round_*.jsonl")):
            rnum = int(rf.stem.replace("round_", ""))
            with open(rf) as f:
                for idx, line in enumerate(f):
                    rec = json.loads(line)
                    per_turn = labels.get((src, rnum, idx))
                    if per_turn is None:
                        continue
                    msgs = rec.get("conversation", [])
                    goal = rec.get("goal", "")
                    behavior = rec.get("behavior", "")
                    category = rec.get("category", "")
                    asst_indices = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
                    for t, ai in enumerate(asst_indices):
                        if t >= len(per_turn):
                            break
                        out.append({
                            "source": src, "round": rnum, "idx": idx, "turn": t,
                            "messages": msgs[: ai + 1],
                            "label": int(bool(per_turn[t])),
                            "goal": goal, "behavior": behavior, "category": category,
                        })
    return out


def stratified_split(examples, eval_frac=0.1, seed=0):
    rng = random.Random(seed)
    by_group = {}
    for i, ex in enumerate(examples):
        by_group.setdefault((ex["source"], ex["round"]), []).append(i)
    groups = list(by_group.keys())
    rng.shuffle(groups)
    n_eval = max(1, int(len(groups) * eval_frac))
    eval_groups = set(groups[:n_eval])
    train_idx, eval_idx = [], []
    for g, ids in by_group.items():
        (eval_idx if g in eval_groups else train_idx).extend(ids)
    return [examples[i] for i in train_idx], [examples[i] for i in eval_idx]


@torch.no_grad()
def gen(model, tokenizer, messages, max_new=256):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.eos_token_id, use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


@torch.no_grad()
def per_turn_ce(model, tokenizer, full_messages, max_length=2048):
    """NLL/token on the LAST assistant span (the corpus response)."""
    tokens, spans = _compute_assistant_spans(tokenizer, full_messages)
    if not spans:
        return None
    start, end = spans[-1]
    if tokens.shape[0] > max_length:
        drop = tokens.shape[0] - max_length
        tokens = tokens[drop:]
        start -= drop; end -= drop
        if start < 1:
            return None
    if start < 0 or end > tokens.shape[0] or end <= start:
        return None
    input_ids = tokens.unsqueeze(0).to(model.device)
    attention = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[0, start:end] = tokens[start:end]
    out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
    return float(out.loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-convs", type=int, default=50,
                    help="Number of held-out conversations to process")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ----- data -----
    labels_path = os.path.join(args.root, LABELS_REL)
    print(f"[data] loading examples (labels={labels_path})")
    examples = load_examples(args.root, POOL_SOURCES, labels_path)
    _, eval_ex = stratified_split(examples, eval_frac=0.1, seed=args.seed)
    print(f"[data] held-out per-turn pool: {len(eval_ex)}")

    convs = {}
    for ex in eval_ex:
        convs.setdefault((ex["source"], ex["round"], ex["idx"]), []).append(ex)
    rng = random.Random(args.seed + 1)
    all_keys = list(convs.keys())
    rng.shuffle(all_keys)
    picked = all_keys[: args.n_convs]
    total_turns = sum(len(convs[k]) for k in picked)
    print(f"[picks] {len(picked)} conversations ({total_turns} per-turn rows)")

    # Resumability — skip rows already in output JSONL
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r["source"], r["round"], r["idx"], r["turn"]))
    if done:
        print(f"[resume] {len(done)} rows already complete, will skip")

    # ----- models -----
    print(f"[load] victim {VICTIM_MODEL} (4-bit) + adapter {args.adapter}")
    victim, vt = load_model(VICTIM_MODEL, adapter_path=args.adapter)
    victim.eval()
    print("[load] judge: Llama-3.1-70B-Instruct (4-bit) + Llama-Guard-3-1B")
    judge = StrictDualJudge()

    out_fh = open(args.output, "a")
    t0 = time.time()
    n_done_this_run = 0
    n_judge_fail = 0

    for ki, key in enumerate(picked):
        src, rnd, idx = key
        for ex in convs[key]:
            full_key = (src, rnd, idx, ex["turn"])
            if full_key in done:
                continue

            history = ex["messages"][:-1]
            corpus_resp = ex["messages"][-1]["content"]

            # --- generations + CE under both arms ---
            try:
                with victim.disable_adapter():
                    base_resp = gen(victim, vt, history, max_new=args.max_new_tokens)
                    base_ce = per_turn_ce(victim, vt, ex["messages"])
                lora_resp = gen(victim, vt, history, max_new=args.max_new_tokens)
                lora_ce = per_turn_ce(victim, vt, ex["messages"])
            except RuntimeError as e:
                print(f"  [skip] {full_key} victim error: {e!r}")
                continue

            # --- judge both ---
            def _verdict(conv, goal):
                try:
                    v = judge.judge_conversation(conv, goal=goal)
                    return {
                        "unsafe": bool(v.unsafe),
                        "guard_unsafe": bool(v.guard_unsafe),
                        "judge_unsafe": (None if v.together_unsafe is None
                                         else bool(v.together_unsafe)),
                        "reason": v.reason,
                        "raw_70b": v.raw_70b_output,
                    }
                except LocalJudgeFailure as e:
                    print(f"  [judge-fail] {full_key}: {e!r}")
                    return {"unsafe": None, "error": str(e)}

            base_conv = history + [{"role": "assistant", "content": base_resp}]
            lora_conv = history + [{"role": "assistant", "content": lora_resp}]
            base_verdict = _verdict(base_conv, ex["goal"])
            lora_verdict = _verdict(lora_conv, ex["goal"])
            if base_verdict.get("unsafe") is None or lora_verdict.get("unsafe") is None:
                n_judge_fail += 1

            row = {
                "source": src, "round": rnd, "idx": idx, "turn": ex["turn"],
                "label": ex["label"],
                "goal": ex["goal"], "behavior": ex["behavior"],
                "category": ex["category"],
                "last_user": history[-1]["content"][:500] if history else "",
                "corpus_response": corpus_resp,
                "base_response": base_resp, "base_ce": base_ce,
                "lora_response": lora_resp, "lora_ce": lora_ce,
                "base_verdict": base_verdict,
                "lora_verdict": lora_verdict,
            }
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            n_done_this_run += 1

        elapsed = time.time() - t0
        per_row = elapsed / max(1, n_done_this_run)
        print(f"  [{ki+1}/{len(picked)}] {src}/r{rnd}/i{idx} done  "
              f"({n_done_this_run} rows, {elapsed:.0f}s, {per_row:.1f}s/row, "
              f"judge_fails={n_judge_fail})")

    out_fh.close()

    # ----- aggregate -----
    rows = []
    with open(args.output) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows
            if (r["base_verdict"].get("unsafe") is not None
                and r["lora_verdict"].get("unsafe") is not None)]

    def _asr(rs, arm):
        if not rs:
            return None
        return sum(1 for r in rs if r[f"{arm}_verdict"]["unsafe"]) / len(rs)

    def _mean(rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    summary = {
        "n_rows_valid_both_verdicts": len(rows),
        "n_judge_fail": n_judge_fail,
        "base_ASR": _asr(rows, "base"),
        "lora_ASR": _asr(rows, "lora"),
        "base_CE_mean": _mean(rows, "base_ce"),
        "lora_CE_mean": _mean(rows, "lora_ce"),
        "by_label": {},
        "by_turn": {},
        "by_source": {},
    }
    for lab in (0, 1):
        sub = [r for r in rows if r["label"] == lab]
        summary["by_label"][str(lab)] = {
            "n": len(sub),
            "base_ASR": _asr(sub, "base"),
            "lora_ASR": _asr(sub, "lora"),
            "base_CE": _mean(sub, "base_ce"),
            "lora_CE": _mean(sub, "lora_ce"),
        }
    for t in range(5):
        sub = [r for r in rows if r["turn"] == t]
        if sub:
            summary["by_turn"][str(t)] = {
                "n": len(sub),
                "base_ASR": _asr(sub, "base"),
                "lora_ASR": _asr(sub, "lora"),
            }
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        summary["by_source"][src] = {
            "n": len(sub),
            "base_ASR": _asr(sub, "base"),
            "lora_ASR": _asr(sub, "lora"),
        }

    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"n_valid={len(rows)}  judge_fails={n_judge_fail}")
    print(f"base_ASR = {summary['base_ASR']}")
    print(f"lora_ASR = {summary['lora_ASR']}")
    print(f"base_CE  = {summary['base_CE_mean']}")
    print(f"lora_CE  = {summary['lora_CE_mean']}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
