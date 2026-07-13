"""Audit: do experiments/{exp}/hidden_states/round_N.pt and
experiments/{exp}/rounds/round_N.jsonl correspond to the same corpus, or
could the .pt files be stale from an earlier run?

Checks per round:
  1. n_convs match between .pt and .jsonl
  2. labels in .pt == unsafe in .jsonl (index by index)
  3. turns_of_breach in .pt == turn_of_breach in .jsonl (index by index)
  4. hidden state shape consistent with conversation length
  5. .pt mtime >= .jsonl mtime
  6. persisted "round" field matches filename
  7. hidden_dim invariant across all .pt files

Plus a cross-round byte-equality scan: how many (round, round) pairs have
bit-identical first-conv-first-turn hidden states? Identical inputs produce
deterministic outputs, so duplicates here corroborate the memorized-opener
finding rather than indicating a bug.

Usage:
  python -m turnstile.intent_audit_alignment \
      --experiments experiments/{authority_dpo,...} \
      [--output experiments/intent_obliteration/audit.json]
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    n_rounds = 0
    label_violations = []
    tob_violations = []
    count_violations = []
    shape_violations = []
    mtime_violations = []
    round_field_violations = []
    dims = set()
    hs_pairs = []  # (label, first-conv-first-turn-hs) for cross-round scan
    duplicate_pairs = []

    for exp_dir in args.experiments:
        exp = os.path.basename(exp_dir.rstrip("/"))
        rounds_dir = Path(exp_dir) / "rounds"
        hs_dir = Path(exp_dir) / "hidden_states"
        if not rounds_dir.is_dir() or not hs_dir.is_dir():
            print(f"  [skip] {exp}: missing rounds/ or hidden_states/")
            continue
        for jsonl in sorted(rounds_dir.glob("round_*.jsonl"),
                            key=lambda q: int(q.stem.split("_")[1])):
            rd = int(jsonl.stem.split("_")[1])
            pt = hs_dir / f"round_{rd}.pt"
            if not pt.exists():
                count_violations.append(f"{exp}/r{rd}: missing .pt")
                continue
            n_rounds += 1
            with open(jsonl) as f:
                convs = [json.loads(l) for l in f if l.strip()]
            d = torch.load(pt, weights_only=False)

            if len(d["hidden_states"]) != len(convs):
                count_violations.append(
                    f"{exp}/r{rd}: hs={len(d['hidden_states'])} jsonl={len(convs)}")
                continue
            if d.get("round") != rd:
                round_field_violations.append(
                    f"{exp}/r{rd}: stored round={d.get('round')}")
            for i, (hs, lab, tob, rec) in enumerate(zip(
                d["hidden_states"], d["labels"], d["turns_of_breach"], convs
            )):
                if bool(lab) != bool(rec.get("unsafe", False)):
                    label_violations.append(f"{exp}/r{rd}/i{i}")
                stored_tob = tob if tob is not None else None
                if stored_tob != rec.get("turn_of_breach"):
                    tob_violations.append(f"{exp}/r{rd}/i{i}")
                if hs is not None:
                    n_user = sum(1 for m in rec["conversation"]
                                 if m["role"] == "user")
                    if hs.shape[0] > n_user:
                        shape_violations.append(
                            f"{exp}/r{rd}/i{i}: hs turns {hs.shape[0]} > "
                            f"user msgs {n_user}")
                    dims.add(int(hs.shape[-1]))

            if pt.stat().st_mtime < jsonl.stat().st_mtime - 1.0:
                mtime_violations.append(f"{exp}/r{rd}")

            for hs in d["hidden_states"]:
                if hs is not None:
                    hs_pairs.append((f"{exp}/r{rd}", hs[0].clone()))
                    break

    # cross-round byte-equality scan
    n_dup = 0
    for (la, ha), (lb, hb) in itertools.combinations(hs_pairs, 2):
        if torch.equal(ha, hb):
            n_dup += 1
            if n_dup <= 5:
                duplicate_pairs.append({"a": la, "b": lb})

    print("=" * 60)
    print("Activation-conversation alignment audit")
    print("=" * 60)
    print(f"rounds checked:           {n_rounds}")
    print(f"hidden_dim values seen:   {dims}")
    print(f"label-mismatch:           {len(label_violations)}")
    print(f"tob-mismatch:             {len(tob_violations)}")
    print(f"count-mismatch:           {len(count_violations)}")
    print(f"shape-mismatch:           {len(shape_violations)}")
    print(f"mtime-violations:         {len(mtime_violations)}")
    print(f"round-field violations:   {len(round_field_violations)}")
    print(f"\ncross-round byte-identical first-conv-first-turn hidden states: "
          f"{n_dup} / {len(list(itertools.combinations(hs_pairs, 2)))} pairs "
          f"({n_dup / max(1, len(list(itertools.combinations(hs_pairs, 2)))) * 100:.2f}%)")
    print("(expected: nonzero, due to memorized opener / same-seed-same-goal)")

    summary = {
        "rounds_checked": n_rounds,
        "hidden_dim_values": sorted(dims),
        "violations": {
            "label_mismatch": label_violations,
            "tob_mismatch": tob_violations,
            "count_mismatch": count_violations,
            "shape_mismatch": shape_violations,
            "mtime": mtime_violations,
            "round_field": round_field_violations,
        },
        "cross_round_byte_identical_first_turn": {
            "n_duplicate_pairs": n_dup,
            "n_total_pairs": len(list(itertools.combinations(hs_pairs, 2))),
            "first_5_dups": duplicate_pairs,
        },
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
