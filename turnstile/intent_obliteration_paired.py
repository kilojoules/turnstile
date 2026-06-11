"""Paired-probe analysis: original (unsafe) vs translated (benign) at matched
turn positions, with identical preludes (turns 0..t*-1).

This is the controlled obliteration test. The earlier ``intent_obliteration.py``
just compared adversarial pre-pivot turns against the existing benign-curious
set, which was confounded by structural differences (different distributions,
the trained adversary's memorized opener appears only on the adversarial side,
etc.). This script eliminates those confounds by construction:

  - Both sides come from the same conversations.
  - The benign side inherits the original prelude verbatim, including the
    fake-refusal opener.
  - Only the user pivot turn (and its successors) differ.

Headline question: at the breach turn t* (and t*+1, ...), can a probe on the
victim's L16 hidden state distinguish the unsafe-payload from the benign
counterfactual? If yes, the model encodes intent. If no, it does not.

Pre-pivot turns (t < t*) have *literally identical* hidden states between
the two conditions (deterministic extraction on identical inputs), so AUC at
those turns is 0.5 by construction. We report it anyway as a validity check.

Usage:
  python -m turnstile.intent_obliteration_paired \
      --replay-pt experiments/intent_obliteration_paired/replay.pt \
      --output experiments/intent_obliteration_paired/paired_probe.json \
      --n-seeds 5
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from turnstile.intent_probe_lib import (
    seed_sweep_dense, permutation_test_dense, manifest_dense, write_json,
)


def collect_paired_at_turn(records, t: int):
    """Return (X_orig, X_trans) at turn position t, both as (N, hidden_dim)."""
    X_orig, X_trans = [], []
    for r in records:
        ohs = r.get("original_hs")
        ths = r.get("translated_hs")
        if ohs is None or ths is None:
            continue
        if t < ohs.shape[0] and t < ths.shape[0]:
            X_orig.append(ohs[t].numpy())
            X_trans.append(ths[t].numpy())
    if not X_orig:
        return None, None
    return np.stack(X_orig), np.stack(X_trans)


def collect_paired_pivot_onward(records):
    """Pool hidden states from t >= turn_of_breach across all records."""
    X_orig_pool, X_trans_pool = [], []
    per_turn_offset = []  # for analysis
    for r in records:
        tob = r["turn_of_breach"]
        ohs = r.get("original_hs"); ths = r.get("translated_hs")
        if ohs is None or ths is None:
            continue
        n = min(ohs.shape[0], ths.shape[0])
        for t in range(tob, n):
            X_orig_pool.append(ohs[t].numpy())
            X_trans_pool.append(ths[t].numpy())
            per_turn_offset.append(t - tob)
    if not X_orig_pool:
        return None, None, []
    return (np.stack(X_orig_pool), np.stack(X_trans_pool),
            np.asarray(per_turn_offset))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-pt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    args = p.parse_args()

    seeds = list(range(args.n_seeds))

    print(f"Loading replay output from {args.replay_pt}...")
    records = torch.load(args.replay_pt, weights_only=False)
    n_records = sum(1 for r in records
                    if r.get("original_hs") is not None
                    and r.get("translated_hs") is not None)
    print(f"  {n_records} records with paired hidden states")

    tob_hist = defaultdict(int)
    for r in records:
        tob_hist[int(r["turn_of_breach"])] += 1
    print(f"  turn_of_breach histogram: {dict(sorted(tob_hist.items()))}")

    results = {
        "summary": {
            "n_records": n_records,
            "tob_histogram": dict(sorted(tob_hist.items())),
            "n_seeds": args.n_seeds, "seeds": seeds,
            "layer_convention": "L16 (n_layers // 2 for 32-layer Llama-3.1-8B)",
        },
        "per_turn": {},
        "pivot_pooled": None,
    }

    # Per-turn paired probes.
    print("\nPer-turn paired AUC (original vs translation, same prelude):")
    print(f"{'turn':>4}  {'AUC':<22}  {'perm AUC':<22}  {'note'}")
    print("-" * 80)
    max_turns = max(r["original_hs"].shape[0] for r in records
                    if r.get("original_hs") is not None)
    for t in range(max_turns):
        X_orig, X_trans = collect_paired_at_turn(records, t)
        if X_orig is None:
            continue
        X = np.concatenate([X_orig, X_trans], axis=0)
        y = np.concatenate([np.ones(len(X_orig)), np.zeros(len(X_trans))])
        # Skip if pre-pivot for ALL records (deterministic identical -> AUC 0.5)
        n_at_or_after_pivot = sum(1 for r in records
                                  if t >= r["turn_of_breach"]
                                  and r.get("original_hs") is not None)
        sweep = seed_sweep_dense(X, y, seeds=seeds)
        perm = permutation_test_dense(X, y, n_perms=3, seed=0)
        man = manifest_dense(X, y)
        results["per_turn"][t] = {
            "sweep": sweep, "permutation": perm, "manifest": man,
            "n_records_at_or_after_pivot": int(n_at_or_after_pivot),
        }
        note = ("at/after pivot" if n_at_or_after_pivot > 0 else
                "ALL pre-pivot (should be 0.5 by construction)")
        print(f"  {t}   {sweep['auc_seed_mean']:.4f} +- {sweep['auc_seed_std']:.4f}  "
              f"perm = {perm['auc_mean']:.4f} +- {perm['auc_std']:.4f}  "
              f"({note}, n={int(n_at_or_after_pivot)})")

    # Pivot-onward pool.
    X_orig, X_trans, offsets = collect_paired_pivot_onward(records)
    if X_orig is not None:
        X = np.concatenate([X_orig, X_trans], axis=0)
        y = np.concatenate([np.ones(len(X_orig)), np.zeros(len(X_trans))])
        sweep = seed_sweep_dense(X, y, seeds=seeds)
        perm = permutation_test_dense(X, y, n_perms=3, seed=0)
        unique_offsets, offset_counts = np.unique(offsets, return_counts=True)
        results["pivot_pooled"] = {
            "sweep": sweep, "permutation": perm,
            "n_examples": int(len(y)),
            # cast np.int64 -> Python int so json.dump doesn't choke
            "offset_histogram": {
                int(k): int(v) for k, v in zip(unique_offsets, offset_counts)
            },
        }
        print(f"\nPivot-onward pooled: AUC = {sweep['auc_seed_mean']:.4f} "
              f"+- {sweep['auc_seed_std']:.4f}  "
              f"perm = {perm['auc_mean']:.4f} +- {perm['auc_std']:.4f}  "
              f"(n={int(len(y))})")

    write_json(results, args.output)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
