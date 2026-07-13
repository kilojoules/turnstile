"""Intent-probe stress test via unsafe-turn obliteration (rock-solid edition).

Paper claim under test: a logistic probe on victim residual stream activations at
layer 16 separates adversarial from benign conversations at AUC ~0.97. We worry
this is dominated by the harmful payload turns; the trust-building prelude turns
of an adversarial conversation are surface-benign by design.

This run applies four slices to the adversarial side:
  - all              : every turn (the published baseline)
  - winners_pre      : winners only, t < turn_of_breach
  - full_pre         : winners pre-pivot + losers all turns
  - full_strict_pre  : full_pre but drop the last turn of losers

Each slice is evaluated:
  - over multiple CV seeds (default 5)
  - both with full-pool and balanced-subsample sampling
  - with a permutation-label validity test (random labels => AUC ~ 0.5)

Granular per-seed/per-fold AUCs are saved to JSON.

Usage:
  python -m turnstile.intent_obliteration \
      --experiments experiments/{stealth_s42,control_s42,...} \
      --benign-hs experiments/network_hacking_corpus/benign_curious/hs.pt \
      --output experiments/intent_obliteration/results.json \
      --n-seeds 5
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch

from turnstile.intent_probe_lib import (
    auc_cv_dense,
    seed_sweep_dense,
    permutation_test_dense,
    balanced_subsample_seed_sweep_dense,
    manifest_dense,
    write_json,
)


def _natural_sort_key(fname: str) -> int:
    base = fname.replace(".pt", "")
    parts = base.split("_")
    return int(parts[-1]) if parts[-1].isdigit() else 0


def load_adversarial(experiment_dirs: Iterable[str]):
    convs = []
    sources = defaultdict(int)
    for exp_dir in experiment_dirs:
        hs_dir = os.path.join(exp_dir, "hidden_states")
        if not os.path.isdir(hs_dir):
            print(f"  [skip] {exp_dir}: no hidden_states/")
            continue
        files = sorted(
            (f for f in os.listdir(hs_dir)
             if f.startswith("round_") and f.endswith(".pt")),
            key=_natural_sort_key,
        )
        n_added = 0
        for fname in files:
            data = torch.load(os.path.join(hs_dir, fname), weights_only=False)
            for hs, lab, tob in zip(
                data["hidden_states"], data["labels"], data["turns_of_breach"]
            ):
                if hs is None:
                    continue
                convs.append({
                    "hs": hs, "label": bool(lab),
                    "tob": tob if tob is not None else None,
                    "source": os.path.basename(exp_dir.rstrip("/")),
                })
                n_added += 1
        sources[os.path.basename(exp_dir.rstrip("/"))] += n_added
        print(f"  [load] {exp_dir}: {n_added} conversations")
    return convs, dict(sources)


def load_benign(benign_hs_path: str):
    data = torch.load(benign_hs_path, weights_only=False)
    convs = [{"hs": hs} for hs in data["hidden_states"] if hs is not None]
    print(f"  [load] {benign_hs_path}: {len(convs)} benign conversations")
    return convs


def collect_pre_pivot_turns(adv_convs, slice_name):
    out = []
    for c in adv_convs:
        T = c["hs"].shape[0]
        is_win = c["label"]
        tob = c["tob"]
        if slice_name == "all":
            keep = list(range(T))
        elif slice_name == "winners_pre":
            if not is_win or tob is None:
                continue
            keep = list(range(min(tob, T)))
        elif slice_name == "full_pre":
            if is_win and tob is not None:
                keep = list(range(min(tob, T)))
            else:
                keep = list(range(T))
        elif slice_name == "full_strict_pre":
            if is_win and tob is not None:
                keep = list(range(min(tob, T)))
            else:
                keep = list(range(max(0, T - 1)))
        else:
            raise ValueError(slice_name)
        for t in keep:
            out.append((c, t))
    return out


def stack_turns(adv_turn_pairs, benign_convs, max_turns=5):
    X_pos = (np.stack([c["hs"][t].numpy() for c, t in adv_turn_pairs])
             if adv_turn_pairs else np.empty((0, 4096)))
    X_neg_list = []
    for c in benign_convs:
        for t in range(min(max_turns, c["hs"].shape[0])):
            X_neg_list.append(c["hs"][t].numpy())
    X_neg = np.stack(X_neg_list) if X_neg_list else np.empty((0, 4096))
    return X_pos, X_neg


def per_turn_auc(adv_convs, benign_convs, seeds):
    out = {}
    max_turns = max(c["hs"].shape[0] for c in adv_convs + benign_convs)
    for t in range(max_turns):
        X_adv = [c["hs"][t].numpy() for c in adv_convs if t < c["hs"].shape[0]]
        X_ben = [c["hs"][t].numpy() for c in benign_convs if t < c["hs"].shape[0]]
        if not X_adv or not X_ben:
            continue
        X_adv = np.stack(X_adv)
        X_ben = np.stack(X_ben)
        X = np.concatenate([X_adv, X_ben], axis=0)
        y = np.concatenate([np.ones(len(X_adv)), np.zeros(len(X_ben))])
        sweep = seed_sweep_dense(X, y, seeds=seeds)
        perm = permutation_test_dense(X, y, n_perms=3, seed=0)
        out[t] = {"sweep": sweep, "permutation": perm}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--benign-hs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-balanced-per-class", type=int, default=500)
    args = p.parse_args()

    seeds = list(range(args.n_seeds))

    print("=" * 72)
    print("Intent-obliteration: hidden-state probe at layer 16")
    print("=" * 72)
    print("\n[load adversarial]")
    adv, sources = load_adversarial(args.experiments)
    print("\n[load benign]")
    ben = load_benign(args.benign_hs)

    n_win = sum(1 for c in adv if c["label"])
    n_with_tob = sum(1 for c in adv if c["tob"] is not None)
    tob_hist = defaultdict(int)
    for c in adv:
        if c["tob"] is not None:
            tob_hist[int(c["tob"])] += 1

    summary = {
        "n_adv_convs": len(adv),
        "n_wins": n_win,
        "n_with_turn_of_breach": n_with_tob,
        "turn_of_breach_histogram": dict(sorted(tob_hist.items())),
        "sources": sources,
        "n_benign_convs": len(ben),
        "n_seeds": args.n_seeds,
        "seeds": seeds,
        "n_balanced_per_class": args.n_balanced_per_class,
    }
    print("\n[summary]")
    print(f"  adv convs:          {summary['n_adv_convs']}")
    print(f"  wins:               {summary['n_wins']}")
    print(f"  with TOB:           {summary['n_with_turn_of_breach']}")
    print(f"  TOB histogram:      {summary['turn_of_breach_histogram']}")
    print(f"  benign convs:       {summary['n_benign_convs']}")
    print(f"  hidden state shape: {adv[0]['hs'].shape} dtype={adv[0]['hs'].dtype}")
    print(f"  inferred layer:    16 (default `extract_hidden_states` -> n_layers//2 for 32-layer victim)")

    results = {"summary": summary}

    # Per-turn AUC.
    print("\n" + "=" * 72)
    print("Per-turn AUC (all adv vs all benign), N seeds = " + str(args.n_seeds))
    print("=" * 72)
    pt = per_turn_auc(adv, ben, seeds)
    for t, v in pt.items():
        s = v["sweep"]
        p_ = v["permutation"]
        print(f"  turn {t}: AUC = {s['auc_seed_mean']:.4f} +- {s['auc_seed_std']:.4f}  "
              f"(perm-label AUC = {p_['auc_mean']:.4f} +- {p_['auc_std']:.4f})  "
              f"n_pos={s['n_pos']} n_neg={s['n_neg']}")
    results["per_turn"] = pt

    # Pooled per-slice.
    print("\n" + "=" * 72)
    print("Pooled AUC per slice (full pool + balanced subsample + perm test)")
    print("=" * 72)
    slice_results = {}
    for slice_name in ["all", "winners_pre", "full_pre", "full_strict_pre"]:
        adv_turns = collect_pre_pivot_turns(adv, slice_name)
        X_pos, X_neg = stack_turns(adv_turns, ben)
        if len(X_pos) == 0 or len(X_neg) == 0:
            slice_results[slice_name] = None
            print(f"  {slice_name}: insufficient data")
            continue

        X = np.concatenate([X_pos, X_neg], axis=0)
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

        manifest = manifest_dense(X, y)
        full = seed_sweep_dense(X, y, seeds=seeds)
        bal = balanced_subsample_seed_sweep_dense(
            X_pos, X_neg, n_per_class=args.n_balanced_per_class, seeds=seeds,
        )
        perm = permutation_test_dense(X, y, n_perms=3, seed=0)

        slice_results[slice_name] = {
            "manifest": manifest,
            "full": full,
            "balanced": bal,
            "permutation": perm,
        }
        print(f"  {slice_name:>20s}: full AUC = {full['auc_seed_mean']:.4f} "
              f"+- {full['auc_seed_std']:.4f} (n={full['n_examples']}); "
              f"balanced n={bal['n_per_class']}: AUC = {bal['auc_seed_mean']:.4f} "
              f"+- {bal['auc_seed_std']:.4f}; "
              f"perm-label AUC = {perm['auc_mean']:.4f} +- {perm['auc_std']:.4f}")
    results["slices"] = slice_results

    write_json(results, args.output)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
