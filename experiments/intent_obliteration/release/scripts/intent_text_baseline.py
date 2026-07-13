"""Text-only baseline for the intent probe (rock-solid edition).

Companion to ``intent_obliteration.py``. The hidden-state probe at L16 hits AUC
~1.00 separating adversarial from benign-curious conversations. This script
asks: how much of that signal is recoverable from the user-turn text alone,
with no model internals at all? We use TF-IDF + logistic regression -- the
weakest possible text baseline. If this already saturates, the residual-stream
probe adds nothing beyond surface lexical features.

Multi-seed averaging, granular per-fold output, permutation-label validity test.

Usage:
  python -m turnstile.intent_text_baseline \
      --experiments experiments/{stealth_s42,...} \
      --benign-convs experiments/network_hacking_corpus/benign_curious/convs.jsonl \
      --output experiments/intent_obliteration/text_baseline.json \
      --n-seeds 5
"""

import argparse
import json
import os

import numpy as np

from turnstile.intent_probe_lib import (
    seed_sweep_text,
    permutation_test_text,
    balanced_subsample_seed_sweep_text,
    write_json,
)


def _natural_sort_key(fname: str) -> int:
    base = fname.replace(".jsonl", "")
    parts = base.split("_")
    return int(parts[-1]) if parts[-1].isdigit() else 0


def load_adversarial(experiment_dirs):
    convs = []
    for exp_dir in experiment_dirs:
        rounds_dir = os.path.join(exp_dir, "rounds")
        if not os.path.isdir(rounds_dir):
            print(f"  [skip] {exp_dir}: no rounds/")
            continue
        files = sorted(
            (f for f in os.listdir(rounds_dir)
             if f.startswith("round_") and f.endswith(".jsonl")),
            key=_natural_sort_key,
        )
        n_added = 0
        for fname in files:
            with open(os.path.join(rounds_dir, fname)) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    conv = rec["conversation"]
                    turns = [m["content"] for m in conv if m["role"] == "user"]
                    convs.append({
                        "turns": turns,
                        "label": bool(rec.get("unsafe", False)),
                        "tob": rec.get("turn_of_breach"),
                        "source": os.path.basename(exp_dir.rstrip("/")),
                    })
                    n_added += 1
        print(f"  [load] {exp_dir}: {n_added} conversations")
    return convs


def load_benign(benign_convs_path):
    convs = []
    with open(benign_convs_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            conv = rec["conversation"]
            turns = [m["content"] for m in conv if m["role"] == "user"]
            convs.append({"turns": turns})
    print(f"  [load] {benign_convs_path}: {len(convs)} benign conversations")
    return convs


def per_turn_auc(adv_convs, benign_convs, seeds):
    out = {}
    max_turns = max(
        max(len(c["turns"]) for c in adv_convs),
        max(len(c["turns"]) for c in benign_convs),
    )
    for t in range(max_turns):
        adv_t = [c["turns"][t] for c in adv_convs if t < len(c["turns"])]
        ben_t = [c["turns"][t] for c in benign_convs if t < len(c["turns"])]
        if not adv_t or not ben_t:
            continue
        texts = adv_t + ben_t
        y = np.array([1] * len(adv_t) + [0] * len(ben_t))
        sweep = seed_sweep_text(texts, y, seeds=seeds)
        perm = permutation_test_text(texts, y, n_perms=3, seed=0)
        out[t] = {"sweep": sweep, "permutation": perm}
    return out


def collect_pre_pivot_turns(adv_convs, slice_name):
    out = []
    for c in adv_convs:
        T = len(c["turns"])
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
            out.append(c["turns"][t])
    return out


def benign_all_turns(benign_convs, max_turns=5):
    out = []
    for c in benign_convs:
        out.extend(c["turns"][:max_turns])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--benign-convs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-balanced-per-class", type=int, default=500)
    args = p.parse_args()

    seeds = list(range(args.n_seeds))

    print("=" * 72)
    print("Intent text baseline: TF-IDF (1-2 grams, max 20k feats) + logreg")
    print("=" * 72)
    print("\n[load adversarial]")
    adv = load_adversarial(args.experiments)
    print("\n[load benign]")
    ben = load_benign(args.benign_convs)

    n_win = sum(1 for c in adv if c["label"])
    n_with_tob = sum(1 for c in adv if c["tob"] is not None)
    print(f"\nAdv convs: {len(adv)} ({n_win} wins, {n_with_tob} with TOB)")
    print(f"Benign convs: {len(ben)}")

    summary = {
        "n_adv_convs": len(adv), "n_wins": n_win,
        "n_with_tob": n_with_tob, "n_benign_convs": len(ben),
        "n_seeds": args.n_seeds, "seeds": seeds,
        "n_balanced_per_class": args.n_balanced_per_class,
    }
    results = {"summary": summary}

    # Per-turn AUC.
    print("\n" + "=" * 72)
    print(f"Per-turn TF-IDF AUC ({args.n_seeds} seeds)")
    print("=" * 72)
    pt = per_turn_auc(adv, ben, seeds)
    for t, v in pt.items():
        s = v["sweep"]; p_ = v["permutation"]
        print(f"  turn {t}: AUC = {s['auc_seed_mean']:.4f} +- {s['auc_seed_std']:.4f}  "
              f"(perm-label = {p_['auc_mean']:.4f} +- {p_['auc_std']:.4f})  "
              f"n_pos={s['n_pos']} n_neg={s['n_neg']}")
    results["per_turn"] = pt

    # Slice AUC.
    print("\n" + "=" * 72)
    print("Pooled TF-IDF AUC per slice (full + balanced + perm)")
    print("=" * 72)
    ben_pool = benign_all_turns(ben)
    slice_results = {}
    for slice_name in ["all", "winners_pre", "full_pre", "full_strict_pre"]:
        adv_pool = collect_pre_pivot_turns(adv, slice_name)
        texts = adv_pool + ben_pool
        y = np.array([1] * len(adv_pool) + [0] * len(ben_pool))

        full = seed_sweep_text(texts, y, seeds=seeds)
        bal = balanced_subsample_seed_sweep_text(
            adv_pool, ben_pool, n_per_class=args.n_balanced_per_class,
            seeds=seeds,
        )
        perm = permutation_test_text(texts, y, n_perms=3, seed=0)

        slice_results[slice_name] = {"full": full, "balanced": bal, "permutation": perm}
        print(f"  {slice_name:>20s}: full AUC = {full['auc_seed_mean']:.4f} "
              f"+- {full['auc_seed_std']:.4f} (n={full['n_examples']}); "
              f"balanced n={bal['n_per_class']}: AUC = {bal['auc_seed_mean']:.4f} "
              f"+- {bal['auc_seed_std']:.4f}; "
              f"perm AUC = {perm['auc_mean']:.4f} +- {perm['auc_std']:.4f}")
    results["slices"] = slice_results

    write_json(results, args.output)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
