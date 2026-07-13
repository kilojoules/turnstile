"""Topic-matched control: nethack malicious vs benign-curious (rock-solid edition).

Both sides are single-topic (network security), so any AUC gap is not
explained by topic. Compares TF-IDF on user text against the L16 hidden-state
probe at each turn position. Multi-seed averaging, granular per-fold output,
permutation-label validity test.

Usage:
  python -m turnstile.intent_topic_matched \
      --corpus-dir experiments/network_hacking_corpus \
      --output experiments/intent_obliteration/topic_matched.json \
      --n-seeds 5
"""

import argparse
import json
import os

import numpy as np
import torch

from turnstile.intent_probe_lib import (
    seed_sweep_dense,
    seed_sweep_text,
    permutation_test_dense,
    permutation_test_text,
    manifest_dense,
    write_json,
)


def load_user_turns(jsonl_path):
    convs = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            turns = [m["content"] for m in rec["conversation"] if m["role"] == "user"]
            convs.append(turns)
    return convs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", default="experiments/network_hacking_corpus")
    p.add_argument("--output", required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    args = p.parse_args()

    seeds = list(range(args.n_seeds))

    mal_text = load_user_turns(os.path.join(args.corpus_dir, "malicious", "convs.jsonl"))
    ben_text = load_user_turns(os.path.join(args.corpus_dir, "benign_curious", "convs.jsonl"))
    mal_hs = torch.load(os.path.join(args.corpus_dir, "malicious", "hs.pt"),
                        weights_only=False)["hidden_states"]
    ben_hs = torch.load(os.path.join(args.corpus_dir, "benign_curious", "hs.pt"),
                        weights_only=False)["hidden_states"]

    print("=" * 72)
    print("Topic-matched: nethack malicious vs benign-curious")
    print("=" * 72)
    print(f"Malicious: {len(mal_text)} convs, hs shape per conv: {mal_hs[0].shape}")
    print(f"Benign-curious: {len(ben_text)} convs, hs shape per conv: {ben_hs[0].shape}")
    print(f"N seeds: {args.n_seeds}")
    print()

    results = {
        "summary": {
            "n_mal_convs": len(mal_text), "n_ben_convs": len(ben_text),
            "n_seeds": args.n_seeds, "seeds": seeds,
            "hidden_state_layer": "16 (default n_layers//2 for 32-layer Llama-3.1-8B)",
            "hidden_state_dim": int(mal_hs[0].shape[-1]),
        },
        "per_turn": {},
        "pooled": {},
    }

    print(f"{'turn':>4}  {'TF-IDF AUC':<24} {'L16 AUC':<24}  {'TF-IDF perm':<18} {'L16 perm':<18}")
    print("-" * 100)
    for t in range(5):
        # text
        adv_t = [c[t] for c in mal_text if t < len(c)]
        ben_t = [c[t] for c in ben_text if t < len(c)]
        txt = adv_t + ben_t
        y = np.array([1] * len(adv_t) + [0] * len(ben_t))
        tfidf = seed_sweep_text(txt, y, seeds=seeds)
        tfidf_perm = permutation_test_text(txt, y, n_perms=3, seed=0)

        # hidden states
        Xa = np.stack([h[t].numpy() for h in mal_hs if t < h.shape[0]])
        Xb = np.stack([h[t].numpy() for h in ben_hs if t < h.shape[0]])
        X = np.concatenate([Xa, Xb], axis=0)
        y_hs = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))])
        l16 = seed_sweep_dense(X, y_hs, seeds=seeds)
        l16_perm = permutation_test_dense(X, y_hs, n_perms=3, seed=0)
        man = manifest_dense(X, y_hs)

        results["per_turn"][t] = {
            "tfidf": tfidf, "tfidf_permutation": tfidf_perm,
            "l16": l16, "l16_permutation": l16_perm,
            "manifest": man,
        }
        print(f"  {t}   "
              f"{tfidf['auc_seed_mean']:.4f} +- {tfidf['auc_seed_std']:.4f}    "
              f"{l16['auc_seed_mean']:.4f} +- {l16['auc_seed_std']:.4f}    "
              f"{tfidf_perm['auc_mean']:.4f} +- {tfidf_perm['auc_std']:.4f}    "
              f"{l16_perm['auc_mean']:.4f} +- {l16_perm['auc_std']:.4f}")

    # pooled
    all_txt = []
    y_pool = []
    for c in mal_text:
        all_txt.extend(c); y_pool.extend([1] * len(c))
    for c in ben_text:
        all_txt.extend(c); y_pool.extend([0] * len(c))
    y_pool = np.array(y_pool)
    tfidf_pool = seed_sweep_text(all_txt, y_pool, seeds=seeds)
    tfidf_pool_perm = permutation_test_text(all_txt, y_pool, n_perms=3, seed=0)

    all_hs = torch.cat(list(mal_hs) + list(ben_hs), dim=0).numpy()
    n_mal_total = sum(h.shape[0] for h in mal_hs)
    n_ben_total = sum(h.shape[0] for h in ben_hs)
    y_hs_pool = np.concatenate([np.ones(n_mal_total), np.zeros(n_ben_total)])
    l16_pool = seed_sweep_dense(all_hs, y_hs_pool, seeds=seeds)
    l16_pool_perm = permutation_test_dense(all_hs, y_hs_pool, n_perms=3, seed=0)

    results["pooled"] = {
        "tfidf": tfidf_pool, "tfidf_permutation": tfidf_pool_perm,
        "l16": l16_pool, "l16_permutation": l16_pool_perm,
    }
    print(f"\nPooled: TF-IDF {tfidf_pool['auc_seed_mean']:.4f} "
          f"+- {tfidf_pool['auc_seed_std']:.4f}    "
          f"L16 {l16_pool['auc_seed_mean']:.4f} +- {l16_pool['auc_seed_std']:.4f}    "
          f"(perm: TF-IDF {tfidf_pool_perm['auc_mean']:.4f}, "
          f"L16 {l16_pool_perm['auc_mean']:.4f})")

    write_json(results, args.output)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
