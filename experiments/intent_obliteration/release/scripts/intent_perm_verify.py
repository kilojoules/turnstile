"""High-resolution permutation-label validity check for the intent-obliteration
analyses.

The main scripts run n_perms=3 because the headline path runs full-pool CV with
4096-dim activations, and the permutation test re-fits the same probe many
times. n_perms=50 on a balanced 500/class subsample takes seconds, gives a
tighter null distribution, and is the right standard for a paper-quality
validity check.

Outputs are written to a separate JSON so the headline files are unchanged.

Usage:
  python -m turnstile.intent_perm_verify \
      --experiments experiments/{...} \
      --benign-hs experiments/network_hacking_corpus/benign_curious/hs.pt \
      --benign-convs experiments/network_hacking_corpus/benign_curious/convs.jsonl \
      --corpus-dir experiments/network_hacking_corpus \
      --output experiments/intent_obliteration/perm_verify.json \
      --n-perms 50 --n-per-class 500
"""

import argparse
import json
import os

import numpy as np
import torch

from turnstile.intent_probe_lib import (
    auc_cv_dense, auc_cv_text, write_json,
)
from turnstile.intent_obliteration import (
    load_adversarial as load_adv_hs,
    load_benign as load_ben_hs,
    collect_pre_pivot_turns as collect_pp_hs,
    stack_turns,
)
from turnstile.intent_text_baseline import (
    load_adversarial as load_adv_txt,
    load_benign as load_ben_txt,
    collect_pre_pivot_turns as collect_pp_txt,
    benign_all_turns,
)


def perm_balanced_dense(X_pos, X_neg, n_per_class, n_perms, seed=0):
    """Permutation test on a balanced subsample of dense features."""
    rng = np.random.default_rng(seed)
    n = min(n_per_class, len(X_pos), len(X_neg))
    aucs = []
    for i in range(n_perms):
        ai = rng.choice(len(X_pos), n, replace=False)
        bi = rng.choice(len(X_neg), n, replace=False)
        X = np.concatenate([X_pos[ai], X_neg[bi]], axis=0)
        y = np.concatenate([np.ones(n), np.zeros(n)])
        rng.shuffle(y)
        cv_seed = int(rng.integers(0, 2**31 - 1))
        r = auc_cv_dense(X, y, seed=cv_seed)
        aucs.append(r["mean"])
    return {
        "n_perms": n_perms, "n_per_class": int(n),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_p2_5": float(np.percentile(aucs, 2.5)),
        "auc_p97_5": float(np.percentile(aucs, 97.5)),
        "per_perm": aucs,
    }


def perm_balanced_text(texts_pos, texts_neg, n_per_class, n_perms, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n_per_class, len(texts_pos), len(texts_neg))
    aucs = []
    for i in range(n_perms):
        ai = rng.choice(len(texts_pos), n, replace=False)
        bi = rng.choice(len(texts_neg), n, replace=False)
        texts = [texts_pos[i] for i in ai] + [texts_neg[i] for i in bi]
        y = np.concatenate([np.ones(n), np.zeros(n)])
        rng.shuffle(y)
        cv_seed = int(rng.integers(0, 2**31 - 1))
        r = auc_cv_text(texts, y, seed=cv_seed)
        aucs.append(r["mean"])
    return {
        "n_perms": n_perms, "n_per_class": int(n),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_p2_5": float(np.percentile(aucs, 2.5)),
        "auc_p97_5": float(np.percentile(aucs, 97.5)),
        "per_perm": aucs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--benign-hs", required=True)
    p.add_argument("--benign-convs", required=True)
    p.add_argument("--corpus-dir", default="experiments/network_hacking_corpus")
    p.add_argument("--output", required=True)
    p.add_argument("--n-perms", type=int, default=50)
    p.add_argument("--n-per-class", type=int, default=500)
    args = p.parse_args()

    out = {"params": {"n_perms": args.n_perms, "n_per_class": args.n_per_class}}

    # --- hidden-state side ---
    print("[load adversarial hs]")
    adv_hs, _ = load_adv_hs(args.experiments)
    print("[load benign hs]")
    ben_hs = load_ben_hs(args.benign_hs)

    print("\n[hidden-state permutation tests, balanced 500/class, 50 perms]")
    out["hidden_state_slices"] = {}
    for slc in ["all", "winners_pre", "full_pre", "full_strict_pre"]:
        adv_turns = collect_pp_hs(adv_hs, slc)
        X_pos, X_neg = stack_turns(adv_turns, ben_hs)
        if len(X_pos) == 0 or len(X_neg) == 0:
            out["hidden_state_slices"][slc] = None
            continue
        r = perm_balanced_dense(X_pos, X_neg, args.n_per_class, args.n_perms,
                                seed=0)
        out["hidden_state_slices"][slc] = r
        print(f"  {slc:>20s}: AUC = {r['auc_mean']:.4f} +- {r['auc_std']:.4f}  "
              f"95% CI [{r['auc_p2_5']:.4f}, {r['auc_p97_5']:.4f}]  "
              f"(n_per_class={r['n_per_class']})")

    # --- text-only side ---
    print("\n[load adversarial text]")
    adv_txt = load_adv_txt(args.experiments)
    print("[load benign text]")
    ben_txt = load_ben_txt(args.benign_convs)
    ben_pool = benign_all_turns(ben_txt)

    print("\n[text-only permutation tests, balanced 500/class, 50 perms]")
    out["text_only_slices"] = {}
    for slc in ["all", "winners_pre", "full_pre", "full_strict_pre"]:
        adv_pool = collect_pp_txt(adv_txt, slc)
        r = perm_balanced_text(adv_pool, ben_pool, args.n_per_class, args.n_perms,
                               seed=0)
        out["text_only_slices"][slc] = r
        print(f"  {slc:>20s}: AUC = {r['auc_mean']:.4f} +- {r['auc_std']:.4f}  "
              f"95% CI [{r['auc_p2_5']:.4f}, {r['auc_p97_5']:.4f}]  "
              f"(n_per_class={r['n_per_class']})")

    # --- topic-matched ---
    print("\n[load topic-matched data]")
    mal_hs = torch.load(os.path.join(args.corpus_dir, "malicious", "hs.pt"),
                        weights_only=False)["hidden_states"]
    ben_hs2 = torch.load(os.path.join(args.corpus_dir, "benign_curious", "hs.pt"),
                         weights_only=False)["hidden_states"]
    Xa = np.concatenate([h.numpy() for h in mal_hs], axis=0)
    Xb = np.concatenate([h.numpy() for h in ben_hs2], axis=0)
    print(f"  topic-matched HS pool: {len(Xa)} mal vs {len(Xb)} ben")

    def load_user_turns(jsonl_path):
        out_ = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                out_.extend(m["content"] for m in rec["conversation"]
                            if m["role"] == "user")
        return out_
    mal_txt = load_user_turns(os.path.join(args.corpus_dir, "malicious", "convs.jsonl"))
    ben_txt2 = load_user_turns(os.path.join(args.corpus_dir, "benign_curious", "convs.jsonl"))

    print("\n[topic-matched permutation tests, 50 perms]")
    n = min(args.n_per_class, len(Xa), len(Xb))
    r_l16 = perm_balanced_dense(Xa, Xb, args.n_per_class, args.n_perms, seed=0)
    r_tfidf = perm_balanced_text(mal_txt, ben_txt2, args.n_per_class, args.n_perms, seed=0)
    out["topic_matched_pooled"] = {"l16": r_l16, "tfidf": r_tfidf}
    print(f"  L16:    AUC = {r_l16['auc_mean']:.4f} +- {r_l16['auc_std']:.4f}  "
          f"95% CI [{r_l16['auc_p2_5']:.4f}, {r_l16['auc_p97_5']:.4f}]")
    print(f"  TF-IDF: AUC = {r_tfidf['auc_mean']:.4f} +- {r_tfidf['auc_std']:.4f}  "
          f"95% CI [{r_tfidf['auc_p2_5']:.4f}, {r_tfidf['auc_p97_5']:.4f}]")

    write_json(out, args.output)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
