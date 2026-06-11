"""Extract per-turn HS at arbitrary layers from original-arm conversations,
then fit compliance LR + turn-position LR + random directions at each.

Used for H3 (test whether compliance amplification appears at layers
other than 16/31 — e.g. L4, L8, L12, L20, L24).

Output: a .pt file mapping label -> {direction, info, median_h}.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from turnstile.model_utils import load_model, unload_model


def fit_lr_direction(X, y):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    w = clf.coef_[0]
    raw_norm = float(np.linalg.norm(w))
    train_acc = float(clf.score(X, y))
    return torch.tensor(w / max(raw_norm, 1e-9), dtype=torch.float32), \
           {"method": "lr", "raw_norm": raw_norm, "train_acc": train_acc}


def fit_random_direction(d_model, seed):
    rng = np.random.RandomState(seed)
    w = rng.randn(d_model).astype(np.float32)
    return torch.tensor(w / np.linalg.norm(w)), {"method": "random", "seed": seed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-pt", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--out-pt", required=True)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    print(f"loading {args.replay_pt}")
    records = torch.load(args.replay_pt, weights_only=False)
    print(f"  {len(records)} records")

    # Reproduce B's split (seed=13, by category)
    by_cat = defaultdict(list)
    for i, r in enumerate(records):
        by_cat[r.get("category")].append(i)
    rng = random.Random(args.seed)
    train_idx = []
    for cat, idxs in by_cat.items():
        rng.shuffle(idxs)
        train_idx.extend(idxs[:len(idxs)//2])
    train_idx.sort()
    print(f"  train split: {len(train_idx)} records")

    print(f"loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)

    # For each train record, walk per-turn user-token positions and
    # capture HS at each requested layer.
    by_layer_hs = {L: [] for L in args.layers}
    by_layer_lab = {L: [] for L in args.layers}
    by_layer_pos = {L: [] for L in args.layers}  # turn position (for turn-pos LR)

    print(f"[extract] {len(train_idx)} train records × all turns × {len(args.layers)} layers...")
    t0 = time.time()
    for k, ri in enumerate(train_idx):
        r = records[ri]
        oc = r.get("original_conversation") or []
        labels = r.get("original_per_turn_labels") or []
        n_turns = (len(oc) + 1) // 2
        for t in range(n_turns):
            if t >= len(labels): break
            lab = labels[t]
            if lab is None: continue
            unsafe = lab.get("unsafe") if isinstance(lab, dict) else lab
            if unsafe is None: continue
            prefix = oc[:2*t + 1]
            templated = vic_tok.apply_chat_template(
                prefix, tokenize=False, add_generation_prompt=True,
            )
            inputs = vic_tok(templated, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                fwd = vic_model(**inputs, output_hidden_states=True, use_cache=False)
            for L in args.layers:
                h = fwd.hidden_states[L + 1][0, -1, :].detach().to("cpu", torch.float32)
                by_layer_hs[L].append(h)
                by_layer_lab[L].append(int(bool(unsafe)))
                by_layer_pos[L].append(t)
            del fwd
        if (k+1) % 25 == 0:
            print(f"[extract] {k+1}/{len(train_idx)}  ({time.time()-t0:.0f}s)")

    print(f"[extract] done in {time.time()-t0:.0f}s")

    out = {}
    for L in args.layers:
        X = torch.stack(by_layer_hs[L]).numpy()
        y = np.array(by_layer_lab[L], dtype=np.int64)
        pos = np.array(by_layer_pos[L], dtype=np.int64)
        median_h = float(torch.norm(torch.tensor(X), dim=1).median())
        # lr_compliance
        d, info = fit_lr_direction(X, y)
        out[f"L{L}_lr_compliance"] = {"direction": d, "info": info,
                                       "median_h": median_h}
        print(f"L{L} lr_compliance  raw_norm={info['raw_norm']:.3f}  acc={info['train_acc']:.3f}  "
              f"n_compliant={int(y.sum())} n_refusal={int((y==0).sum())}  median_h={median_h:.2f}")
        # lr_turn_pos
        if len(set(pos >= 2)) == 2:
            yp = (pos >= 2).astype(np.int64)
            d, info = fit_lr_direction(X, yp)
            out[f"L{L}_lr_turn_pos"] = {"direction": d, "info": info,
                                         "median_h": median_h}
            print(f"L{L} lr_turn_pos    raw_norm={info['raw_norm']:.3f}  acc={info['train_acc']:.3f}")
        # random
        d_model = X.shape[1]
        d, info = fit_random_direction(d_model, seed=42 + L)
        out[f"L{L}_random"] = {"direction": d, "info": info, "median_h": median_h}

    torch.save(out, args.out_pt)
    print(f"wrote {args.out_pt}")
    unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
