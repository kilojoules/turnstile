"""Outcome prediction using SAE features vs raw hidden states.

Encodes pooled L16 hidden states through the frozen_v1 SAE (16384 features),
then trains a logistic probe on SAE activations to predict jailbreak outcome.
Compares to raw-HS baseline and prints per-(turn, category) AUCs.
"""
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from turnstile.probe import SparseAutoencoder, normalize_activations

POOLED_DIR = "experiments/pooled_hs"
SAE_PATH = "results/probe/frozen_v1/sae.pt"
OUT_JSON = "experiments/outcome_probe_v1/sae_outcome_pooled.json"
LAYER = 16  # SAE was trained on L16


def load_sae():
    ckpt = torch.load(SAE_PATH, weights_only=False)
    sae = SparseAutoencoder(ckpt["d_model"], ckpt["n_features"],
                             l1_coeff=ckpt.get("l1_coeff", 5.0))
    sae.load_state_dict(ckpt["sae_state_dict"])
    sae.eval()
    norm_scale = ckpt.get("normalize_scale", None)
    return sae, norm_scale


def encode_turn(sae, norm_scale, turn, batch_size=512):
    """Encode all pooled L16 hidden states at a given turn. Returns (features, y, cats)."""
    hs_list, y, cats = [], [], []
    for src in sorted(os.listdir(POOLED_DIR)):
        sdir = os.path.join(POOLED_DIR, src)
        if not os.path.isdir(sdir):
            continue
        for f in sorted(os.listdir(sdir)):
            if not f.endswith(".pt"):
                continue
            data = torch.load(os.path.join(sdir, f), weights_only=False)
            arr = data["hidden_states_by_layer"][LAYER].numpy()  # (n, 5, 4096)
            labels = data["labels"].tolist()
            src_cats = data["categories"]
            for i in range(len(labels)):
                hs_list.append(arr[i, turn])
                y.append(int(bool(labels[i])))
                cats.append(src_cats[i])
            del data, arr
    gc.collect()
    X = np.stack(hs_list)  # (N, 4096)

    # Normalize + encode in batches
    X_t = torch.from_numpy(X).float()
    if norm_scale is not None:
        # Use stored training-set scale
        X_t = X_t * norm_scale
    # else: per-batch normalize using probe.normalize_activations (unlikely path)
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            chunk = X_t[i:i + batch_size]
            f = sae.encode(chunk)
            feats.append(f.numpy().astype(np.float32))
    feats = np.concatenate(feats, axis=0)
    return feats, np.array(y), np.array(cats)


def kfold_auc(X, y, n_splits=5, seed=42):
    if len(np.unique(y)) < 2:
        return None
    k = min(n_splits, int(y.sum()), int(len(y) - y.sum()))
    if k < 2:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    if not aucs:
        return None
    return float(np.mean(aucs)), float(np.std(aucs))


def sparsity_stats(feats):
    """Return mean L0, mean nonzero count."""
    nonzero = (feats > 1e-6)
    l0 = nonzero.sum(axis=1).mean()
    pct = 100.0 * nonzero.any(axis=0).mean()
    return float(l0), float(pct)


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    sae, norm_scale = load_sae()
    print(f"Loaded SAE: {sae.d_model} -> {sae.n_features}, "
          f"normalize_scale={norm_scale}")
    print(f"Layer: {LAYER}, pooled data from {POOLED_DIR}\n")

    results = {"global": {}, "per_category": {}, "sparsity": {}}

    for T in range(5):
        print(f"=== Turn {T} ===")
        feats, y, cats = encode_turn(sae, norm_scale, T)
        n = len(y)
        l0, pct = sparsity_stats(feats)
        print(f"  {n} convs  L0={l0:.1f}  active_feats={pct:.1f}%")
        results["sparsity"][f"T{T}"] = {"mean_l0": l0, "pct_active_feats": pct}

        # Global
        r = kfold_auc(feats, y)
        if r:
            results["global"][f"T{T}"] = {"auc": r[0], "std": r[1], "n": n,
                                          "n_pos": int(y.sum())}
            print(f"  Global AUC: {r[0]:.4f} ± {r[1]:.4f}")

        # Per-category
        for cat in sorted(set(cats.tolist())):
            mask = cats == cat
            Xc, yc = feats[mask], y[mask]
            rc = kfold_auc(Xc, yc)
            if rc is not None:
                results["per_category"].setdefault(cat, {})[f"T{T}"] = {
                    "auc": rc[0], "std": rc[1],
                    "n": int(mask.sum()), "n_pos": int(yc.sum())
                }
        del feats, y, cats
        gc.collect()

    print(f"\n{'Category':<28s}  " + "  ".join(f"T{t}" for t in range(5)))
    print("-" * 70)
    for cat in sorted(results["per_category"]):
        row = [cat[:28].ljust(28)]
        for T in range(5):
            v = results["per_category"][cat].get(f"T{T}")
            row.append(f"{v['auc']:.3f}" if v else "  -  ")
        print("  ".join(row))

    # Compare to raw HS at L16
    try:
        raw = json.load(open("experiments/outcome_probe_v1/pooled_layer_turn_sweep.json"))
        raw_l16 = {f"T{t}": raw["matrix"].get(f"L{LAYER}_T{t}") for t in range(5)}
        print("\n=== SAE vs Raw L16 (global) ===")
        print(f"{'Turn':<4s}   {'Raw':>6s}   {'SAE':>6s}   {'Δ':>6s}")
        for T in range(5):
            r = raw_l16[f"T{T}"]
            s = results["global"].get(f"T{T}")
            if r and s:
                delta = s["auc"] - r["auc"]
                print(f"T{T}     {r['auc']:.3f}   {s['auc']:.3f}   {delta:+.3f}")
    except Exception as e:
        print(f"(could not load raw comparison: {e})")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
