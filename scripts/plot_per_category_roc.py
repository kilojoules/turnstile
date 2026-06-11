"""Per-category ROC curves on the 9,400-conv corpus.

For each of the 10 JBB categories:
  - load per-turn rows at L16
  - StratifiedGroupKFold (5-fold by conversation_id) — no leakage
  - collect out-of-fold predictions
  - compute ROC from those OOB scores + per-turn labels

Plot all 10 curves on one figure with per-category AUC in the legend.
"""
import gc
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYER = 16


def load_layer_rows(layer):
    Xs, ys, cs, gs = [], [], [], []
    cid = 0
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            labels = data["labels"].tolist()
            tobs = data["turns_of_breach"]
            cats = data["categories"]
            for i in range(len(labels)):
                breach = bool(labels[i])
                t_star = tobs[i] if breach else None
                if breach and t_star is None:
                    continue
                t_max = 4 if not breach else int(t_star)
                conv_id = cid; cid += 1
                for t in range(t_max + 1):
                    Xs.append(arr[i, t])
                    ys.append(1 if (breach and t == t_star) else 0)
                    cs.append(cats[i]); gs.append(conv_id)
            del data, arr
    gc.collect()
    return np.stack(Xs), np.array(ys), np.array(cs), np.array(gs)


def oof_predictions(X, y, groups, n_splits=5, seed=42):
    pos_groups = np.unique(groups[y == 1])
    neg_groups = np.unique(groups[y == 0])
    k = min(n_splits, len(pos_groups), len(neg_groups))
    if k < 2:
        return None
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in sgkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def main():
    print(f"Loading L{LAYER} ...", flush=True)
    X, y, cats, groups = load_layer_rows(LAYER)
    print(f"  rows: {len(y)}, positives: {int(y.sum())}, "
          f"unique categories: {len(np.unique(cats))}", flush=True)

    rocs = {}
    for cat in sorted(np.unique(cats)):
        mask = cats == cat
        Xc, yc, gc_ = X[mask], y[mask], groups[mask]
        oof = oof_predictions(Xc, yc, gc_)
        if oof is None:
            continue
        keep = ~np.isnan(oof)
        fpr, tpr, _ = roc_curve(yc[keep], oof[keep])
        auc = roc_auc_score(yc[keep], oof[keep])
        rocs[cat] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                     "auc": float(auc),
                     "n": int(mask.sum()), "n_pos": int(yc.sum())}
        print(f"  {cat:<28}  AUC={auc:.4f}  n={int(mask.sum())}, "
              f"pos={int(yc.sum())}", flush=True)

    # also pooled
    oof_all = oof_predictions(X, y, groups)
    keep = ~np.isnan(oof_all)
    fpr_all, tpr_all, _ = roc_curve(y[keep], oof_all[keep])
    auc_all = roc_auc_score(y[keep], oof_all[keep])
    print(f"\n  {'POOLED':<28}  AUC={auc_all:.4f}  n={len(y)}, "
          f"pos={int(y.sum())}", flush=True)

    out = {"layer": LAYER, "per_category": rocs,
           "pooled": {"fpr": fpr_all.tolist(), "tpr": tpr_all.tolist(),
                      "auc": float(auc_all),
                      "n": len(y), "n_pos": int(y.sum())}}
    with open(f"{OUT}/per_category_roc_L{LAYER}.json", "w") as f:
        json.dump(out, f)
    print(f"\nWrote {OUT}/per_category_roc_L{LAYER}.json")

    # ----- plot -----
    cats_sorted = sorted(rocs.keys(), key=lambda c: -rocs[c]["auc"])
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    for i, c in enumerate(cats_sorted):
        r = rocs[c]
        ax.plot(r["fpr"], r["tpr"], linewidth=1.6, color=cmap(i % 10),
                label=f"{c}: AUC = {r['auc']:.3f} (n_pos={r['n_pos']})")
    # pooled
    ax.plot(fpr_all, tpr_all, linewidth=2.2, color="black", linestyle="--",
            label=f"pooled: AUC = {auc_all:.3f} (n_pos={int(y.sum())})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8,
            alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"Per-category ROC at L{LAYER}, per-turn label, "
                 f"GroupKFold by conversation\n"
                 f"(9,400-conv corpus, OOB predictions; n_pos = breach turns per category)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.93)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/per_category_roc_L{LAYER}.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
