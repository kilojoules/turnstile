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


def render(rocs, pooled, layer):
    """Professional, large-text render from ROC dicts (per_category + pooled)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 17, "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#4d4d4d", "axes.linewidth": 1.1,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    PALETTE = ["#3b6ea5", "#e07b39", "#4f9d69", "#c0455a", "#7d6bb0",
               "#8f6f57", "#cf7fb3", "#c9a441", "#4bacc6", "#7f7f7f"]
    cats_sorted = sorted(rocs.keys(), key=lambda c: -rocs[c]["auc"])

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    ax.set_axisbelow(True)
    ax.grid(True, color="#e9e9e9", linewidth=1.0)
    # chance diagonal
    ax.plot([0, 1], [0, 1], color="#b0b0b0", linestyle=(0, (2, 3)), linewidth=1.3)
    ax.text(0.62, 0.57, "chance", color="#9a9a9a", fontsize=13, rotation=39,
            rotation_mode="anchor", ha="left", va="center")

    handles = []
    for i, c in enumerate(cats_sorted):
        r = rocs[c]
        (h,) = ax.plot(r["fpr"], r["tpr"], lw=2.2, color=PALETTE[i % len(PALETTE)],
                       solid_capstyle="round", label=f"{c}   {r['auc']:.3f}")
        handles.append(h)
    (hp,) = ax.plot(pooled["fpr"], pooled["tpr"], "--", lw=3.4, color="#1a1a1a",
                    zorder=10, label=f"pooled   {pooled['auc']:.3f}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.003); ax.set_aspect("equal")
    ax.set_xlabel("false-positive rate", labelpad=9)
    ax.set_ylabel("true-positive rate", labelpad=9)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.tick_params(length=0)

    ax.text(0.0, 1.115, f"Compliance decodes uniformly across categories (L{layer})",
            transform=ax.transAxes, fontsize=17, fontweight="semibold", ha="left")
    ax.text(0.0, 1.045, "per-turn compliance probe  ·  out-of-fold, GroupKFold by "
            "conversation  ·  9,400-conv pool",
            transform=ax.transAxes, fontsize=12.5, color="#6b6b6b", ha="left")

    leg = ax.legend([hp] + handles, [hp.get_label()] + [h.get_label() for h in handles],
                    loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=13,
                    frameon=False, handlelength=1.7, labelspacing=0.65,
                    title="category  ·  AUC", title_fontsize=13.5, alignment="left")
    leg.get_title().set_fontweight("semibold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/per_category_roc_L{layer}.{ext}"
        fig.savefig(out_p, bbox_inches="tight", dpi=200 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


def main():
    jf = f"{OUT}/per_category_roc_L{LAYER}.json"
    if os.path.exists(jf):
        print(f"Loading cached {jf} (skip recompute)", flush=True)
        d = json.load(open(jf))
        render(d["per_category"], d["pooled"], LAYER)
        return

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
        rocs[cat] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc),
                     "n": int(mask.sum()), "n_pos": int(yc.sum())}
        print(f"  {cat:<28}  AUC={auc:.4f}", flush=True)

    oof_all = oof_predictions(X, y, groups)
    keep = ~np.isnan(oof_all)
    fpr_all, tpr_all, _ = roc_curve(y[keep], oof_all[keep])
    auc_all = roc_auc_score(y[keep], oof_all[keep])
    pooled = {"fpr": fpr_all.tolist(), "tpr": tpr_all.tolist(),
              "auc": float(auc_all), "n": len(y), "n_pos": int(y.sum())}
    with open(jf, "w") as f:
        json.dump({"layer": LAYER, "per_category": rocs, "pooled": pooled}, f)
    print(f"\nWrote {jf}")
    render(rocs, pooled, LAYER)


if __name__ == "__main__":
    main()
