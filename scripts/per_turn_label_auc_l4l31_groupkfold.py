"""Per-category compliance AUC using CONCATENATED [L4 ; L31] residuals.

Identical to per_turn_label_auc_groupkfold.py except features are the
concatenation of layer-4 and layer-31 hidden states at the same per-turn
position (8192-d total). GroupKFold by conversation_id, no leakage.

Outputs:
- experiments/outcome_probe_v1/per_turn_label_per_category_l4l31.json
- figures/per_category_auc_l4l31_vs_peak.{pdf,png}
"""
import gc
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

POOLED_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/pooled_hs"
OUT_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/outcome_probe_v1"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"

T_CRIT_4 = 2.776
HALFWIDTH_95 = T_CRIT_4 / math.sqrt(5)
LAYER_A = 4
LAYER_B = 31


def load_concat():
    """Load (X, y, cats, groups) with X = concat(L4, L31) per turn-row."""
    Xs, ys, cs, gs = [], [], [], []
    conv_counter = 0
    for src in sorted(os.listdir(POOLED_DIR)):
        sdir = os.path.join(POOLED_DIR, src)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".pt"):
                continue
            data = torch.load(os.path.join(sdir, fn), weights_only=False)
            arrA = data["hidden_states_by_layer"][LAYER_A].numpy()
            arrB = data["hidden_states_by_layer"][LAYER_B].numpy()
            labels = data["labels"].tolist()
            tobs = data["turns_of_breach"]
            cats = data["categories"]
            for i in range(len(labels)):
                cid = conv_counter; conv_counter += 1
                breach = bool(labels[i])
                t_star = tobs[i] if breach else None
                if breach and t_star is None:
                    continue
                t_max = 4 if not breach else int(t_star)
                for t in range(t_max + 1):
                    Xs.append(np.concatenate([arrA[i, t], arrB[i, t]]))
                    ys.append(1 if (breach and t == t_star) else 0)
                    cs.append(cats[i]); gs.append(cid)
            del data, arrA, arrB
    gc.collect()
    return np.stack(Xs), np.array(ys), np.array(cs), np.array(gs)


def group_kfold_auc(X, y, groups, n_splits=5, seed=42):
    if len(np.unique(y)) < 2:
        return None
    pos_groups = np.unique(groups[y == 1])
    neg_groups = np.unique(groups[y == 0])
    k = min(n_splits, len(pos_groups), len(neg_groups))
    if k < 2:
        return None
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    try:
        splits = list(sgkf.split(X, y, groups=groups))
    except ValueError:
        return None
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    if not aucs:
        return None
    return float(np.mean(aucs)), float(np.std(aucs)), k


def main():
    print(f"Loading concat(L{LAYER_A}, L{LAYER_B}) features...", flush=True)
    X, y, cats, groups = load_concat()
    print(f"  total turn-rows: {len(y)}, positives: {int(y.sum())}, "
          f"feature dim: {X.shape[1]}, unique convs: {len(np.unique(groups))}",
          flush=True)

    results = {}
    rg = group_kfold_auc(X, y, groups)
    print(f"\nGLOBAL  AUC={rg[0]:.4f} ± {rg[1]:.4f}  (k={rg[2]})", flush=True)
    results["__global__"] = {"auc": rg[0], "std": rg[1],
                             "n": len(y), "n_pos": int(y.sum()),
                             "k_used": rg[2]}
    for cat in sorted(set(cats)):
        mask = cats == cat
        Xc, yc, gc_ = X[mask], y[mask], groups[mask]
        rc = group_kfold_auc(Xc, yc, gc_)
        if rc is None:
            continue
        results[cat] = {
            "auc": rc[0], "std": rc[1],
            "n": int(mask.sum()), "n_pos": int(yc.sum()),
            "n_groups": int(len(np.unique(gc_))), "k_used": rc[2],
        }
        print(f"  {cat:<28}  AUC={rc[0]:.4f} ± {rc[1]:.4f}  "
              f"(n={int(mask.sum())}, pos={int(yc.sum())})", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/per_turn_label_per_category_l4l31.json", "w") as f:
        json.dump({"layer_a": LAYER_A, "layer_b": LAYER_B,
                   "per_category": results}, f, indent=2)
    print(f"\nWrote {OUT_DIR}/per_turn_label_per_category_l4l31.json")

    # ----- compare to GroupK single-layer peak -----
    gkf = json.load(open(f"{OUT_DIR}/per_turn_label_per_category_groupkfold.json"))

    def peak(cat):
        best = None
        for L, cell in gkf["per_category_per_layer"][cat].items():
            if best is None or cell["auc"] > best["auc"]:
                best = {**cell, "layer": L}
        return best

    g_glob = max(gkf["global_per_layer"].values(), key=lambda c: c["auc"])

    cats_sorted = sorted(
        (c for c in results.keys() if c != "__global__"),
        key=lambda c: results[c]["auc"]
    )

    # bars: 2 per category (single peak vs concat L4||L31)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    yy = np.arange(len(cats_sorted))
    h = 0.36
    aucs_peak = [peak(c)["auc"] for c in cats_sorted]
    err_peak = [HALFWIDTH_95 * peak(c)["std"] for c in cats_sorted]
    aucs_cc = [results[c]["auc"] for c in cats_sorted]
    err_cc = [HALFWIDTH_95 * results[c]["std"] for c in cats_sorted]

    ax.barh(yy - h/2, aucs_peak, height=h, color="#bcbd22",
            edgecolor="black", linewidth=0.4, alpha=0.92,
            label="single-layer peak (GroupKFold)")
    ax.errorbar(aucs_peak, yy - h/2, xerr=err_peak, fmt="none",
                ecolor="black", capsize=2.5, linewidth=0.9)
    ax.barh(yy + h/2, aucs_cc, height=h, color="#3b6fb0",
            edgecolor="black", linewidth=0.4, alpha=0.92,
            label=f"concat [L{LAYER_A};L{LAYER_B}] (GroupKFold)")
    ax.errorbar(aucs_cc, yy + h/2, xerr=err_cc, fmt="none",
                ecolor="black", capsize=2.5, linewidth=0.9)

    ax.axvline(g_glob["auc"], color="#bcbd22", linestyle="--", linewidth=0.9,
               label=f"single-layer pooled peak ({g_glob['auc']:.3f})")
    ax.axvline(results["__global__"]["auc"], color="#3b6fb0", linestyle="--",
               linewidth=0.9,
               label=f"concat pooled ({results['__global__']['auc']:.3f})")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.6)

    ax.set_yticks(yy)
    ax.set_yticklabels(cats_sorted, fontsize=9)
    for i, c in enumerate(cats_sorted):
        d = aucs_cc[i] - aucs_peak[i]
        x_right = max(aucs_peak[i] + err_peak[i], aucs_cc[i] + err_cc[i])
        ax.text(x_right + 0.012, i, f"$\\Delta=$ {d:+.3f}",
                va="center", fontsize=8.5,
                color="#1a6e1a" if d > 0 else "#a02020")

    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel(f"compliance AUC (per-turn label, GroupKFold by conv., "
                  "95% CI $t_4$)")
    ax.set_title(f"Per-category compliance AUC: single-layer peak vs "
                 f"concat[L{LAYER_A};L{LAYER_B}]",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/per_category_auc_l4l31_vs_peak.{ext}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
