"""Per-category compliance AUC, per-turn labels, GROUP-K-FOLD by conversation.

Identical to per_turn_label_auc.py except the CV splitter is
StratifiedGroupKFold with groups = conversation_id, so no conversation
contributes turn-rows to both train and test folds. Eliminates the
within-conversation leakage flagged in the methodology review.

Outputs:
- experiments/outcome_probe_v1/per_turn_label_per_category_groupkfold.json
- figures/per_category_auc_new_groupkfold.{pdf,png}
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
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]

T_CRIT_4 = 2.776
HALFWIDTH_95 = T_CRIT_4 / math.sqrt(5)


def load_layer(layer):
    """Returns (X, y, cats, turns, groups). `groups` is a globally-unique
    integer conversation_id; rows from the same conversation share one id.
    """
    Xs, ys, cs, ts, gs = [], [], [], [], []
    conv_counter = 0
    for src in sorted(os.listdir(POOLED_DIR)):
        sdir = os.path.join(POOLED_DIR, src)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".pt"):
                continue
            data = torch.load(os.path.join(sdir, fn), weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            labels = data["labels"].tolist()
            tobs = data["turns_of_breach"]
            cats = data["categories"]
            for i in range(len(labels)):
                conv_id = conv_counter
                conv_counter += 1
                breach = bool(labels[i])
                t_star = tobs[i] if breach else None
                if breach and t_star is None:
                    continue
                t_max = 4 if not breach else int(t_star)
                for t in range(t_max + 1):
                    y_t = 1 if (breach and t == t_star) else 0
                    Xs.append(arr[i, t])
                    ys.append(y_t)
                    cs.append(cats[i])
                    ts.append(t)
                    gs.append(conv_id)
            del data, arr
    gc.collect()
    return (np.stack(Xs), np.array(ys), np.array(cs),
            np.array(ts), np.array(gs))


def group_kfold_auc(X, y, groups, n_splits=5, seed=42):
    if len(np.unique(y)) < 2:
        return None
    # Need at least n_splits unique groups with each class somewhere.
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return None
    # n_pos groups: groups containing at least one positive row
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
    results = {}
    global_results = {}
    total_rows = 0
    cat_counts = defaultdict(lambda: {"n": 0, "n_pos": 0, "n_groups": 0})
    n_groups_total = 0
    for L in LAYERS:
        print(f"\nLayer {L}:")
        X, y, cats, turns, groups = load_layer(L)
        if total_rows == 0:
            total_rows = len(y)
            n_groups_total = int(len(np.unique(groups)))
            print(f"  total turn-rows: {total_rows}, "
                  f"positives: {int(y.sum())}, "
                  f"unique conversations (groups): {n_groups_total}")
            for c in sorted(set(cats)):
                m = cats == c
                cat_counts[c]["n"] = int(m.sum())
                cat_counts[c]["n_pos"] = int(y[m].sum())
                cat_counts[c]["n_groups"] = int(len(np.unique(groups[m])))

        r = group_kfold_auc(X, y, groups)
        if r:
            global_results[f"L{L}"] = {
                "auc": r[0], "std": r[1], "n": len(y),
                "n_pos": int(y.sum()), "k_used": r[2],
            }
            print(f"  global  AUC={r[0]:.4f} ± {r[1]:.4f}  (k={r[2]})")

        for cat in sorted(set(cats)):
            mask = cats == cat
            Xc, yc, gc_ = X[mask], y[mask], groups[mask]
            rc = group_kfold_auc(Xc, yc, gc_)
            if rc is None:
                continue
            results.setdefault(cat, {})[f"L{L}"] = {
                "auc": rc[0], "std": rc[1],
                "n": int(mask.sum()), "n_pos": int(yc.sum()),
                "n_groups": int(len(np.unique(gc_))),
                "k_used": rc[2],
            }
            print(f"  {cat[:24]:<24}  AUC={rc[0]:.4f} ± {rc[1]:.4f}  "
                  f"(n={int(mask.sum())}, pos={int(yc.sum())}, "
                  f"groups={int(len(np.unique(gc_)))}, k={rc[2]})")
        del X, y, cats, turns, groups
        gc.collect()

    out = {
        "global_per_layer": global_results,
        "per_category_per_layer": results,
        "layers": LAYERS,
        "total_turn_rows": total_rows,
        "total_conversations": n_groups_total,
        "category_counts": dict(cat_counts),
        "splitter": "StratifiedGroupKFold (groups=conversation_id)",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/per_turn_label_per_category_groupkfold.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/per_turn_label_per_category_groupkfold.json")

    # ----- plot peak per category -----
    rows = []
    for cat, by_L in results.items():
        best = None
        for L, cell in by_L.items():
            if best is None or cell["auc"] > best["auc"]:
                best = {**cell, "layer": L}
        rows.append({"cat": cat, **best})
    rows.sort(key=lambda r: r["auc"])

    glob_peak = max(c["auc"] for c in global_results.values())

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    yy = list(range(len(rows)))
    aucs = [r["auc"] for r in rows]
    ci = [HALFWIDTH_95 * r["std"] for r in rows]

    ax.barh(yy, aucs, color="#3b6fb0", edgecolor="black",
            linewidth=0.5, alpha=0.92)
    ax.errorbar(aucs, yy, xerr=ci, fmt="none", ecolor="black",
                capsize=3.5, linewidth=1.0)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.6)
    ax.axvline(glob_peak, color="#d65a31", linestyle="--", linewidth=0.9,
               label=f"pooled-corpus peak ({glob_peak:.3f})")

    ax.set_yticks(yy)
    ax.set_yticklabels([r["cat"] for r in rows], fontsize=9)
    for yi, (r, c) in enumerate(zip(rows, ci)):
        ax.text(r["auc"] + c + 0.012, yi,
                f"{r['auc']:.3f} ± {c:.3f}  (L{r['layer'].lstrip('L')}, "
                f"k={r['k_used']}, n={r['n_pos']}/{r['n']})",
                va="center", fontsize=8)
    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("compliance AUC (peak over layers), "
                  "95% CI from group-stratified k-fold CV ($t$-multiplier)")
    ax.set_title("Per-category compliance AUC, per-turn labels,\n"
                 "GroupKFold by conversation (no within-conversation leakage)",
                 fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/per_category_auc_new_groupkfold.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
