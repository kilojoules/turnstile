"""Per-category compliance AUC using PER-TURN labels (not conversation-level).

Dataset construction (no new judge calls):
- For each conversation:
    - if non-breach (labels[i] == 0): include (HS at turn t, y=0) for t in 0..4
    - if breach (labels[i] == 1) at known turn t*: include (HS at turn t, y=t==t*)
      for t in 0..t* only (drop turns t*+1..4 — never judged by upstream)
- Fit one logistic probe per layer per category, k-fold CV AUC.
- Take peak over layers per category.

Outputs `experiments/outcome_probe_v1/per_turn_label_per_category.json`
and `figures/per_category_auc_per_turn.{pdf,png}`.
"""
import gc
import glob
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
from sklearn.model_selection import StratifiedKFold

POOLED_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/pooled_hs"
OUT_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/outcome_probe_v1"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]

T_CRIT_4 = 2.776  # t_{0.975, df=4}
HALFWIDTH_95 = T_CRIT_4 / math.sqrt(5)  # ≈ 1.241


def load_layer(layer):
    """Returns (X: [n_turn_rows, 4096], y: [n_turn_rows], cats, turns).

    Each row is a (conversation, turn) pair with a known per-turn label.
    """
    Xs, ys, cs, ts = [], [], [], []
    for src in sorted(os.listdir(POOLED_DIR)):
        sdir = os.path.join(POOLED_DIR, src)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".pt"):
                continue
            data = torch.load(os.path.join(sdir, fn), weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()  # (n, 5, 4096)
            labels = data["labels"].tolist()
            tobs = data["turns_of_breach"]
            cats = data["categories"]
            for i in range(len(labels)):
                breach = bool(labels[i])
                t_star = tobs[i] if breach else None
                if breach and t_star is None:
                    continue  # safety
                # determine which turns have known labels
                t_max = 4 if not breach else int(t_star)
                for t in range(t_max + 1):
                    y_t = 1 if (breach and t == t_star) else 0
                    Xs.append(arr[i, t])
                    ys.append(y_t)
                    cs.append(cats[i])
                    ts.append(t)
            del data, arr
    gc.collect()
    return np.stack(Xs), np.array(ys), np.array(cs), np.array(ts)


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


def main():
    results = {}  # category -> layer -> {auc, std, n, n_pos}
    global_results = {}  # layer -> {auc, std, n, n_pos}
    total_rows = 0
    cat_counts = defaultdict(lambda: {"n": 0, "n_pos": 0})
    for L in LAYERS:
        print(f"\nLayer {L}:")
        X, y, cats, turns = load_layer(L)
        if total_rows == 0:
            total_rows = len(y)
            print(f"  total turn-rows: {total_rows}, "
                  f"positives: {int(y.sum())}, "
                  f"negatives: {len(y) - int(y.sum())}")
            for c in sorted(set(cats)):
                m = cats == c
                cat_counts[c]["n"] = int(m.sum())
                cat_counts[c]["n_pos"] = int(y[m].sum())

        # global
        r = kfold_auc(X, y)
        if r:
            global_results[f"L{L}"] = {"auc": r[0], "std": r[1],
                                       "n": len(y), "n_pos": int(y.sum())}
            print(f"  global  AUC={r[0]:.4f} ± {r[1]:.4f}")

        # per category
        for cat in sorted(set(cats)):
            mask = cats == cat
            Xc, yc = X[mask], y[mask]
            rc = kfold_auc(Xc, yc)
            if rc is None:
                continue
            results.setdefault(cat, {})[f"L{L}"] = {
                "auc": rc[0], "std": rc[1],
                "n": int(mask.sum()), "n_pos": int(yc.sum()),
            }
            print(f"  {cat[:24]:<24}  AUC={rc[0]:.4f} ± {rc[1]:.4f}  "
                  f"(n={int(mask.sum())}, pos={int(yc.sum())})")
        del X, y, cats, turns
        gc.collect()

    out = {
        "global_per_layer": global_results,
        "per_category_per_layer": results,
        "layers": LAYERS,
        "total_turn_rows": total_rows,
        "category_counts": dict(cat_counts),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/per_turn_label_per_category.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/per_turn_label_per_category.json")

    # ----- plot peak per category -----
    rows = []
    for cat, by_L in results.items():
        best = None
        for L, cell in by_L.items():
            if best is None or cell["auc"] > best["auc"]:
                best = {**cell, "layer": L}
        rows.append({"cat": cat, **best})
    rows.sort(key=lambda r: r["auc"])

    glob_peak_cell = max(global_results.values(), key=lambda c: c["auc"])
    glob_peak = glob_peak_cell["auc"]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    yy = list(range(len(rows)))
    aucs = [r["auc"] for r in rows]
    ci95 = [HALFWIDTH_95 * r["std"] for r in rows]

    ax.barh(yy, aucs, color="#3b6fb0", edgecolor="black", linewidth=0.5,
            alpha=0.92)
    ax.errorbar(aucs, yy, xerr=ci95, fmt="none", ecolor="black",
                capsize=3.5, linewidth=1.0)

    ax.set_yticks(yy)
    ax.set_yticklabels([r["cat"] for r in rows], fontsize=9)

    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.7,
               label="chance (0.5)")
    ax.axvline(glob_peak, color="#d65a31", linestyle="--", linewidth=0.9,
               label=f"pooled-corpus peak ({glob_peak:.3f})")

    for yi, r in enumerate(rows):
        L = r["layer"].lstrip("L")
        lo = r["auc"] - HALFWIDTH_95 * r["std"]
        hi = r["auc"] + HALFWIDTH_95 * r["std"]
        ax.text(r["auc"] + 0.012 + HALFWIDTH_95 * r["std"], yi,
                f"{r['auc']:.3f} [{lo:.3f}, {hi:.3f}]  "
                f"L{L}  n={r['n_pos']}/{r['n']}",
                va="center", fontsize=8)

    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("compliance AUC (peak over layers, per-turn label), "
                  "95% CI from 5-fold CV ($t_4$)")
    ax.set_title("Per-category peak compliance AUC, PER-TURN LABELS\n"
                 "(probe at turn $t$ predicts: is the turn-$t$ response itself unsafe?)",
                 fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{FIG_DIR}/per_category_auc_per_turn.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)

    print(f"\nGlobal peak (per-turn-label) AUC: {glob_peak:.4f}  "
          f"at {[L for L, c in global_results.items() if c['auc']==glob_peak][0]}")
    print(f"\nPer-category peak AUC (per-turn labels, 5-fold CV; "
          f"95% half-width = {HALFWIDTH_95:.3f} × std):")
    print(f"{'category':<30}  {'AUC':>6}  {'95% CI':>15}   layer    n_pos/n")
    for r in sorted(rows, key=lambda r: -r["auc"]):
        lo = r["auc"] - HALFWIDTH_95 * r["std"]
        hi = r["auc"] + HALFWIDTH_95 * r["std"]
        print(f"  {r['cat']:<28}  {r['auc']:>6.4f}  [{lo:.4f}, {hi:.4f}]"
              f"   {r['layer']}     {r['n_pos']}/{r['n']}")


if __name__ == "__main__":
    main()
