"""Full Layer x Turn outcome prediction sweep on pooled hidden states.

Streaming: processes one (L, T) at a time to fit in RAM.
"""
import gc
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

POOLED_DIR = "experiments/pooled_hs"
OUT_DIR = "experiments/outcome_probe_v1"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def load_single_lt(layer, turn):
    """Stream single (layer, turn) across all sources. Returns X, y, cats."""
    hs, y, cats = [], [], []
    for src in sorted(os.listdir(POOLED_DIR)):
        sdir = os.path.join(POOLED_DIR, src)
        if not os.path.isdir(sdir):
            continue
        for f in sorted(os.listdir(sdir)):
            if not f.endswith(".pt"):
                continue
            data = torch.load(os.path.join(sdir, f), weights_only=False)
            labels = data["labels"].tolist()
            src_cats = data["categories"]
            arr = data["hidden_states_by_layer"][layer].numpy()  # (n, 5, 4096)
            for i in range(len(labels)):
                hs.append(arr[i, turn])
                y.append(int(bool(labels[i])))
                cats.append(src_cats[i])
            del data, arr
            gc.collect()
    return np.stack(hs), np.array(y), cats


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
    os.makedirs(OUT_DIR, exist_ok=True)

    matrix = {}
    per_cat_cache = {}  # (L, T) -> per-category dict

    print("=== Pooled L x T AUC sweep ===")
    print(f"{'':6s}" + "  ".join(f"  T{t}  " for t in range(5)))

    total_n = None
    total_wins = None

    for L in LAYERS:
        row_txt = [f"L{L:>2d} "]
        for T in range(5):
            X, y, cats = load_single_lt(L, T)
            if total_n is None:
                total_n, total_wins = len(y), int(y.sum())
                print(f"[{total_n} convs, {total_wins} wins]\n")
                print(f"{'':6s}" + "  ".join(f"  T{t}  " for t in range(5)))
                print(f"L{L:>2d} ", end="", flush=True)

            r = kfold_auc(X, y)
            if r is None:
                row_txt.append("  --  ")
                continue
            auc, std = r
            matrix[f"L{L}_T{T}"] = {"auc": auc, "std": std, "n": len(y),
                                    "n_pos": int(y.sum()), "layer": L, "turn": T}
            row_txt.append(f"{auc:.3f} ")

            # Save per-category only for key cells to save memory (L16, L20, all turns)
            if L in (16, 20):
                cats_arr = np.array(cats)
                per_cat = {}
                for cat in sorted(set(cats)):
                    mask = cats_arr == cat
                    Xc, yc = X[mask], y[mask]
                    rc = kfold_auc(Xc, yc)
                    if rc is not None:
                        per_cat[cat] = {"auc": rc[0], "std": rc[1],
                                        "n": int(mask.sum()), "n_pos": int(yc.sum())}
                per_cat_cache[f"L{L}_T{T}"] = per_cat

            # Free memory
            del X, y, cats
            gc.collect()

        print("  ".join(row_txt[1:]))
        if L != LAYERS[-1]:
            print(f"L{LAYERS[LAYERS.index(L)+1]:>2d} ", end="", flush=True)

    peak_key = max(matrix, key=lambda k: matrix[k]["auc"])
    print(f"\nPeak: {peak_key} AUC = {matrix[peak_key]['auc']:.4f}")

    # Per-category at peak
    if peak_key in per_cat_cache:
        print(f"\n=== Within-category at peak {peak_key} ===")
        pc = per_cat_cache[peak_key]
        for cat, v in sorted(pc.items(), key=lambda kv: -kv[1]["auc"]):
            print(f"  {cat:<30s} AUC={v['auc']:.3f} ± {v['std']:.3f} "
                  f"(n={v['n']}, {v['n_pos']} wins)")
        aucs = [v["auc"] for v in pc.values()]
        print(f"\n  Macro: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # Save
    out = {
        "matrix": matrix,
        "peak": {"key": peak_key, **matrix[peak_key]},
        "per_category_L16_L20": per_cat_cache,
        "total_convs": total_n, "total_wins": total_wins,
    }
    with open(f"{OUT_DIR}/pooled_layer_turn_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/pooled_layer_turn_sweep.json")

    # Heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.array([[matrix.get(f"L{L}_T{T}", {"auc": np.nan})["auc"]
                     for T in range(5)] for L in LAYERS])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0.50, vmax=0.80)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"Turn {t}" for t in range(5)])
    ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels([f"Layer {L}" for L in LAYERS])
    ax.set_title(f"Pooled Outcome Prediction AUC ({total_n:,} convs, 11 experiments)")
    peak_auc = matrix[peak_key]["auc"]
    for i, L in enumerate(LAYERS):
        for j in range(5):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.72 else "black"
                weight = "bold" if v >= peak_auc - 0.005 else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight=weight)
    pL, pT = matrix[peak_key]["layer"], matrix[peak_key]["turn"]
    ax.add_patch(plt.Rectangle((pT - 0.5, LAYERS.index(pL) - 0.5), 1, 1,
                               fill=False, edgecolor="white", linewidth=2.5))
    plt.colorbar(im, ax=ax, shrink=0.8).set_label("AUC (5-fold CV)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pooled_layer_turn_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/pooled_layer_turn_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_DIR}/pooled_layer_turn_heatmap.png")


if __name__ == "__main__":
    main()
