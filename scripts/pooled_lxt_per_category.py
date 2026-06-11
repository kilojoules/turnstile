"""Per-(layer, turn, category) outcome AUC on pooled data.

Produces a 3D result: layer x turn x category.
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
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i in range(len(labels)):
                hs.append(arr[i, turn])
                y.append(int(bool(labels[i])))
                cats.append(src_cats[i])
            del data, arr
    gc.collect()
    return np.stack(hs), np.array(y), np.array(cats)


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
    results = {}  # category -> layer -> turn -> {auc, std, n, n_pos}
    global_matrix = {}  # layer -> turn -> auc (all cats pooled)

    total = 0
    all_cats = set()

    for L in LAYERS:
        print(f"\nLayer {L}:")
        for T in range(5):
            X, y, cats = load_single_lt(L, T)
            if total == 0:
                total = len(y)
                print(f"[{total} convs, {int(y.sum())} wins]")

            # Global
            r = kfold_auc(X, y)
            if r:
                global_matrix.setdefault(f"L{L}", {})[f"T{T}"] = {
                    "auc": r[0], "std": r[1], "n": len(y), "n_pos": int(y.sum())
                }

            # Per-category
            row = [f"  T{T}:"]
            for cat in sorted(set(cats)):
                all_cats.add(cat)
                mask = cats == cat
                Xc, yc = X[mask], y[mask]
                rc = kfold_auc(Xc, yc)
                if rc is not None:
                    results.setdefault(cat, {}).setdefault(f"L{L}", {})[f"T{T}"] = {
                        "auc": rc[0], "std": rc[1],
                        "n": int(mask.sum()), "n_pos": int(yc.sum())
                    }
                    row.append(f"{cat[:4]}={rc[0]:.2f}")
            print(" ".join(row))
            del X, y, cats
            gc.collect()

    out = {
        "global": global_matrix,
        "per_category": results,
        "layers": LAYERS,
        "categories": sorted(all_cats),
        "total_convs": total,
    }
    with open(f"{OUT_DIR}/pooled_lxt_per_category.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/pooled_lxt_per_category.json")

    # One heatmap per category
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cats = len(all_cats)
    cols = 5
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.8))
    axes = np.array(axes).reshape(rows, cols)
    sorted_cats = sorted(all_cats)

    # Global range for consistent colormap
    all_aucs = []
    for c in results.values():
        for L in c.values():
            for t in L.values():
                all_aucs.append(t["auc"])
    vmin, vmax = 0.45, max(0.80, max(all_aucs) if all_aucs else 0.80)

    for idx, cat in enumerate(sorted_cats):
        ax = axes[idx // cols, idx % cols]
        mat = np.full((len(LAYERS), 5), np.nan)
        for i, L in enumerate(LAYERS):
            for T in range(5):
                v = results.get(cat, {}).get(f"L{L}", {}).get(f"T{T}")
                if v:
                    mat[i, T] = v["auc"]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(cat[:22], fontsize=9)
        ax.set_xticks(range(5))
        ax.set_xticklabels([str(t) for t in range(5)], fontsize=7)
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels([str(L) for L in LAYERS], fontsize=7)
        # Annotate peak
        if not np.all(np.isnan(mat)):
            pi, pj = np.unravel_index(np.nanargmax(mat), mat.shape)
            ax.add_patch(plt.Rectangle((pj - 0.5, pi - 0.5), 1, 1,
                                       fill=False, edgecolor="white", linewidth=1.5))
            ax.text(pj, pi, f"{mat[pi, pj]:.2f}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    # Hide unused axes
    for idx in range(n_cats, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(f"Outcome Prediction AUC by Category (layer × turn, pooled {total} convs)",
                 fontsize=11, y=1.02)
    fig.text(0.5, -0.01, "Turn", ha="center", fontsize=10)
    fig.text(-0.01, 0.5, "Layer", va="center", rotation="vertical", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pooled_lxt_per_category.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/pooled_lxt_per_category.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_DIR}/pooled_lxt_per_category.png")


if __name__ == "__main__":
    main()
