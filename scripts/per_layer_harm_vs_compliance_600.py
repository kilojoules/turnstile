"""Re-do per-layer compliance AUC restricted to the 600-conv Stage B
subset, so the comparison to harm-AUC is apples-to-apples on the same
conversation pool.

Pipeline:
  - For each layer L in {0,4,8,12,16,20,24,28,31}:
      build per-turn rows from the 600 Stage B conversations
      (wins: turns 0..t* labelled, positive at t*;
       losses: all 5 turns labelled safe)
      compute compliance AUC via GroupKFold by conversation_id
      reuse the harm AUCs already computed in
      per_layer_harm_vs_compliance.json

Output:
  experiments/outcome_probe_v1/per_layer_harm_vs_compliance_600.json
  figures/per_layer_harm_vs_compliance_600.{pdf,png}
"""
import gc
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4

T_CRIT_4 = 2.776
HALFWIDTH_95 = T_CRIT_4 / math.sqrt(5)


def load_sb_keys():
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        parts = r["conv_id"].split("/")
        if len(parts) != 3:
            continue
        k = (parts[0], int(parts[1].lstrip("r")),
             int(parts[2].lstrip("i")))
        if k in cand:
            keys.add(k)
    return keys


def load_layer_rows_for_sb(layer, sb_keys):
    """Return X, y, groups for per-turn rows of the 600 SB convs at layer L."""
    Xs, ys, gs = [], [], []
    conv_counter = 0
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        src = os.path.basename(sdir.rstrip("/"))
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            rnum = int(os.path.basename(path).replace("round_", "")
                       .replace(".pt", ""))
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i, (lab, tob) in enumerate(
                    zip(data["labels"].tolist(),
                        data["turns_of_breach"])):
                key = (src, rnum, i)
                if key not in sb_keys:
                    continue
                conv_id = conv_counter; conv_counter += 1
                breach = bool(lab)
                t_star = tob if breach else None
                if breach and t_star is None:
                    continue
                t_max = 4 if not breach else int(t_star)
                for t in range(t_max + 1):
                    Xs.append(arr[i, t])
                    ys.append(1 if (breach and t == t_star) else 0)
                    gs.append(conv_id)
            del data, arr
    gc.collect()
    return np.stack(Xs), np.array(ys), np.array(gs)


def group_kfold_auc(X, y, groups, n_splits=5, seed=42):
    pos_groups = np.unique(groups[y == 1])
    neg_groups = np.unique(groups[y == 0])
    k = min(n_splits, len(pos_groups), len(neg_groups))
    if k < 2:
        return None
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in sgkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return (float(np.mean(aucs)), float(np.std(aucs)), k) if aucs else None


def main():
    sb_keys = load_sb_keys()
    print(f"SB conversations: {len(sb_keys)}", flush=True)

    out = {"layers": LAYERS, "harm_thresh": HARM_THRESH, "per_layer": {}}
    for L in LAYERS:
        print(f"\nLayer L{L}...", flush=True)
        X, y, groups = load_layer_rows_for_sb(L, sb_keys)
        print(f"  rows: {len(y)}, positives: {int(y.sum())}, "
              f"groups: {len(np.unique(groups))}", flush=True)
        r = group_kfold_auc(X, y, groups)
        if r:
            out["per_layer"][f"L{L}"] = {
                "auc": r[0], "std": r[1], "k_used": r[2],
                "n_rows": len(y), "n_pos": int(y.sum()),
                "n_groups": int(len(np.unique(groups))),
            }
            print(f"  compliance AUC (600-conv subset, GroupK) "
                  f"= {r[0]:.4f} ± {r[1]:.4f} (k={r[2]})", flush=True)
        del X, y, groups
        gc.collect()

    with open(f"{OUT}/per_layer_harm_vs_compliance_600.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}/per_layer_harm_vs_compliance_600.json")

    # ----- plot, reusing harm AUCs from full-pool run -----
    prior = json.load(open(f"{OUT}/per_layer_harm_vs_compliance.json"))
    harm_within = prior["harm_auc_within_wins"]
    harm_all = prior["harm_auc_all600"]

    comp_600 = [out["per_layer"].get(f"L{L}", {}).get("auc", np.nan)
                for L in LAYERS]
    comp_600_std = [out["per_layer"].get(f"L{L}", {}).get("std", np.nan)
                    for L in LAYERS]
    ci_600 = [HALFWIDTH_95 * s for s in comp_600_std]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.errorbar(LAYERS, comp_600, yerr=ci_600,
                marker="o", markersize=6, linewidth=1.7,
                color="#3b6fb0", capsize=3,
                label=f"general compliance AUC  (600-conv subset, "
                      f"GroupKFold)")
    ax.plot(LAYERS, harm_within, marker="s", markersize=6, linewidth=1.7,
            color="#d65a31",
            label=f"harm AUC within wins  (n={prior['n_wins']}, "
                  f"high≥{HARM_THRESH}: {prior['n_wins_high_harm']}, "
                  f"low: {prior['n_wins_low_harm']})")
    ax.plot(LAYERS, harm_all, marker="^", markersize=6, linewidth=1.4,
            color="#7f7f7f", linestyle=":",
            label="harm AUC all 600  (high: "
                  f"{prior['n_wins_high_harm'] + sum(1 for _ in [])}/600, "
                  "approx.)")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_xlabel("Llama-3.1-8B-Instruct layer")
    ax.set_ylabel("AUC")
    ax.set_title("Compliance probe AUC vs harm-severity AUC, per layer\n"
                 "(blue: 600-conv subset only; orange/gray: same 600)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_ylim(0.45, 0.85)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/per_layer_harm_vs_compliance_600.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
