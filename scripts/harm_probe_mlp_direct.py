"""Experiment 4: MLP direct-harm probe vs Linear direct-harm probe.

If MLP-direct-harm AUC ≈ linear direct-harm AUC, harm is approximately
linearly represented. If MLP jumps significantly, there's additional
nonlinear structure.

Pipeline:
  for L in {16, 20, 24, 28, 31}:
    5-fold StratifiedKFold on the 289 Stage B wins
    fit MLP (4096 -> 64 -> 1, ReLU, alpha=1e-2, balanced sample weights)
    collect OOB predictions, bootstrap CI for AUC
    compare to linear direct-harm AUC from harm_probe_experiments.json

Output:
  experiments/outcome_probe_v1/harm_probe_mlp_direct.json
  figures/harm_probe_mlp_vs_linear.{pdf,png}
"""
import gc
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4
N_BOOT = 500
SEED = 42


def load_sb():
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    out = []
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        parts = r["conv_id"].split("/")
        if len(parts) != 3:
            continue
        key = (parts[0], int(parts[1].lstrip("r")), int(parts[2].lstrip("i")))
        if key not in cand:
            continue
        conv = cand[key]["conversation"]
        asst_turns = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        if not asst_turns:
            continue
        if r["unsafe"] and r.get("turn_of_breach") is not None and \
                int(r["turn_of_breach"]) < len(asst_turns):
            rated_turn = int(r["turn_of_breach"])
        else:
            rated_turn = len(asst_turns) - 1
        out.append({"key": key, "rated_turn": rated_turn,
                    "unsafe": r["unsafe"], "harm": r["rating"]})
    return out


def load_pool_layer(layer):
    out = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i in range(len(arr)):
                # use file basename as key — we just need src/round/idx
                pass
            for i, _ in enumerate(data["labels"].tolist()):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_", "")
                           .replace(".pt", ""))
                out[(src, rnum, i)] = arr[i]
    return out


def bootstrap_auc(y, scores, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]; sb = scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def main():
    sb = load_sb()
    wins = [r for r in sb if r["unsafe"]]
    y_harm = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)
    n_hi = int(y_harm.sum()); n_lo = len(y_harm) - n_hi
    print(f"Wins: {len(wins)}, high≥{HARM_THRESH}: {n_hi}, low: {n_lo}",
          flush=True)

    results = {"harm_thresh": HARM_THRESH, "per_layer": {}}

    for L in LAYERS:
        print(f"\nLayer L{L}...", flush=True)
        pool = load_pool_layer(L)
        X = np.stack([pool[r["key"]][r["rated_turn"]] for r in wins])
        del pool; gc.collect()

        # MLP 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.full(len(wins), np.nan)
        fold_aucs = []
        for fi, (tr, te) in enumerate(skf.split(X, y_harm)):
            if len(np.unique(y_harm[tr])) < 2:
                continue
            n_pos_tr = int(y_harm[tr].sum())
            n_neg_tr = int((1 - y_harm[tr]).sum())
            sw = np.where(y_harm[tr] == 1,
                          len(tr) / (2 * max(1, n_pos_tr)),
                          len(tr) / (2 * max(1, n_neg_tr)))
            clf = MLPClassifier(hidden_layer_sizes=(64,),
                                activation="relu",
                                alpha=1e-2,
                                learning_rate_init=1e-3,
                                max_iter=200, random_state=SEED + fi,
                                early_stopping=True,
                                validation_fraction=0.15,
                                n_iter_no_change=10)
            try:
                clf.fit(X[tr], y_harm[tr], sample_weight=sw)
            except TypeError:
                # fallback: oversample positives
                pos = np.where(y_harm[tr] == 1)[0]
                neg = np.where(y_harm[tr] == 0)[0]
                rng_ = np.random.default_rng(SEED + fi)
                pos_up = rng_.choice(pos, size=len(neg), replace=True)
                idx_bal = np.concatenate([neg, pos_up])
                rng_.shuffle(idx_bal)
                clf.fit(X[tr][idx_bal], y_harm[tr][idx_bal])
            s = clf.predict_proba(X[te])[:, 1]
            oof[te] = s
            fold_aucs.append(roc_auc_score(y_harm[te], s))

        keep = ~np.isnan(oof)
        auc_oob = roc_auc_score(y_harm[keep], oof[keep])
        bs = bootstrap_auc(y_harm[keep], oof[keep])
        results["per_layer"][f"L{L}"] = {
            "mlp_auc_meanfold": float(np.mean(fold_aucs)),
            "mlp_auc_meanfold_std": float(np.std(fold_aucs)),
            "mlp_auc_oob": float(auc_oob),
            "mlp_auc_ci95_boot": [bs[1], bs[2]],
        }
        print(f"  MLP DIRECT-HARM: meanfold "
              f"{np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}; "
              f"OOB {auc_oob:.3f} [{bs[1]:.3f}, {bs[2]:.3f}]", flush=True)
        del X; gc.collect()

    with open(f"{OUT}/harm_probe_mlp_direct.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}/harm_probe_mlp_direct.json")

    # ---- plot ----
    prior = json.load(open(f"{OUT}/harm_probe_experiments.json"))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    Ls = LAYERS
    mlp_oob = [results["per_layer"][f"L{L}"]["mlp_auc_oob"] for L in Ls]
    mlp_lo = [mlp_oob[i] - results["per_layer"][f"L{L}"]["mlp_auc_ci95_boot"][0]
              for i, L in enumerate(Ls)]
    mlp_hi = [results["per_layer"][f"L{L}"]["mlp_auc_ci95_boot"][1] - mlp_oob[i]
              for i, L in enumerate(Ls)]

    lin_oob = [prior["exp3_per_layer"][f"L{L}"]["auc_oob"] for L in Ls]
    lin_lo = [lin_oob[i] - prior["exp3_per_layer"][f"L{L}"]["auc_ci95_boot"][0]
              for i, L in enumerate(Ls)]
    lin_hi = [prior["exp3_per_layer"][f"L{L}"]["auc_ci95_boot"][1] - lin_oob[i]
              for i, L in enumerate(Ls)]

    ax.errorbar(Ls, lin_oob, yerr=[lin_lo, lin_hi], marker="s",
                markersize=6, linewidth=1.7, color="#d65a31", capsize=3,
                label="LINEAR direct-harm probe (5-fold CV on wins, "
                      "bootstrap 95% CI)")
    ax.errorbar(Ls, mlp_oob, yerr=[mlp_lo, mlp_hi], marker="D",
                markersize=6, linewidth=1.7, color="#7f3fbf", capsize=3,
                label="MLP direct-harm probe (5-fold CV on wins, "
                      "bootstrap 95% CI)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6)
    ax.set_xticks(Ls)
    ax.set_xticklabels([f"L{L}" for L in Ls])
    ax.set_xlabel("Llama-3.1-8B-Instruct layer")
    ax.set_ylabel("harm AUC within wins (n=289, high≥4 vs low≤3)")
    ax.set_title("Direct-harm probe: linear vs MLP, per layer\n"
                 "(if curves match: harm is linearly representable)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_ylim(0.5, 0.9)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/harm_probe_mlp_vs_linear.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
