"""Three experiments on harm vs compliance representation.

Experiment 1 (MLP compliance probe at L16):
  Train a 2-layer MLP (4096 -> 128 -> 1, ReLU, dropout) on the same per-turn
  compliance task at L16. Apply to Stage B wins at breach turn. Compute
  harm AUC (Likert >=4 vs <=3). Compare to linear baseline (0.564).

Experiment 2 (Linear harm probe at L16, directly):
  Train a linear probe at L16 with HARM labels (high vs low). 5-fold CV on
  Stage B wins (n=289). If AUC >> 0.55, harm is linearly accessible.

Experiment 3 (Linear harm probe at every layer):
  Same as Exp 2, at L in {0, 4, 8, 12, 16, 20, 24, 28, 31}.

Outputs:
  experiments/outcome_probe_v1/harm_probe_experiments.json
  figures/harm_probe_experiments.{pdf,png}
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
LAYER_MAIN = 16
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
                    "unsafe": r["unsafe"], "harm": r["rating"],
                    "category": r["category"]})
    return out


def load_pool_layer(layer):
    out = {}
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
                out[(src, rnum, i)] = {"hs": arr[i],
                                       "label": bool(lab),
                                       "turn_of_breach": tob}
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
    sb_keys = set(r["key"] for r in sb)
    wins = [r for r in sb if r["unsafe"]]
    print(f"SB records: {len(sb)} (wins={len(wins)})", flush=True)
    n_hi = sum(1 for r in wins if r["harm"] >= HARM_THRESH)
    n_lo = len(wins) - n_hi
    print(f"  wins with harm>={HARM_THRESH}: {n_hi};  <{HARM_THRESH}: {n_lo}",
          flush=True)

    results = {"harm_thresh": HARM_THRESH, "layer_main": LAYER_MAIN}

    # ====== EXPERIMENT 1 (MLP at L16) + EXPERIMENT 2 (linear harm at L16) ======
    print(f"\n=== L{LAYER_MAIN} experiments ===", flush=True)
    pool = load_pool_layer(LAYER_MAIN)
    # Training set for compliance task (non-SB convs, per-turn rows)
    X_tr, y_tr = [], []
    for key, rec in pool.items():
        if key in sb_keys:
            continue
        breach = rec["label"]
        t_star = rec["turn_of_breach"] if breach else None
        if breach and t_star is None:
            continue
        t_max = 4 if not breach else int(t_star)
        for t in range(t_max + 1):
            X_tr.append(rec["hs"][t])
            y_tr.append(1 if (breach and t == t_star) else 0)
    X_tr = np.stack(X_tr); y_tr = np.array(y_tr)
    print(f"  compliance training rows: {len(y_tr)}, positives: {int(y_tr.sum())}",
          flush=True)

    # Linear compliance baseline (Exp 1 control)
    clf_lin = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs").fit(X_tr, y_tr)

    # MLP compliance probe (Exp 1)
    print("  training MLP (4096 -> 128 -> 1, ReLU, dropout)...", flush=True)
    sample_w = np.where(y_tr == 1,
                        len(y_tr) / (2 * max(1, y_tr.sum())),
                        len(y_tr) / (2 * max(1, len(y_tr) - y_tr.sum())))
    clf_mlp = MLPClassifier(hidden_layer_sizes=(128,),
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=1e-3,
                            max_iter=80, random_state=SEED,
                            early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=5)
    # MLPClassifier supports sample_weight in newer sklearn versions; fall
    # back to oversampling positives if not.
    try:
        clf_mlp.fit(X_tr, y_tr, sample_weight=sample_w)
    except TypeError:
        pos = np.where(y_tr == 1)[0]
        neg = np.where(y_tr == 0)[0]
        rng = np.random.default_rng(SEED)
        pos_up = rng.choice(pos, size=len(neg), replace=True)
        idx = np.concatenate([neg, pos_up])
        rng.shuffle(idx)
        clf_mlp.fit(X_tr[idx], y_tr[idx])
    print(f"  MLP train acc: {clf_mlp.score(X_tr, y_tr):.3f}", flush=True)

    # Apply both to Stage B wins HS at breach turn
    X_win = np.stack([pool[r["key"]]["hs"][r["rated_turn"]] for r in wins])
    y_harm = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)

    s_lin = clf_lin.predict_proba(X_win)[:, 1]
    s_mlp = clf_mlp.predict_proba(X_win)[:, 1]
    auc_lin = roc_auc_score(y_harm, s_lin)
    auc_mlp = roc_auc_score(y_harm, s_mlp)
    bs_lin = bootstrap_auc(y_harm, s_lin)
    bs_mlp = bootstrap_auc(y_harm, s_mlp)

    print(f"\n  EXP 1 (linear compliance → harm AUC within wins): "
          f"{auc_lin:.3f}  [{bs_lin[1]:.3f}, {bs_lin[2]:.3f}]", flush=True)
    print(f"  EXP 1 (MLP    compliance → harm AUC within wins): "
          f"{auc_mlp:.3f}  [{bs_mlp[1]:.3f}, {bs_mlp[2]:.3f}]", flush=True)

    results["exp1_L16"] = {
        "linear_compliance_harm_auc": auc_lin,
        "linear_compliance_harm_ci95": [bs_lin[1], bs_lin[2]],
        "mlp_compliance_harm_auc": auc_mlp,
        "mlp_compliance_harm_ci95": [bs_mlp[1], bs_mlp[2]],
    }

    # ====== EXPERIMENT 2 (linear harm probe at L16, direct) ======
    # 5-fold CV on the 289 wins, target = harm high/low
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs_direct = []
    oof_scores = np.full(len(wins), np.nan)
    for tr, te in skf.split(X_win, y_harm):
        if len(np.unique(y_harm[tr])) < 2:
            continue
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X_win[tr], y_harm[tr])
        s_te = clf.predict_proba(X_win[te])[:, 1]
        aucs_direct.append(roc_auc_score(y_harm[te], s_te))
        oof_scores[te] = s_te
    auc_direct_mean = float(np.mean(aucs_direct))
    auc_direct_std = float(np.std(aucs_direct))
    keep = ~np.isnan(oof_scores)
    auc_direct_oof = roc_auc_score(y_harm[keep], oof_scores[keep])
    bs_direct = bootstrap_auc(y_harm[keep], oof_scores[keep])
    print(f"\n  EXP 2 (linear DIRECT-HARM probe at L16, 5-fold CV):", flush=True)
    print(f"    mean fold AUC = {auc_direct_mean:.3f} ± {auc_direct_std:.3f}",
          flush=True)
    print(f"    OOB-concat AUC = {auc_direct_oof:.3f}  "
          f"[{bs_direct[1]:.3f}, {bs_direct[2]:.3f}]", flush=True)

    results["exp2_L16"] = {
        "linear_harm_auc_meanfold": auc_direct_mean,
        "linear_harm_auc_meanfold_std": auc_direct_std,
        "linear_harm_auc_oob": auc_direct_oof,
        "linear_harm_auc_ci95_boot": [bs_direct[1], bs_direct[2]],
    }
    del pool; gc.collect()

    # ====== EXPERIMENT 3 (linear harm probe at every layer) ======
    print(f"\n=== EXP 3: linear DIRECT-HARM probe at every layer ===",
          flush=True)
    layer_results = {}
    for L in LAYERS:
        pool = load_pool_layer(L)
        X_L = np.stack([pool[r["key"]]["hs"][r["rated_turn"]] for r in wins])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        aucs = []
        oof = np.full(len(wins), np.nan)
        for tr, te in skf.split(X_L, y_harm):
            if len(np.unique(y_harm[tr])) < 2:
                continue
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000, solver="lbfgs")
            clf.fit(X_L[tr], y_harm[tr])
            s = clf.predict_proba(X_L[te])[:, 1]
            aucs.append(roc_auc_score(y_harm[te], s))
            oof[te] = s
        keep = ~np.isnan(oof)
        auc_oob = roc_auc_score(y_harm[keep], oof[keep])
        bs = bootstrap_auc(y_harm[keep], oof[keep])
        layer_results[f"L{L}"] = {
            "auc_meanfold": float(np.mean(aucs)),
            "auc_meanfold_std": float(np.std(aucs)),
            "auc_oob": float(auc_oob),
            "auc_ci95_boot": [bs[1], bs[2]],
        }
        print(f"  L{L:>2}: meanfold {np.mean(aucs):.3f} ± {np.std(aucs):.3f}; "
              f"OOB {auc_oob:.3f} [{bs[1]:.3f}, {bs[2]:.3f}]", flush=True)
        del pool, X_L; gc.collect()
    results["exp3_per_layer"] = layer_results

    with open(f"{OUT}/harm_probe_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}/harm_probe_experiments.json")

    # ====== plot ======
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                    gridspec_kw={"width_ratios": [1.0, 1.6]})

    # Left: bar comparison at L16
    methods = [
        ("linear compliance\nprobe → harm AUC",
         results["exp1_L16"]["linear_compliance_harm_auc"],
         results["exp1_L16"]["linear_compliance_harm_ci95"]),
        ("MLP compliance\nprobe → harm AUC",
         results["exp1_L16"]["mlp_compliance_harm_auc"],
         results["exp1_L16"]["mlp_compliance_harm_ci95"]),
        ("linear DIRECT-HARM\nprobe (CV on wins)",
         results["exp2_L16"]["linear_harm_auc_oob"],
         results["exp2_L16"]["linear_harm_auc_ci95_boot"]),
    ]
    xs = np.arange(len(methods))
    aucs = [m[1] for m in methods]
    lo = [m[1] - m[2][0] for m in methods]
    hi = [m[2][1] - m[1] for m in methods]
    colors = ["#3b6fb0", "#7f7f7f", "#d65a31"]
    ax1.bar(xs, aucs, color=colors, edgecolor="black", linewidth=0.5,
            alpha=0.92)
    ax1.errorbar(xs, aucs, yerr=[lo, hi], fmt="none", ecolor="black",
                 linewidth=1.0, capsize=4)
    for i, (m, a) in enumerate(zip(methods, aucs)):
        ax1.text(i, a + 0.012, f"{a:.3f}", ha="center", fontsize=10)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([m[0] for m in methods], fontsize=9)
    ax1.axhline(0.5, color="black", linestyle=":", linewidth=0.6)
    ax1.set_ylabel("harm AUC within wins (n=289, high≥4 vs low≤3)")
    ax1.set_ylim(0.45, 0.85)
    ax1.set_title(f"At L{LAYER_MAIN}: linear vs MLP vs harm-supervised\n"
                  "(bootstrap 95% CIs)", fontsize=10.5)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.4)

    # Right: linear harm probe AUC vs layer
    Ls = LAYERS
    oob = [layer_results[f"L{L}"]["auc_oob"] for L in Ls]
    lo3 = [oob[i] - layer_results[f"L{L}"]["auc_ci95_boot"][0]
           for i, L in enumerate(Ls)]
    hi3 = [layer_results[f"L{L}"]["auc_ci95_boot"][1] - oob[i]
           for i, L in enumerate(Ls)]
    ax2.errorbar(Ls, oob, yerr=[lo3, hi3], marker="s", markersize=6,
                 linewidth=1.7, color="#d65a31", capsize=3,
                 label="linear DIRECT-HARM probe at L (5-fold CV on wins, "
                       "bootstrap 95% CI)")

    # Overlay: compliance->harm AUC from per_layer_harm_with_cis.json (for context)
    try:
        prior = json.load(open(f"{OUT}/per_layer_harm_with_cis.json"))
        comp_harm = [prior["per_layer"][f"L{L}"]["harm_auc_wins_point"]
                     for L in Ls]
        comp_harm_lo = [comp_harm[i]
                        - prior["per_layer"][f"L{L}"]["harm_auc_wins_lo95"]
                        for i, L in enumerate(Ls)]
        comp_harm_hi = [prior["per_layer"][f"L{L}"]["harm_auc_wins_hi95"]
                        - comp_harm[i] for i, L in enumerate(Ls)]
        ax2.errorbar(Ls, comp_harm, yerr=[comp_harm_lo, comp_harm_hi],
                     marker="o", markersize=6, linewidth=1.4,
                     color="#3b6fb0", capsize=3,
                     label="linear COMPLIANCE probe → harm AUC "
                           "(bootstrap CI)")
    except FileNotFoundError:
        pass

    ax2.axhline(0.5, color="black", linestyle=":", linewidth=0.6)
    ax2.set_xticks(Ls)
    ax2.set_xticklabels([f"L{L}" for L in Ls])
    ax2.set_xlabel("Llama-3.1-8B-Instruct layer")
    ax2.set_ylabel("harm AUC within wins (n=289)")
    ax2.set_title("Direct-harm probe vs compliance-then-harm, per layer",
                  fontsize=10.5)
    ax2.grid(alpha=0.25, linewidth=0.4)
    ax2.set_ylim(0.42, 0.85)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/harm_probe_experiments.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
