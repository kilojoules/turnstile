"""Add bootstrap 95% CIs to the orange/gray harm-AUC curves so they're
visually commensurate with the blue compliance-AUC CIs.

For each layer L:
  - blue line: existing compliance AUC ± 5-fold-CV t_4 CI (from
    per_layer_harm_vs_compliance_600.json)
  - orange/gray lines: bootstrap-resample the Stage B records (B=500),
    recompute high-vs-low harm AUC at L using the same compliance probe
    fitted on non-SB rows. 95% percentile CIs.
"""
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

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4
N_BOOT = 500

T_CRIT_4 = 2.776
HALFWIDTH_95 = T_CRIT_4 / math.sqrt(5)


def load_sb_records():
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
        src = os.path.basename(sdir.rstrip("/"))
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            rnum = int(os.path.basename(path).replace("round_", "")
                       .replace(".pt", ""))
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i, (lab, tob) in enumerate(
                    zip(data["labels"].tolist(),
                        data["turns_of_breach"])):
                out[(src, rnum, i)] = {
                    "hs": arr[i], "label": bool(lab),
                    "turn_of_breach": tob,
                }
    return out


def main():
    sb = load_sb_records()
    sb_keys = set(r["key"] for r in sb)
    n_wins = sum(r["unsafe"] for r in sb)
    print(f"SB records: {len(sb)} (wins={n_wins})", flush=True)

    summary = {"layers": LAYERS, "harm_thresh": HARM_THRESH,
               "n_boot": N_BOOT, "per_layer": {}}

    rng = np.random.default_rng(42)

    for L in LAYERS:
        print(f"\nLayer L{L}...", flush=True)
        pool = load_pool_layer(L)

        # train probe on non-SB per-turn rows
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
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X_tr, y_tr)

        # SB test scores
        X_te = np.stack([pool[r["key"]]["hs"][r["rated_turn"]] for r in sb])
        probe = clf.predict_proba(X_te)[:, 1]
        for r, s in zip(sb, probe):
            r["probe"] = float(s)

        # point estimates
        wins = [r for r in sb if r["unsafe"]]
        wp = np.array([r["probe"] for r in wins])
        wh = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)
        auc_w_point = (roc_auc_score(wh, wp)
                       if wh.sum() > 0 and (1 - wh).sum() > 0
                       else np.nan)

        ap = np.array([r["probe"] for r in sb])
        ah = np.array([r["harm"] >= HARM_THRESH for r in sb], dtype=int)
        auc_a_point = (roc_auc_score(ah, ap)
                       if ah.sum() > 0 and (1 - ah).sum() > 0
                       else np.nan)

        # bootstrap
        aucs_w = []
        aucs_a = []
        for b in range(N_BOOT):
            idx_w = rng.integers(0, len(wins), size=len(wins))
            wp_b = wp[idx_w]; wh_b = wh[idx_w]
            if wh_b.sum() > 0 and (1 - wh_b).sum() > 0:
                aucs_w.append(roc_auc_score(wh_b, wp_b))

            idx_a = rng.integers(0, len(sb), size=len(sb))
            ap_b = ap[idx_a]; ah_b = ah[idx_a]
            if ah_b.sum() > 0 and (1 - ah_b).sum() > 0:
                aucs_a.append(roc_auc_score(ah_b, ap_b))

        aucs_w = np.array(aucs_w); aucs_a = np.array(aucs_a)
        summary["per_layer"][f"L{L}"] = {
            "harm_auc_wins_point": float(auc_w_point),
            "harm_auc_wins_lo95": float(np.percentile(aucs_w, 2.5)),
            "harm_auc_wins_hi95": float(np.percentile(aucs_w, 97.5)),
            "harm_auc_all_point": float(auc_a_point),
            "harm_auc_all_lo95": float(np.percentile(aucs_a, 2.5)),
            "harm_auc_all_hi95": float(np.percentile(aucs_a, 97.5)),
        }
        print(f"  harm AUC wins  : {auc_w_point:.3f}  "
              f"[{np.percentile(aucs_w, 2.5):.3f}, "
              f"{np.percentile(aucs_w, 97.5):.3f}]", flush=True)
        print(f"  harm AUC all600: {auc_a_point:.3f}  "
              f"[{np.percentile(aucs_a, 2.5):.3f}, "
              f"{np.percentile(aucs_a, 97.5):.3f}]", flush=True)

        del pool, X_tr, y_tr, X_te
        import gc; gc.collect()

    with open(f"{OUT}/per_layer_harm_with_cis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT}/per_layer_harm_with_cis.json")

    # ----- plot -----
    comp = json.load(open(f"{OUT}/per_layer_harm_vs_compliance_600.json"))
    comp_aucs = [comp["per_layer"][f"L{L}"]["auc"] for L in LAYERS]
    comp_stds = [comp["per_layer"][f"L{L}"]["std"] for L in LAYERS]
    comp_ci = [HALFWIDTH_95 * s for s in comp_stds]

    wins_pt = [summary["per_layer"][f"L{L}"]["harm_auc_wins_point"]
               for L in LAYERS]
    wins_lo = [wins_pt[i] - summary["per_layer"][f"L{L}"]["harm_auc_wins_lo95"]
               for i, L in enumerate(LAYERS)]
    wins_hi = [summary["per_layer"][f"L{L}"]["harm_auc_wins_hi95"] - wins_pt[i]
               for i, L in enumerate(LAYERS)]
    all_pt = [summary["per_layer"][f"L{L}"]["harm_auc_all_point"]
              for L in LAYERS]
    all_lo = [all_pt[i] - summary["per_layer"][f"L{L}"]["harm_auc_all_lo95"]
              for i, L in enumerate(LAYERS)]
    all_hi = [summary["per_layer"][f"L{L}"]["harm_auc_all_hi95"] - all_pt[i]
              for i, L in enumerate(LAYERS)]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.errorbar(LAYERS, comp_aucs, yerr=comp_ci,
                marker="o", markersize=6, linewidth=1.7,
                color="#3b6fb0", capsize=3,
                label="general compliance AUC  (600-conv subset, "
                      "GroupKFold $t_4$ CI)")
    ax.errorbar(LAYERS, wins_pt, yerr=[wins_lo, wins_hi],
                marker="s", markersize=6, linewidth=1.7,
                color="#d65a31", capsize=3,
                label="harm AUC within wins  (n=289, bootstrap 95% CI)")
    ax.errorbar(LAYERS, all_pt, yerr=[all_lo, all_hi],
                marker="^", markersize=6, linewidth=1.4,
                color="#7f7f7f", linestyle=":", capsize=3,
                label="harm AUC all 600  (bootstrap 95% CI)")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_xlabel("Llama-3.1-8B-Instruct layer")
    ax.set_ylabel("AUC")
    ax.set_title("Compliance probe AUC vs harm-severity AUC, per layer\n"
                 "(all three curves with 95% CIs)", fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_ylim(0.42, 0.85)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/per_layer_harm_vs_compliance_cis.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
