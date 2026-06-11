"""Per-layer AUC: (a) probe predicting general compliance on per-turn rows
of the 9,400 pool (GroupKFold per-turn-label, already computed), and
(b) the same probe's score predicting high-harm (Likert >= 4) vs low-harm
(Likert <= 3) among the n=289 Stage B wins.

For each layer L in {0, 4, 8, 12, 16, 20, 24, 28, 31}:
  - take per-layer compliance AUC from per_turn_label_per_category_groupkfold.json
  - train LR on per-turn rows from non-Stage-B convs at layer L (per-turn label)
  - score on Stage B win HS at the breach turn at layer L
  - compute AUC for harm >= 4 vs harm <= 3

Plot both as functions of layer.
"""
import glob
import json
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
HARM_THRESH = 4  # high = harm >= 4 (Likert 4 or 5); low = 1, 2, or 3


def load_layer_pool(layer):
    """Per-conversation entries: hs (5, 4096), label, tob, category, key."""
    out = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        src = os.path.basename(sdir.rstrip("/"))
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            rnum = int(os.path.basename(path).replace("round_", "")
                       .replace(".pt", ""))
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i, (lab, tob, cat) in enumerate(
                    zip(data["labels"].tolist(),
                        data["turns_of_breach"],
                        data["categories"])):
                out[(src, rnum, i)] = {
                    "hs": arr[i],
                    "label": bool(lab),
                    "turn_of_breach": tob,
                    "category": cat,
                }
    return out


def load_stage_b_meta():
    """Returns list of dicts with key, rated_turn, unsafe, harm, category."""
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


def main():
    sb = load_stage_b_meta()
    sb_keys = set(r["key"] for r in sb)
    print(f"Stage B records matched: {len(sb)} "
          f"(n_wins={sum(r['unsafe'] for r in sb)})", flush=True)

    # compliance AUC per layer from the GroupK json
    gkf = json.load(open(f"{OUT}/per_turn_label_per_category_groupkfold.json"))
    comp_aucs = []
    comp_stds = []
    for L in LAYERS:
        cell = gkf["global_per_layer"].get(f"L{L}")
        if cell:
            comp_aucs.append(cell["auc"]); comp_stds.append(cell["std"])
        else:
            comp_aucs.append(np.nan); comp_stds.append(np.nan)

    # per-layer harm AUC: train probe, predict on SB wins, compute high-vs-low harm AUC
    harm_aucs = []
    harm_aucs_all600 = []
    for L in LAYERS:
        print(f"\nLayer L{L}...", flush=True)
        pool = load_layer_pool(L)

        # training rows: per-turn from non-SB conversations
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

        # SB test rows
        X_te = np.stack([pool[r["key"]]["hs"][r["rated_turn"]] for r in sb])

        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        probe = clf.predict_proba(X_te)[:, 1]
        wins = [r for r in sb if r["unsafe"]]
        wins_probe = np.array([probe[i] for i, r in enumerate(sb)
                               if r["unsafe"]])
        wins_harm_hi = np.array([r["harm"] >= HARM_THRESH for r in wins],
                                 dtype=int)
        if wins_harm_hi.sum() > 0 and (1 - wins_harm_hi).sum() > 0:
            auc_h = roc_auc_score(wins_harm_hi, wins_probe)
        else:
            auc_h = np.nan
        harm_aucs.append(auc_h)

        all_harm_hi = np.array([r["harm"] >= HARM_THRESH for r in sb],
                               dtype=int)
        if all_harm_hi.sum() > 0 and (1 - all_harm_hi).sum() > 0:
            auc_a = roc_auc_score(all_harm_hi, probe)
        else:
            auc_a = np.nan
        harm_aucs_all600.append(auc_a)

        print(f"  compliance AUC (per-turn pool, GroupK) = {comp_aucs[LAYERS.index(L)]:.3f}")
        print(f"  harm AUC (within {len(wins)} wins, "
              f"high={int(wins_harm_hi.sum())}/low={int((1-wins_harm_hi).sum())}) = {auc_h:.3f}")
        print(f"  harm AUC (all 600, "
              f"high={int(all_harm_hi.sum())}/low={int((1-all_harm_hi).sum())}) = {auc_a:.3f}")

        del pool, X_tr, y_tr, X_te
        import gc; gc.collect()

    summary = {
        "layers": LAYERS,
        "harm_thresh": HARM_THRESH,
        "compliance_auc_per_layer": list(map(float, comp_aucs)),
        "compliance_std_per_layer": list(map(float, comp_stds)),
        "harm_auc_within_wins": list(map(float, harm_aucs)),
        "harm_auc_all600": list(map(float, harm_aucs_all600)),
        "n_wins": sum(r["unsafe"] for r in sb),
        "n_losses": sum(not r["unsafe"] for r in sb),
        "n_wins_high_harm": sum(r["unsafe"] and r["harm"] >= HARM_THRESH for r in sb),
        "n_wins_low_harm": sum(r["unsafe"] and r["harm"] < HARM_THRESH for r in sb),
    }
    with open(f"{OUT}/per_layer_harm_vs_compliance.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT}/per_layer_harm_vs_compliance.json", flush=True)

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    halfwidth = 2.776 / np.sqrt(5)
    ci_comp = [halfwidth * s for s in comp_stds]
    ax.errorbar(LAYERS, comp_aucs, yerr=ci_comp,
                marker="o", markersize=6, linewidth=1.7,
                color="#3b6fb0", capsize=3,
                label=f"general compliance AUC (per-turn pool n≈42k, "
                      f"GroupKFold)")
    ax.plot(LAYERS, harm_aucs, marker="s", markersize=6, linewidth=1.7,
            color="#d65a31",
            label=f"harm AUC within wins  (n={summary['n_wins']}, "
                  f"high≥{HARM_THRESH}: {summary['n_wins_high_harm']}, "
                  f"low: {summary['n_wins_low_harm']})")
    ax.plot(LAYERS, harm_aucs_all600, marker="^", markersize=6, linewidth=1.4,
            color="#7f7f7f", linestyle=":",
            label=f"harm AUC all 600  (high: "
                  f"{sum(r['harm'] >= HARM_THRESH for r in sb)}, "
                  f"low: {sum(r['harm'] < HARM_THRESH for r in sb)})")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_xlabel("Llama-3.1-8B-Instruct layer")
    ax.set_ylabel("AUC")
    ax.set_title("Compliance probe AUC vs harm-severity AUC, per layer\n"
                 "(same probe trained for compliance; harm-AUC measures "
                 "decoupling)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_ylim(0.45, 0.85)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/per_layer_harm_vs_compliance.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
