"""Phase 2: re-fit LR-harm and MD-harm at all 9 layers using Llama Stage-B labels.

Compliance directions are unchanged — only harm directions are re-fit.

Inputs:
  working/uplift/stage_b_scores_llama.jsonl  (Phase 1.2 Llama labels)
  working/uplift/stage_b_candidates.jsonl    (the 600-conv pool)
  experiments/pooled_hs/{source}/round_*.pt  (residuals; on disk, no harvest)

Outputs:
  experiments/steering_v3/layer_sweep/directions_llama/v_{lr,md}_harm_L{L}.pt  (18 files)
  experiments/steering_v3/layer_sweep/metadata_llama.json  (new AUCs + the
    AUC delta vs the Qwen-fit metadata.json)

Uses the same fit recipe as scripts/layer_sweep_prep.py for harm directions:
  LR with class_weight='balanced', C=1.0, lbfgs
  Mean-diff = μ_{Likert≥4} − μ_{Likert<4}
  5-fold stratified CV for AUC
"""
import gc
import glob
import json
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/steering_v3/layer_sweep"
DIRS_LLAMA = f"{OUT}/directions_llama"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4
SEED = 42

os.makedirs(DIRS_LLAMA, exist_ok=True)


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_harm(layer, scores_path):
    """Same join logic as scripts/layer_sweep_prep.py:load_harm, but reads
    Stage-B scores from a configurable path so we can swap Qwen→Llama."""
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    wins = []
    for line in open(scores_path):
        r = json.loads(line)
        if not r.get("parse_ok") or not r.get("unsafe"):
            continue
        # Both Qwen and Llama files use the same conv_id format
        parts = r["conv_id"].split("/")
        if len(parts) != 3:
            continue
        key = (parts[0], int(parts[1].lstrip("r")), int(parts[2].lstrip("i")))
        if key not in cand:
            continue
        conv = cand[key]["conversation"]
        asst = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        if not asst:
            continue
        if r.get("turn_of_breach") is not None and int(r["turn_of_breach"]) < len(asst):
            rated = int(r["turn_of_breach"])
        else:
            rated = len(asst) - 1
        wins.append({"key": key, "rated_turn": rated, "harm": r["rating"]})

    win_keys = {r["key"] for r in wins}
    hs_by_key = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False, map_location="cpu")
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i in range(len(data["labels"])):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_", "").replace(".pt", ""))
                key = (src, rnum, i)
                if key in win_keys:
                    hs_by_key[key] = arr[i]
            del data, arr
            gc.collect()
    # If any wins lack residuals (e.g., not in pool), drop them
    wins = [w for w in wins if w["key"] in hs_by_key]
    X = np.stack([hs_by_key[w["key"]][w["rated_turn"]] for w in wins])
    y = np.array([w["harm"] >= HARM_THRESH for w in wins], dtype=int)
    return X, y, len(wins)


def cv_auc(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs_lr, aucs_md = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aucs_lr.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        mu_pos = X[tr][y[tr] == 1].mean(0)
        mu_neg = X[tr][y[tr] == 0].mean(0)
        v_md = normalize(mu_pos - mu_neg)
        mu_tr = X[tr].mean(0)
        scores = (X[te] - mu_tr) @ v_md
        aucs_md.append(roc_auc_score(y[te], scores))
    return float(np.mean(aucs_lr)), float(np.mean(aucs_md))


def main():
    scores_llama = f"{ROOT}/working/uplift/stage_b_scores_llama.jsonl"
    if not os.path.exists(scores_llama):
        raise FileNotFoundError(f"Llama scores not found at {scores_llama}")

    # Read Qwen metadata for comparison
    qwen_meta = json.load(open(f"{OUT}/metadata.json"))

    out_meta = {"layers": LAYERS, "judge": "llama70b",
                "scores_path": scores_llama, "per_layer": {}}

    print(f"\n=== fitting per-layer (LR + MD), HARM_THRESH={HARM_THRESH} ===\n")
    print(f"{'L':>3}  {'n_wins':>7}  {'n_pos':>6}  "
          f"{'LR_AUC_llama':>13}  {'LR_AUC_qwen':>13}  ΔAUC")
    print("-" * 78)
    for L in LAYERS:
        Xh, yh, n_wins = load_harm(L, scores_llama)
        n_pos = int(yh.sum())
        n_neg = int((1 - yh).sum())

        clf_h = LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=2000, solver="lbfgs")
        clf_h.fit(Xh, yh)
        v_lr_h = normalize(clf_h.coef_.ravel().astype(np.float32))
        v_md_h = normalize(Xh[yh == 1].mean(0) - Xh[yh == 0].mean(0)).astype(np.float32)
        auc_lr_h, auc_md_h = cv_auc(Xh, yh)

        torch.save(torch.tensor(v_lr_h), f"{DIRS_LLAMA}/v_lr_harm_L{L}.pt")
        torch.save(torch.tensor(v_md_h), f"{DIRS_LLAMA}/v_md_harm_L{L}.pt")

        qwen_auc_lr = qwen_meta["per_layer"][f"L{L}"]["lr_harm_auc"]
        delta = auc_lr_h - qwen_auc_lr
        print(f"L{L:<2}  {n_wins:>7}  {n_pos:>3}/{n_pos+n_neg:<3}  "
              f"{auc_lr_h:>13.4f}  {qwen_auc_lr:>13.4f}  {delta:+.4f}")

        out_meta["per_layer"][f"L{L}"] = {
            "h_norm_median": qwen_meta["per_layer"][f"L{L}"]["h_norm_median"],
            "lr_comp_auc": qwen_meta["per_layer"][f"L{L}"]["lr_comp_auc"],
            "md_comp_auc": qwen_meta["per_layer"][f"L{L}"]["md_comp_auc"],
            "lr_harm_auc_llama": auc_lr_h,
            "md_harm_auc_llama": auc_md_h,
            "lr_harm_auc_qwen": qwen_auc_lr,
            "md_harm_auc_qwen": qwen_meta["per_layer"][f"L{L}"]["md_harm_auc"],
            "n_wins_llama": n_wins,
            "n_pos_llama": n_pos,
            "n_neg_llama": n_neg,
        }
        del Xh, yh
        gc.collect()

    with open(f"{OUT}/metadata_llama.json", "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"\nWrote {OUT}/metadata_llama.json")
    print(f"Wrote {len(LAYERS) * 2} direction vectors to {DIRS_LLAMA}/")


if __name__ == "__main__":
    main()
