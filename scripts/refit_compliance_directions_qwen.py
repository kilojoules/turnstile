"""Re-fit LR-comp and MD-comp at all 9 layers using Qwen per-turn JBB labels.

Parallel to scripts/refit_harm_directions_llama.py. Same recipe as
scripts/layer_sweep_prep.py:load_compliance but reads alt label source
from working/uplift/.. — here: experiments/steering_v3/layer_sweep/
qwen_per_turn_compliance.jsonl.

Outputs:
  experiments/steering_v3/layer_sweep/directions_qwen_comp/v_{lr,md}_comp_L{L}.pt
  experiments/steering_v3/layer_sweep/metadata_qwen_comp.json (AUCs)
"""
import gc
import glob
import json
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/steering_v3/layer_sweep"
DIRS_QWEN = f"{OUT}/directions_qwen_comp"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
SEED = 42

os.makedirs(DIRS_QWEN, exist_ok=True)


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_qwen_labels():
    """Returns dict (source, round, idx) -> (qwen_unsafe: bool, qwen_turn_of_breach: int|None)."""
    out = {}
    for line in open(f"{OUT}/qwen_per_turn_compliance.jsonl"):
        r = json.loads(line)
        out[(r["source"], int(r["round"]), int(r["idx"]))] = (
            bool(r.get("qwen_unsafe")),
            r.get("qwen_turn_of_breach"),
        )
    return out


def load_compliance_qwen(layer, sb_keys, qwen_labels):
    """Same shape as layer_sweep_prep.load_compliance but with Qwen alt labels.

    For each conv: breach=qwen_unsafe, turn_of_breach=qwen_turn_of_breach.
    For breach convs: y=1 at the breach turn, y=0 at earlier turns.
    For non-breach: y=0 for all 5 turns.
    """
    X, y, groups = [], [], []
    cid = 0
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False, map_location="cpu")
            arr = data["hidden_states_by_layer"][layer].numpy()
            n_convs = len(data["labels"])
            for i in range(n_convs):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_", "").replace(".pt", ""))
                if (src, rnum, i) in sb_keys:
                    continue
                key = (src, rnum, i)
                if key not in qwen_labels:
                    continue  # not in pool we judged
                breach, t_star = qwen_labels[key]
                if breach and t_star is None:
                    continue  # malformed
                t_max = 4 if not breach else int(t_star)
                t_max = min(t_max, arr.shape[1] - 1)
                conv_id = cid; cid += 1
                for t in range(t_max + 1):
                    X.append(arr[i, t])
                    y.append(1 if (breach and t == t_star) else 0)
                    groups.append(conv_id)
            del data, arr; gc.collect()
    return np.stack(X), np.array(y, dtype=int), np.array(groups)


def comp_cv_auc(X, y, groups, n_splits=5):
    pos_groups = np.unique(groups[y == 1])
    neg_groups = np.unique(groups[y == 0])
    k = min(n_splits, len(pos_groups), len(neg_groups))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    aucs_lr, aucs_md = [], []
    for tr, te in sgkf.split(X, y, groups=groups):
        rng = np.random.default_rng(SEED)
        tr_pos = tr[y[tr] == 1]
        tr_neg = tr[y[tr] == 0]
        tr_neg_keep = rng.choice(tr_neg, size=min(len(tr_pos), len(tr_neg)), replace=False)
        tr_bal = np.concatenate([tr_pos, tr_neg_keep])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr_bal], y[tr_bal])
        aucs_lr.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        mu_pos = X[tr_bal][y[tr_bal] == 1].mean(0)
        mu_neg = X[tr_bal][y[tr_bal] == 0].mean(0)
        v_md = normalize(mu_pos - mu_neg)
        mu_tr = X[tr_bal].mean(0)
        scores = (X[te] - mu_tr) @ v_md
        aucs_md.append(roc_auc_score(y[te], scores))
    return float(np.mean(aucs_lr)), float(np.mean(aucs_md))


def main():
    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))
    print(f"Stage-B candidates (held out): {len(sb_keys)}")

    qwen_labels = load_qwen_labels()
    print(f"Qwen per-turn labels: {len(qwen_labels)} convs  "
          f"({sum(1 for v in qwen_labels.values() if v[0])} unsafe)")

    meta = {"layers": LAYERS, "per_layer": {}, "label_source": "qwen_per_turn_compliance"}

    print(f"\n{'L':>3}  {'n_rows':>8}  {'n_pos':>6}  {'LR AUC':>8}  {'MD AUC':>8}")
    print("-" * 50)
    for L in LAYERS:
        Xc, yc, gc_ = load_compliance_qwen(L, sb_keys, qwen_labels)
        n_pos = int(yc.sum())
        # Balanced fit on a balanced subsample (matches Phase 2 recipe)
        local_rng = np.random.default_rng(SEED + L)
        neg_idx = np.where(yc == 0)[0]
        neg_keep = local_rng.choice(neg_idx, size=n_pos, replace=False)
        keep = np.concatenate([np.where(yc == 1)[0], neg_keep])
        Xb, yb = Xc[keep], yc[keep]
        clf_c = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_c.fit(Xb, yb)
        v_lr_c = normalize(clf_c.coef_.ravel().astype(np.float32))
        v_md_c = normalize(Xb[yb == 1].mean(0) - Xb[yb == 0].mean(0)).astype(np.float32)
        auc_lr_c, auc_md_c = comp_cv_auc(Xc, yc, gc_)
        torch.save(torch.tensor(v_lr_c), f"{DIRS_QWEN}/v_lr_comp_L{L}.pt")
        torch.save(torch.tensor(v_md_c), f"{DIRS_QWEN}/v_md_comp_L{L}.pt")
        print(f"L{L:<2}  {len(Xc):>8}  {n_pos:>6}  {auc_lr_c:>8.4f}  {auc_md_c:>8.4f}")
        meta["per_layer"][f"L{L}"] = {
            "lr_comp_auc_qwen": auc_lr_c,
            "md_comp_auc_qwen": auc_md_c,
            "n_rows": int(len(Xc)),
            "n_pos": n_pos,
        }
        del Xc, yc, gc_, Xb, yb; gc.collect()

    with open(f"{OUT}/metadata_qwen_comp.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {OUT}/metadata_qwen_comp.json")


if __name__ == "__main__":
    main()
