"""Phase 1 prep for the layer-sweep steering experiment.

For each layer in {0, 4, 8, 12, 16, 20, 24, 28, 31}:
  - Train LR-compliance (downsampled balanced training)
  - Compute mean-diff compliance (on same balanced subsample, for direct comparison)
  - Train LR-harm (on Stage B wins, class_weight='balanced')
  - Compute mean-diff harm
  - Generate one random unit vector (per-layer seed for reproducibility)
  - Record AUCs for all four real directions (cross-validated)
  - Measure h_norm = median residual stream norm at last-token-of-prompt
    over all pooled_hs records (~47k measurements per layer)

Output: experiments/steering_v3/layer_sweep/
  directions/v_{lr_comp,md_comp,lr_harm,md_harm,random}_L{layer}.pt  (45 files)
  metadata.json (h_norms + AUCs + sample sizes)
"""
import gc, glob, json, math, os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/steering_v3/layer_sweep"
DIRS = f"{OUT}/directions"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4
SEED = 42

os.makedirs(DIRS, exist_ok=True)


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_compliance(layer, sb_keys):
    """Returns (X, y, groups, h_norms_at_layer)."""
    X, y, groups, norms = [], [], [], []
    cid = 0
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            labels = data["labels"].tolist()
            tobs = data["turns_of_breach"]
            for i in range(len(labels)):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_","").replace(".pt",""))
                # h_norms: sum all turn HS for norm calculation (regardless of SB)
                for t in range(5):
                    norms.append(float(np.linalg.norm(arr[i, t])))
                if (src, rnum, i) in sb_keys:
                    continue
                breach = bool(labels[i])
                t_star = tobs[i] if breach else None
                if breach and t_star is None:
                    continue
                t_max = 4 if not breach else int(t_star)
                conv_id = cid; cid += 1
                for t in range(t_max + 1):
                    X.append(arr[i, t])
                    y.append(1 if (breach and t == t_star) else 0)
                    groups.append(conv_id)
            del data, arr; gc.collect()
    return np.stack(X), np.array(y), np.array(groups), np.array(norms)


def load_harm(layer):
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    wins = []
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok") or not r.get("unsafe"): continue
        parts = r["conv_id"].split("/")
        if len(parts) != 3: continue
        key = (parts[0], int(parts[1].lstrip("r")), int(parts[2].lstrip("i")))
        if key not in cand: continue
        conv = cand[key]["conversation"]
        asst = [i for i, t in enumerate(conv) if t["role"]=="assistant"]
        if not asst: continue
        if r.get("turn_of_breach") is not None and int(r["turn_of_breach"]) < len(asst):
            rated = int(r["turn_of_breach"])
        else:
            rated = len(asst)-1
        wins.append({"key": key, "rated_turn": rated, "harm": r["rating"]})
    win_keys = {r["key"] for r in wins}
    hs_by_key = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i in range(len(data["labels"])):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_","").replace(".pt",""))
                key = (src, rnum, i)
                if key in win_keys:
                    hs_by_key[key] = arr[i]
            del data, arr; gc.collect()
    X = np.stack([hs_by_key[r["key"]][r["rated_turn"]] for r in wins])
    y = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)
    return X, y


def comp_cv_auc(X, y, groups, n_splits=5):
    pos_groups = np.unique(groups[y == 1])
    neg_groups = np.unique(groups[y == 0])
    k = min(n_splits, len(pos_groups), len(neg_groups))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    aucs_lr, aucs_md = [], []
    for tr, te in sgkf.split(X, y, groups=groups):
        rng = np.random.default_rng(SEED)
        tr_pos = tr[y[tr] == 1]; tr_neg = tr[y[tr] == 0]
        tr_neg_keep = rng.choice(tr_neg, size=min(len(tr_pos), len(tr_neg)), replace=False)
        tr_bal = np.concatenate([tr_pos, tr_neg_keep])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr_bal], y[tr_bal])
        aucs_lr.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        mu_pos = X[tr_bal][y[tr_bal]==1].mean(0)
        mu_neg = X[tr_bal][y[tr_bal]==0].mean(0)
        v_md = normalize(mu_pos - mu_neg)
        mu_tr = X[tr_bal].mean(0)
        scores = (X[te] - mu_tr) @ v_md
        aucs_md.append(roc_auc_score(y[te], scores))
    return float(np.mean(aucs_lr)), float(np.mean(aucs_md))


def harm_cv_auc(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs_lr, aucs_md = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aucs_lr.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        mu_pos = X[tr][y[tr]==1].mean(0)
        mu_neg = X[tr][y[tr]==0].mean(0)
        v_md = normalize(mu_pos - mu_neg)
        mu_tr = X[tr].mean(0)
        scores = (X[te] - mu_tr) @ v_md
        aucs_md.append(roc_auc_score(y[te], scores))
    return float(np.mean(aucs_lr)), float(np.mean(aucs_md))


def main():
    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))

    meta = {"layers": LAYERS, "per_layer": {}}
    rng = np.random.default_rng(SEED)
    for L in LAYERS:
        print(f"\n=== Layer L{L} ===", flush=True)
        # Compliance
        Xc, yc, gc_, norms = load_compliance(L, sb_keys)
        h_norm = float(np.median(norms))
        print(f"  h_norm L{L} (median, n={len(norms)} measurements) = {h_norm:.3f}")
        n_pos = int(yc.sum())
        # Balanced subsample for both LR and mean-diff at this layer
        local_rng = np.random.default_rng(SEED + L)
        neg_keep = local_rng.choice(np.where(yc == 0)[0], size=n_pos, replace=False)
        keep = np.concatenate([np.where(yc == 1)[0], neg_keep])
        Xb, yb = Xc[keep], yc[keep]
        # LR (final model on full balanced subsample)
        clf_c = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_c.fit(Xb, yb)
        v_lr_c = normalize(clf_c.coef_.ravel().astype(np.float32))
        # Mean-diff on same balanced subsample
        v_md_c = normalize(Xb[yb==1].mean(0) - Xb[yb==0].mean(0)).astype(np.float32)
        # CV AUC
        auc_lr_c, auc_md_c = comp_cv_auc(Xc, yc, gc_)
        print(f"  Compliance: LR AUC = {auc_lr_c:.4f}, mean-diff AUC = {auc_md_c:.4f}")
        torch.save(torch.tensor(v_lr_c), f"{DIRS}/v_lr_comp_L{L}.pt")
        torch.save(torch.tensor(v_md_c), f"{DIRS}/v_md_comp_L{L}.pt")
        del Xc, yc, gc_, Xb, yb, norms; gc.collect()

        # Harm
        Xh, yh = load_harm(L)
        clf_h = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
        clf_h.fit(Xh, yh)
        v_lr_h = normalize(clf_h.coef_.ravel().astype(np.float32))
        v_md_h = normalize(Xh[yh==1].mean(0) - Xh[yh==0].mean(0)).astype(np.float32)
        auc_lr_h, auc_md_h = harm_cv_auc(Xh, yh)
        print(f"  Harm: LR AUC = {auc_lr_h:.4f}, mean-diff AUC = {auc_md_h:.4f}")
        torch.save(torch.tensor(v_lr_h), f"{DIRS}/v_lr_harm_L{L}.pt")
        torch.save(torch.tensor(v_md_h), f"{DIRS}/v_md_harm_L{L}.pt")
        del Xh, yh; gc.collect()

        # Random direction (per-layer)
        v_rand = rng.standard_normal(4096).astype(np.float32)
        v_rand = normalize(v_rand)
        torch.save(torch.tensor(v_rand), f"{DIRS}/v_random_L{L}.pt")

        meta["per_layer"][f"L{L}"] = {
            "h_norm_median": h_norm,
            "lr_comp_auc": auc_lr_c,
            "md_comp_auc": auc_md_c,
            "lr_harm_auc": auc_lr_h,
            "md_harm_auc": auc_md_h,
        }

    with open(f"{OUT}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {OUT}/metadata.json")
    print(f"Saved {len(LAYERS)*5} direction vectors to {DIRS}/")


if __name__ == "__main__":
    main()
