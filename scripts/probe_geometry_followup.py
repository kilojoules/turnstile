"""Probe-geometry follow-up.

Analysis 1: mean-diff directions as probes (cross-validated AUC + cross-method
score correlation) for both compliance and harm at L16.

Analysis 2: continuous-Likert regression for harm. r² and Spearman ρ between
held-out projection and Likert; compares to binary AUC framing.

Held-out splits:
  - Compliance: 5-fold StratifiedGroupKFold by conversation_id (same as
    per_turn_label_per_category_groupkfold)
  - Harm: 5-fold StratifiedKFold on the 289 Stage B wins (same as
    harm_probe_experiments exp2)

For each fold: fit BOTH LR and mean-diff on train, evaluate on test, collect
out-of-fold scores. Mean-diff "score" = (h - μ_train) · v_md / ||v_md||;
LR score = clf.decision_function (which is h · w + b).
"""
import gc, glob, json, math, os
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/steering_v3"
LAYER = 16
HARM_THRESH = 4


def load_compliance(layer, sb_keys):
    X, y, groups = [], [], []
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
    return np.stack(X), np.array(y), np.array(groups)


def load_harm(layer):
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    wins = []
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok") or not r.get("unsafe"):
            continue
        parts = r["conv_id"].split("/")
        if len(parts) != 3:
            continue
        key = (parts[0], int(parts[1].lstrip("r")), int(parts[2].lstrip("i")))
        if key not in cand:
            continue
        conv = cand[key]["conversation"]
        asst = [i for i, t in enumerate(conv) if t["role"]=="assistant"]
        if not asst:
            continue
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
    y_bin = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)
    y_likert = np.array([r["harm"] for r in wins], dtype=float)
    return X, y_bin, y_likert


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def meandiff_direction(X_tr, y_tr):
    mu_pos = X_tr[y_tr == 1].mean(axis=0)
    mu_neg = X_tr[y_tr == 0].mean(axis=0)
    return normalize(mu_pos - mu_neg)


def main():
    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))

    # ========================== COMPLIANCE ==========================
    print("=== Loading compliance L16... ===")
    Xc, yc, gc_ = load_compliance(LAYER, sb_keys)
    print(f"  n={len(yc)}, n_pos={int(yc.sum())}, n_neg={int(len(yc)-yc.sum())}")

    pos_groups = np.unique(gc_[yc == 1])
    neg_groups = np.unique(gc_[yc == 0])
    k = min(5, len(pos_groups), len(neg_groups))
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)

    oof_lr_c = np.full(len(yc), np.nan)
    oof_md_c = np.full(len(yc), np.nan)
    for fold, (tr, te) in enumerate(skf.split(Xc, yc, groups=gc_)):
        # Subsample negatives for balanced training (so LR isn't dominated by class imbalance)
        # — but use unweighted LR + balanced data, matching the convention from probe_stability_tests
        rng = np.random.default_rng(42 + fold)
        tr_pos = tr[yc[tr] == 1]
        tr_neg_all = tr[yc[tr] == 0]
        tr_neg = rng.choice(tr_neg_all, size=min(len(tr_pos), len(tr_neg_all)), replace=False)
        tr_bal = np.concatenate([tr_pos, tr_neg])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Xc[tr_bal], yc[tr_bal])
        oof_lr_c[te] = clf.decision_function(Xc[te])

        v_md = meandiff_direction(Xc[tr_bal], yc[tr_bal])
        mu_tr = Xc[tr_bal].mean(axis=0)
        oof_md_c[te] = (Xc[te] - mu_tr) @ v_md
        print(f"  fold {fold+1}/{k}: tr_pos={len(tr_pos)}, te_size={len(te)}", flush=True)

    keep_c = ~np.isnan(oof_lr_c)
    auc_lr_c = roc_auc_score(yc[keep_c], oof_lr_c[keep_c])
    auc_md_c = roc_auc_score(yc[keep_c], oof_md_c[keep_c])
    pearson_c, _ = pearsonr(oof_lr_c[keep_c], oof_md_c[keep_c])
    print(f"\n  LR-comp AUC (OOB)     = {auc_lr_c:.4f}")
    print(f"  Mean-diff comp AUC (OOB) = {auc_md_c:.4f}")
    print(f"  Pearson r (LR scores vs mean-diff scores, OOB) = {pearson_c:+.4f}")

    del Xc, yc, gc_; gc.collect()

    # ========================== HARM ==========================
    print("\n=== Loading harm L16 (Stage B wins)... ===")
    Xh, yh_bin, yh_lik = load_harm(LAYER)
    print(f"  n={len(yh_bin)}, n_high={int(yh_bin.sum())}, n_low={int(len(yh_bin)-yh_bin.sum())}")
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lr_h = np.full(len(yh_bin), np.nan)
    oof_md_h = np.full(len(yh_bin), np.nan)
    for fold, (tr, te) in enumerate(skf2.split(Xh, yh_bin)):
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
        clf.fit(Xh[tr], yh_bin[tr])
        oof_lr_h[te] = clf.decision_function(Xh[te])

        v_md = meandiff_direction(Xh[tr], yh_bin[tr])
        mu_tr = Xh[tr].mean(axis=0)
        oof_md_h[te] = (Xh[te] - mu_tr) @ v_md

    auc_lr_h = roc_auc_score(yh_bin, oof_lr_h)
    auc_md_h = roc_auc_score(yh_bin, oof_md_h)
    pearson_h, _ = pearsonr(oof_lr_h, oof_md_h)
    print(f"\n  LR-harm AUC (OOB)     = {auc_lr_h:.4f}")
    print(f"  Mean-diff harm AUC (OOB) = {auc_md_h:.4f}")
    print(f"  Pearson r (LR vs mean-diff, OOB) = {pearson_h:+.4f}")

    # Analysis 2: continuous Likert regression
    # Use the same OOB scores from the binary harm probe
    print("\n=== Analysis 2: continuous Likert regression (harm only) ===")
    # Linear regression: y_likert ~ score; r^2 and Spearman
    def regress_stats(scores, likert):
        # r^2 from linear regression
        x = scores - scores.mean()
        y = likert - likert.mean()
        if np.linalg.norm(x) < 1e-12:
            return float('nan'), float('nan'), float('nan')
        slope = (x @ y) / (x @ x)
        pred = slope * (scores - scores.mean()) + likert.mean()
        ss_res = float(np.sum((likert - pred) ** 2))
        ss_tot = float(np.sum((likert - likert.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot
        rho, _ = spearmanr(scores, likert)
        pearson, _ = pearsonr(scores, likert)
        return r2, float(rho), float(pearson)

    r2_lr, rho_lr, pear_lr = regress_stats(oof_lr_h, yh_lik)
    r2_md, rho_md, pear_md = regress_stats(oof_md_h, yh_lik)

    print(f"\n  LR-harm score vs Likert: r²={r2_lr:.4f}, Spearman={rho_lr:+.4f}, Pearson={pear_lr:+.4f}")
    print(f"  Mean-diff-harm score vs Likert: r²={r2_md:.4f}, Spearman={rho_md:+.4f}, Pearson={pear_md:+.4f}")

    # ----- output markdown -----
    md = []
    md.append("## Analysis 1: Mean-diff directions as probes (cross-validated AUC)")
    md.append("")
    md.append("All cross-validated with the same fold splits LR was evaluated on.")
    md.append("")
    md.append("| Concept | LR AUC | Mean-diff AUC | Pearson r (scores) | Flag |")
    md.append("|---|---|---|---|---|")
    flag_c = " ⚠️ AUC < 0.6" if auc_md_c < 0.6 else ""
    flag_h = " ⚠️ AUC < 0.6" if auc_md_h < 0.6 else ""
    md.append(f"| Compliance (per-turn, 9,400-conv pool) | {auc_lr_c:.3f} | {auc_md_c:.3f}{flag_c} | {pearson_c:+.3f} | |")
    md.append(f"| Harm (Stage B wins, Likert ≥4 vs ≤3) | {auc_lr_h:.3f} | {auc_md_h:.3f}{flag_h} | {pearson_h:+.3f} | |")
    md.append("")
    md.append("## Analysis 2: Continuous Likert regression (harm only)")
    md.append("")
    md.append("| Direction | r² | Spearman ρ | Pearson r | Flag |")
    md.append("|---|---|---|---|---|")
    flagr_lr = " ⚠️ r² < 0.2 (threshold not magnitude?)" if r2_lr < 0.2 else ""
    flagr_md = " ⚠️ r² < 0.2 (threshold not magnitude?)" if r2_md < 0.2 else ""
    md.append(f"| LR-harm projection vs Likert | {r2_lr:.3f}{flagr_lr} | {rho_lr:+.3f} | {pear_lr:+.3f} | |")
    md.append(f"| Mean-diff-harm projection vs Likert | {r2_md:.3f}{flagr_md} | {rho_md:+.3f} | {pear_md:+.3f} | |")

    out_text = "\n".join(md)
    with open(f"{OUT}/probe_geometry_followup.md", "w") as f:
        f.write(out_text)
    print("\n" + out_text)
    print(f"\nWrote {OUT}/probe_geometry_followup.md")


if __name__ == "__main__":
    main()
