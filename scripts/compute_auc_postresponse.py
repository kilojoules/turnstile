"""Compute per-layer LR + MD AUC at BOTH loci from postresponse_alllayer/reps.npz,
replicating layer_sweep_prep's comp_cv_auc / harm_cv_auc exactly (only the input
matrix changes: *_in_L = pre-response prior, *_out_L = response-mean posterior).

Writes experiments/postresponse_alllayer/auc_by_layer.json and prints a table.
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
D = f"{ROOT}/experiments/postresponse_alllayer"
SEED = 0
# harm + compliance are saved separately (harm-first, comp checkpointed); merge them.
ZH = np.load(f"{D}/reps_harm.npz")
ZC = np.load(f"{D}/reps_comp.npz")
Z = {k: ZH[k] for k in ZH.files}
Z.update({k: ZC[k] for k in ZC.files})
LAYERS = list(Z["layers"])
norm = lambda v: v / max(np.linalg.norm(v), 1e-12)


def comp_cv_auc(X, y, groups, n_splits=5):
    pos_g, neg_g = np.unique(groups[y == 1]), np.unique(groups[y == 0])
    k = min(n_splits, len(pos_g), len(neg_g))
    if k < 2:
        return float("nan"), float("nan")
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    lr, md = [], []
    for tr, te in sgkf.split(X, y, groups=groups):
        rng = np.random.default_rng(SEED)
        tp, tn = tr[y[tr] == 1], tr[y[tr] == 0]
        keep = rng.choice(tn, size=min(len(tp), len(tn)), replace=False)
        b = np.concatenate([tp, keep])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X[b], y[b])
        lr.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        v = norm(X[b][y[b] == 1].mean(0) - X[b][y[b] == 0].mean(0))
        md.append(roc_auc_score(y[te], (X[te] - X[b].mean(0)) @ v))
    return float(np.mean(lr)), float(np.mean(md))


def harm_cv_auc(X, y, groups, n_splits=5):
    # grouped by goal so the probe can't leak goal identity across folds (honest AUC)
    k = min(n_splits, len(np.unique(groups[y == 1])), len(np.unique(groups[y == 0])))
    if k < 2:
        return float("nan"), float("nan")
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    lr, md = [], []
    for tr, te in sgkf.split(X, y, groups=groups):
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs").fit(X[tr], y[tr])
        lr.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        v = norm(X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0))
        md.append(roc_auc_score(y[te], (X[te] - X[tr].mean(0)) @ v))
    return float(np.mean(lr)), float(np.mean(md))


res = {"layers": [int(L) for L in LAYERS], "comp": {}, "harm": {}}
cyl, cyq, cgrp = Z["comp_yl"], Z["comp_yq"], Z["comp_grp"]
hrl, hrq, hgrp = Z["harm_rl"], Z["harm_rq"], Z["harm_grp"]
_mL = (hrl >= 4) | (hrl <= 2)
print(f"comp n={len(cyl)} complied={int((cyl==1).sum())} qwen_labeled={int((cyq>=0).sum())} | "
      f"harm FULL corpus n={len(hrl)}; >=4-vs-<=2 usable={int(_mL.sum())} (pos={int((hrl>=4).sum())}/neg={int((hrl<=2).sum())}, rating3 dropped={int(((hrl==3)).sum())})\n")

# affirm the seam is closed: the all-data L16-output MD on this corpus == the steered harm_dm_llama
try:
    import torch
    _d = (lambda X, y: X[y == 1].mean(0) - X[y == 0].mean(0))(Z["harm_out_L16"][_mL], (hrl[_mL] >= 4).astype(int))
    _hd = torch.load(f"{ROOT}/directions/harm_dm_llama.pt").float().numpy()
    _c = float((_d/np.linalg.norm(_d)) @ (_hd/np.linalg.norm(_hd)))
    print(f"[seam check] cos(all-data L16-out MD on this corpus, steered harm_dm_llama) = {_c:.4f}  (1.000 => identical estimator/definition)\n")
except Exception as e:
    print(f"[seam check] skipped: {e}\n")

hdr = f"{'L':>3} | {'cLR_in':>7} {'cLR_out':>7} | {'cMD_in':>7} {'cMD_out':>7} || {'hLR_in':>7} {'hLR_out':>7} | {'hMD_in':>7} {'hMD_out':>7}"
print(hdr); print("-" * len(hdr))
for L in LAYERS:
    L = int(L)
    ci, co = Z[f"comp_in_L{L}"], Z[f"comp_out_L{L}"]
    hi, ho = Z[f"harm_in_L{L}"], Z[f"harm_out_L{L}"]
    # compliance (Llama labels; Qwen labels where available)
    cl_in = comp_cv_auc(ci, cyl, cgrp); cl_out = comp_cv_auc(co, cyl, cgrp)
    m = cyq >= 0
    cq_in = comp_cv_auc(ci[m], cyq[m], cgrp[m]); cq_out = comp_cv_auc(co[m], cyq[m], cgrp[m])
    # harm (>=4 vs <=2, dropping rating 3) on the FULL-600 corpus, grouped by goal
    # -> the per-fold MD is exactly the steered harm_dm_llama recipe.
    mL = (hrl >= 4) | (hrl <= 2); mQ = (hrq >= 4) | (hrq <= 2)
    hl_in = harm_cv_auc(hi[mL], (hrl[mL] >= 4).astype(int), hgrp[mL]); hl_out = harm_cv_auc(ho[mL], (hrl[mL] >= 4).astype(int), hgrp[mL])
    hq_in = harm_cv_auc(hi[mQ], (hrq[mQ] >= 4).astype(int), hgrp[mQ]); hq_out = harm_cv_auc(ho[mQ], (hrq[mQ] >= 4).astype(int), hgrp[mQ])
    res["comp"][f"L{L}"] = {"lr_in": cl_in[0], "lr_out": cl_out[0], "md_in": cl_in[1], "md_out": cl_out[1],
                            "lr_in_q": cq_in[0], "lr_out_q": cq_out[0], "md_in_q": cq_in[1], "md_out_q": cq_out[1]}
    res["harm"][f"L{L}"] = {"lr_in": hl_in[0], "lr_out": hl_out[0], "md_in": hl_in[1], "md_out": hl_out[1],
                            "lr_in_q": hq_in[0], "lr_out_q": hq_out[0], "md_in_q": hq_in[1], "md_out_q": hq_out[1]}
    print(f"L{L:>2} | {cl_in[0]:7.3f} {cl_out[0]:7.3f} | {cl_in[1]:7.3f} {cl_out[1]:7.3f} || "
          f"{hl_in[0]:7.3f} {hl_out[0]:7.3f} | {hl_in[1]:7.3f} {hl_out[1]:7.3f}")

json.dump(res, open(f"{D}/auc_by_layer.json", "w"), indent=2)
print(f"\nwrote {D}/auc_by_layer.json")
print(f"[harm, FULL-600 >=4-vs-<=2] L16  LR prior={res['harm']['L16']['lr_in']:.3f} posterior={res['harm']['L16']['lr_out']:.3f} | "
      f"MD prior={res['harm']['L16']['md_in']:.3f} posterior={res['harm']['L16']['md_out']:.3f}")
