"""Task 2: pre-response layer-AUC of the ACTUALLY-STEERED vectors (refusal_dm for
compliance, harm_dm_llama for harm), so the paper's 'decodable at 0.7-0.9' numbers
describe the vectors we steer, not re-fit lookalikes.

A steered vector is FIXED, so its decode AUC = roc_auc_score(y, X @ d) at the
pre-response locus (residual at the prompt's last token). We report the point AUC
plus a 5-fold group-k-fold (groups=conversation) mean±std for a CI matching Fig 5.
Decodability is sign-free: report max(auc, 1-auc) and the sign used.

Corpus = pooled_hs (same as Fig 4/5). Compliance labels: Llama per-turn breach
(pooled_hs 'labels'/'turns_of_breach') and Qwen per-turn (qwen_per_turn_compliance).
Harm labels: Stage-B rating>=4 at the breach turn (Llama 'rating', Qwen 'qwen_rating').
"""
import glob, json, os
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = "/Users/julianquick/portfolio_copy/turnstile"; POOL = f"{ROOT}/experiments/pooled_hs"; L = 16
refusal = (lambda v: v/np.linalg.norm(v))(torch.load(f"{ROOT}/directions/refusal_dm.pt").float().numpy())
harmdir = (lambda v: v/np.linalg.norm(v))(torch.load(f"{ROOT}/directions/harm_dm_llama.pt").float().numpy())

def cv_auc(y, score, groups, seed=42):
    y = np.asarray(y); s = np.asarray(score); g = np.asarray(groups)
    pt = roc_auc_score(y, s); pt = max(pt, 1 - pt)
    k = min(5, len(np.unique(g[y == 1])), len(np.unique(g[y == 0])))
    aucs = []
    if k >= 2:
        for _, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(s.reshape(-1,1), y, g):
            if len(np.unique(y[te])) == 2:
                a = roc_auc_score(y[te], s[te]); aucs.append(max(a, 1-a))
    return pt, (np.mean(aucs) if aucs else float("nan")), (np.std(aucs) if aucs else float("nan")), int(y.sum()), len(y)

# ---- load pooled_hs L16 once: build compliance rows + harm-join keys ----
# Qwen per-turn breach map: (source,round,idx) -> qwen breach turn index
qmap = {}
for r in map(json.loads, open(f"{ROOT}/experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl")):
    qmap[(r["source"], r["round"], r["idx"])] = r.get("qwen_turn_of_breach")
# Stage-B harm ratings: (source,round,idx) -> (llama_rating, qwen_rating, breach_turn)
harm_lab = {}
for s in map(json.loads, open(f"{ROOT}/working/uplift/stage_b_scores_llama.jsonl")):
    harm_lab[(s["source"], s["round"], s["idx"])] = (s.get("rating"), s.get("qwen_rating"), s.get("turn_of_breach"))

Xc, yL, yQ, gc = [], [], [], []          # compliance per-turn (proj later)
Xh, hL, hQ, gh = [], [], [], []          # harm at rated breach turn
cid = 0
for sdir in sorted(glob.glob(f"{POOL}/*/")):
    src = os.path.basename(os.path.dirname(sdir))
    for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
        rnd = int(os.path.basename(path).replace("round_", "").replace(".pt", ""))
        d = torch.load(path, weights_only=False)
        arr = d["hidden_states_by_layer"][L].numpy(); labels = d["labels"].tolist(); tobs = d["turns_of_breach"]
        for i in range(len(labels)):
            breach = bool(labels[i]); tstar = tobs[i] if breach else None
            if breach and tstar is None: continue
            key = (src, rnd, i)
            qtob = qmap.get(key)
            tmax = 4 if not breach else int(tstar)
            g = cid; cid += 1
            for t in range(tmax + 1):
                Xc.append(arr[i, t]); gc.append(g)
                yL.append(1 if (breach and t == tstar) else 0)
                yQ.append(1 if (qtob is not None and t == qtob) else 0)
            # harm: at the (llama) breach turn, if rated
            if breach and key in harm_lab:
                rlr, rqr, btob = harm_lab[key]
                bt = int(btob) if isinstance(btob, int) and btob <= tstar else int(tstar)
                if isinstance(rlr, int) and bt <= tstar:
                    Xh.append(arr[i, bt]); gh.append(g)
                    hL.append(1 if rlr >= 4 else 0); hQ.append(1 if (isinstance(rqr, int) and rqr >= 4) else 0)
        del d, arr
Xc = np.stack(Xc); Xh = np.stack(Xh) if Xh else np.zeros((0, 4096))
print(f"compliance rows={len(yL)} (Lpos={sum(yL)} Qpos={sum(yQ)})  harm rows={len(hL)} (Lpos={sum(hL)} Qpos={sum(hQ)})\n")

print("=== PRE-RESPONSE decode AUC of the STEERED vectors (L16) ===")
sc = Xc @ refusal
for nm, y in [("Llama breach", yL), ("Qwen breach", yQ)]:
    pt, m, sd, npos, n = cv_auc(y, sc, gc)
    print(f"  refusal_dm  vs compliance/{nm:12s}: pointAUC={pt:.3f}  cv={m:.3f}±{sd:.3f}  (n={n}, pos={npos})")
sh = Xh @ harmdir
for nm, y in [("Llama>=4", hL), ("Qwen>=4", hQ)]:
    pt, m, sd, npos, n = cv_auc(y, sh, gh)
    print(f"  harm_dm_llama vs harm/{nm:12s}: pointAUC={pt:.3f}  cv={m:.3f}±{sd:.3f}  (n={n}, pos={npos})")
print("\n(For context, Fig-5 RE-FIT pre-response probes: compliance ~0.75, harm ~0.70-0.78.)")
