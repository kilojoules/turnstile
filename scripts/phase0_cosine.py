"""Phase 0 — cosine table + probe AUCs (no GPU). Gates the SIR decision.

Loads directions/ + directions/reps.npz, prints the six claim-bearing cosines,
computes GroupKFold probe AUCs (harm is maximally readable), and saves the
cosine heatmap. Prints the SIR gate verdict (harm_probe·harm_dm ≥/< 0.4).
"""
import json, glob, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

D = "directions"
dirs = {os.path.basename(f)[:-3]: torch.load(f).float() for f in glob.glob(f"{D}/*.pt")}
def cos(a, b): return float(torch.dot(dirs[a], dirs[b]) / (dirs[a].norm()*dirs[b].norm()))

order = ["refusal_dm", "comp_probe", "comp_dm_out", "comp_probe_out",
         "harm_dm_llama", "harm_dm_qwen", "harm_probe_llama", "harm_probe_qwen",
         "length_dm", "random_1", "random_2", "random_3"]
order = [d for d in order if d in dirs]

print("\n===== THE SIX NUMBERS =====")
def show(label, a, b): print(f"  {label:<46} {cos(a,b):+.3f}")
show("1. comp_probe · refusal_dm (compliance clean?)", "comp_probe", "refusal_dm")
g_l = cos("harm_probe_llama", "harm_dm_llama"); g_q = cos("harm_probe_qwen", "harm_dm_qwen")
print(f"  2. harm_probe·harm_dm  Llama={g_l:+.3f}  Qwen={g_q:+.3f}   <-- SIR GATE")
show("3. harm_dm_llama · harm_dm_qwen (expect ~0.93)", "harm_dm_llama", "harm_dm_qwen")
show("4. harm_probe_llama · harm_probe_qwen", "harm_probe_llama", "harm_probe_qwen")
show("5a. harm_dm_llama · refusal_dm (leak)", "harm_dm_llama", "refusal_dm")
show("5b. harm_dm_llama · length_dm (leak)", "harm_dm_llama", "length_dm")
rfloor = [abs(cos(d, f"random_{k}")) for d in order if not d.startswith("random") for k in range(1, 4)]
print(f"  6. |·random_k| floor: max={max(rfloor):.3f} mean={np.mean(rfloor):.3f}  (chance ~0.016)")

gate = max(g_l, g_q)
print(f"\n===== SIR GATE: harm_probe·harm_dm max = {gate:+.3f} =====")
print("  >= 0.40  -> harm ~1-D, SIR is APPENDIX/pre-emption" if gate >= 0.40
      else "  < 0.40  -> whitened & raw harm axes diverge, SIR is a CENTERPIECE (schedule w/ Phase 2)")

# ---- probe AUCs (GroupKFold): harm is maximally readable
if os.path.exists(f"{D}/reps.npz"):
    z = np.load(f"{D}/reps.npz")
    def cv_auc(X, y, groups):
        m = (y >= 4) | (y <= 2) if y.max() > 1 else np.ones(len(y), bool)
        yb = (y[m] >= 4).astype(int) if y.max() > 1 else y[m]
        Xm, gm = X[m], groups[m]; aucs = []
        for tr, te in GroupKFold(5).split(Xm, yb, gm):
            if len(set(yb[tr])) < 2 or len(set(yb[te])) < 2: continue
            c = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xm[tr], yb[tr])
            aucs.append(roc_auc_score(yb[te], c.decision_function(Xm[te])))
        return np.mean(aucs), np.std(aucs)
    print("\n===== PROBE AUC (GroupKFold, out-of-fold) — LABEL THE LOCUS =====")
    print("  WARNING: harm reps here are OUTPUT locus (mean-pooled response) = separability,")
    print("  NOT pre-response readability. Pre-response harm (Fig 5, input locus) is mid-0.7s.")
    for nm, X, y, gg in [("harm  · OUTPUT locus · Llama", z["Xh"], z["rl"], z["gh"]),
                          ("harm  · OUTPUT locus · Qwen", z["Xh"], z["rq"], z["gh"]),
                          ("compliance · INPUT locus (Fig-5)", z["Xci"], z["yc"], z["gc"]),
                          ("compliance · OUTPUT locus", z["Xco"], z["yc"], z["gc"])]:
        a, s = cv_auc(X, y, gg); print(f"  {nm:<34} AUC = {a:.3f} ± {s:.3f}")

# ---- heatmap
M = np.array([[cos(a, b) for b in order] for a in order])
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(order, fontsize=8)
for i in range(len(order)):
    for j in range(len(order)):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                fontsize=6.5, color="white" if abs(M[i,j]) > 0.6 else "black")
fig.colorbar(im, fraction=0.046, pad=0.04, label="cosine similarity")
ax.set_title("Direction inventory — pairwise cosine (L16, hidden_states[17])", fontsize=11)
fig.tight_layout()
os.makedirs("figures/causal_steering", exist_ok=True)
fig.savefig("figures/causal_steering/figX_cosine_matrix.png", dpi=140, bbox_inches="tight")
print("\nsaved figures/causal_steering/figX_cosine_matrix.png")
