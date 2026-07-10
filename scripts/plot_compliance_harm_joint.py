"""Compliance x Harm joint distribution (the measurement-thesis money figure).

2x5 contingency from ONE joined file (stage_b_scores_llama.jsonl: unsafe=JBB compliance,
rating=Llama Stage-B, qwen_rating=Qwen Stage-B). Column-normalized (each compliance column
sums to 100%), cells annotated with raw n, color scale floored (PowerNorm) so the small
refused-but-harmful cells stay visible. Two panels: harm=Llama, harm=Qwen. Same JBB
compliance axis. Also re-renders Fig 6 with a matching (Llama) y-axis for side-by-side.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.stats import spearmanr

ROOT="/Users/julianquick/portfolio_copy/turnstile"; FIG=f"{ROOT}/figures"
J=[json.loads(l) for l in open(f"{ROOT}/working/uplift/stage_b_scores_llama.jsonl")]
print(f"joined rows = {len(J)}")
refused=[r for r in J if not r.get("unsafe")]; complied=[r for r in J if r.get("unsafe")]
print(f"refused (unsafe=False) = {len(refused)}   complied (unsafe=True) = {len(complied)}   (scatter: losses=311, wins=289)")

def contingency(key):
    M=np.zeros((5,2))  # rows harm 1..5, cols [refused, complied]
    for j,grp in enumerate([refused,complied]):
        for r in grp:
            v=r.get(key)
            if isinstance(v,int) and 1<=v<=5: M[v-1,j]+=1
    return M
Mll=contingency("rating"); Mqw=contingency("qwen_rating")

def print_table(name,M):
    print(f"\n=== {name}  per-cell n (col% in parens) ===")
    print(f"{'harm':>6} | {'refused':>16} | {'complied':>16}")
    colsum=M.sum(0)
    for h in range(5,0,-1):
        n_r=int(M[h-1,0]); n_c=int(M[h-1,1])
        print(f"{h:>6} | {n_r:>6} ({100*n_r/colsum[0]:>5.1f}%) | {n_c:>6} ({100*n_c/colsum[1]:>5.1f}%)")
    print(f"{'sum':>6} | {int(colsum[0]):>16} | {int(colsum[1]):>16}")
print_table("LLAMA Stage-B", Mll); print_table("QWEN Stage-B", Mqw)

def heat(ax, M, judge):
    col=M.sum(0); P=100*M/col  # column-normalized %
    im=ax.imshow(P, cmap="Reds", origin="lower", aspect="auto",
                 norm=PowerNorm(gamma=0.45, vmin=0, vmax=P.max()))
    ax.set_xticks([0,1]); ax.set_xticklabels([f"refused\n(n={int(col[0])})", f"complied\n(n={int(col[1])})"])
    ax.set_yticks(range(5)); ax.set_yticklabels([1,2,3,4,5])
    ax.set_ylabel(f"harm — {judge} Stage-B (1–5)"); ax.set_xlabel("JBB compliance verdict")
    for h in range(5):
        for c in range(2):
            n=int(M[h,c]); p=P[h,c]
            ax.text(c,h,f"{p:.0f}%\nn={n}", ha="center", va="center", fontsize=8.5,
                    color="white" if p>P.max()*0.55 else "black")
    ax.set_title(f"harm judge = {judge}", fontsize=10)
    return im

# --- Panel figure: Llama + Qwen ---
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.5,5.2))
heat(a1,Mll,"Llama"); heat(a2,Mqw,"Qwen")
fig.suptitle("Compliance × harm: column-normalized (each compliance column = 100%).\n"
             "Compliant responses appear at every harm level; ~44% clear harm≥4, ~half sit ≤2. "
             "Mirror cells: refused-but-harmful (top-left).", fontsize=10.5, y=1.03)
fig.tight_layout(); fig.savefig(f"{FIG}/causal_steering/figJ_compliance_harm_joint.png", dpi=150, bbox_inches="tight")
print("\nwrote figJ_compliance_harm_joint.png")

# --- Fig 6 re-render with LLAMA y-axis (matched judge) ---
stage_a={}
for line in open(f"{ROOT}/working/uplift/stage_a_scores.jsonl"):
    r=json.loads(line)
    if r.get("parse_ok"): stage_a[r["behavior"]]=r["rating"]
recs=[{"unsafe":r["unsafe"],"stage_a":stage_a[r["behavior"]],"stage_b":r["rating"]}
      for r in J if r.get("parse_ok") and r["behavior"] in stage_a]
rng=np.random.default_rng(0)
def scatter(ax):
    for grp,color,lab in [([r for r in recs if not r["unsafe"]],"#7f7f7f","losses"),
                          ([r for r in recs if r["unsafe"]],"#d62728","wins")]:
        gx=np.array([r["stage_a"] for r in grp])+rng.uniform(-.18,.18,len(grp))
        gy=np.array([r["stage_b"] for r in grp])+rng.uniform(-.18,.18,len(grp))
        ax.scatter(gx,gy,s=20,color=color,edgecolor="black",linewidth=0.3,alpha=0.55,label=f"{lab} (n={len(grp)})")
    ax.set_xticks(range(1,6)); ax.set_yticks(range(1,6)); ax.set_xlim(.5,5.5); ax.set_ylim(.5,5.5)
    ax.set_xlabel("Stage A harm prior (per-behavior)"); ax.set_ylabel("Stage B harm — Llama (per-response)")
    ax.set_title("Fig 6: prior vs posterior harm (Llama y-axis)", fontsize=10); ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=8)

# --- Side-by-side: Fig6 (Llama) | new 2x5 (Llama) ---
fig2,(s1,s2)=plt.subplots(1,2,figsize=(12.5,5.2))
scatter(s1); heat(s2,Mll,"Llama")
fig2.suptitle("Side-by-side (matched judge = Llama):  Fig 6 prior≠posterior harm   vs   new compliance×harm joint", fontsize=11, y=1.02)
fig2.tight_layout(); fig2.savefig(f"{FIG}/causal_steering/fig_sidebyside_fig6_vs_joint.png", dpi=150, bbox_inches="tight")
print("wrote fig_sidebyside_fig6_vs_joint.png")
