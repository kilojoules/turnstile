"""Phase 3 (confound kill) + Phase 4 (locus control) figures."""
import json, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="figures/causal_steering"; U=7.563
TEAL,CRIM,GREY,PURP="#1b9e77","#d1495b","#8a8a8a","#7b3fa0"

def load(p): return [json.loads(l) for l in open(p)]

# ---------------- Phase 3: residualized harm still inert ----------------
r3=defaultdict(list)
for r in load("experiments/phase3_resid_harm/judged_llama.jsonl"): r3[(r["set"],r["direction"],r["magnitude"])].append(r)
mags=[0.0,2.8,5.5,11.1]; xa=[m/U for m in mags]
def h3(s,d):
    ys=[]
    for m in mags:
        k=(s,"baseline",0.0) if m==0 else (s,d,m)
        v=[x["judge_harm_likert"] for x in r3[k] if isinstance(x.get("judge_harm_likert"),(int,float))]
        ys.append(np.mean(v) if v else np.nan)
    return ys
fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),sharey=True)
for ax,s,ttl in [(axes[0],"low","low-uplift replies (base 2.75)"),(axes[1],"high","already-harmful replies (base 3.66)")]:
    ax.axhline(h3(s,"harm_dm_resid")[0],color="#bbb",ls="--",lw=1)
    ax.plot(xa,h3(s,"harm_dm_resid"),"-o",color=CRIM,lw=2.5,ms=6,label="harm dir, refusal+length REMOVED (39%)")
    ax.plot(xa,h3(s,"harm_dm_llama"),"-s",color=PURP,lw=2,ms=5,label="harm dir, raw")
    ax.plot(xa,h3(s,"random_1"),"--o",color=GREY,lw=1.8,ms=5,label="random")
    ax.set_ylim(1,5); ax.axhline(4,color=CRIM,lw=0.6,ls=":"); ax.grid(alpha=0.25)
    ax.set_xlabel("steering strength α (α×7.56=‖added vec‖)"); ax.set_title(ttl,fontsize=10)
    if s=="low": ax.set_ylabel("harm uplift (Stage-B, 1–5)"); ax.legend(frameon=False,fontsize=8,loc="lower left")
fig.suptitle("Phase 3 — the CONFOUND is not why harm won't steer: stripping the 39% refusal+length component\nleaves the pure-harm residual exactly as inert as the raw harm direction (= random)",fontsize=11,y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/figP3_residualized_harm.png",dpi=140,bbox_inches="tight"); print("figP3")

# ---------------- Phase 4: only the INPUT gate steers ----------------
r4=defaultdict(list)
for r in load("experiments/phase4_locus/judged_llama.jsonl"): r4[(r["direction"],r["alpha"])].append(r)
rdm=defaultdict(list)
for r in load("experiments/refusal_harm_vs_compliance_v1/judged.jsonl"): rdm[r["alpha"]].append(r)
A=sorted({a for (d,a) in r4}); Adm=sorted(rdm)
def c4(d,a): k=("baseline",0.0) if a==0 else (d,a); g=r4[k]; return 100*sum(1 for x in g if x.get("judge_compliance_unsafe"))/len(g) if g else np.nan
def coh4(d,a): k=("baseline",0.0) if a==0 else (d,a); g=r4[k]; return np.mean([x["coherence"]["token_unique_ratio"] for x in g]) if g else np.nan
def cdm(a): g=rdm[a]; return 100*sum(1 for x in g if x.get("judge_compliance_unsafe"))/len(g)
fig,(a1,a2)=plt.subplots(1,2,figsize=(12.5,4.7))
a1.plot(Adm,[cdm(a) for a in Adm],"-o",color="k",lw=3,ms=6,label="refusal dir — INPUT gate (steers)")
a1.plot(A,[c4("comp_dm_out",a) for a in A],"-s",color=TEAL,lw=2.2,ms=6,label="compliance dir — OUTPUT locus")
a1.plot(A,[c4("comp_dm_out_resid",a) for a in A],"-^",color=PURP,lw=2.2,ms=6,label="compliance OUTPUT, length-free")
a1.plot(A,[c4("random_1",a) for a in A],"--o",color=GREY,lw=1.8,ms=5,label="random")
a1.set_xlabel("steering strength α (−α = subtract)"); a1.set_ylabel("compliance / attack success (%)")
a1.set_ylim(-4,70); a1.grid(alpha=0.25); a1.axvline(0,color="k",lw=0.5,ls=":"); a1.legend(frameon=False,fontsize=8,loc="upper center")
a1.set_title("Only the INPUT-locus gate steers compliance;\noutput-locus compliance ≈ random",fontsize=10)
for d,c,m in [("comp_dm_out",TEAL,"s"),("comp_dm_out_resid",PURP,"^"),("random_1",GREY,"o")]:
    a2.plot(A,[coh4(d,a) for a in A],"-"+m,color=c,lw=2,ms=5,label=d)
a2.axhspan(0,0.6,color="#f2d0d0",alpha=0.5); a2.text(-1.45,0.30,"degraded",fontsize=8,color=CRIM)
a2.set_xlabel("steering strength α"); a2.set_ylabel("coherence"); a2.set_ylim(0,1.0); a2.grid(alpha=0.25); a2.legend(frameon=False,fontsize=8,loc="lower left")
a2.set_title("output-locus 'compliance' only at coherence collapse\n(degradation false-positive)",fontsize=10)
fig.suptitle("Phase 4 — steerability is INPUT-GATE vs OUTPUT-CONTENT, not harm vs compliance:\noutput-locus compliance is as unsteerable as output-locus harm; the refusal input-gate is the only real lever",fontsize=10.5,y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/figP4_locus.png",dpi=140,bbox_inches="tight"); print("figP4")
