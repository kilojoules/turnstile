"""Harm-MD variant set figure: anchor vs sharpened 5-vs-1 vs residualized — all null."""
import json, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="figures/causal_steering"; U=7.563
r=defaultdict(list)
for l in open("experiments/steer_5v1/judged_llama.jsonl"):
    x=json.loads(l); r[(x["set"],x["direction"],x["magnitude"])].append(x)
mags=[0.0,2.8,5.5,11.1]; xa=[m/U for m in mags]
def h(s,d):
    ys=[]
    for m in mags:
        k=(s,"baseline",0.0) if m==0 else (s,d,m)
        v=[a["judge_harm_likert"] for a in r[k] if isinstance(a.get("judge_harm_likert"),(int,float))]
        ys.append(np.mean(v) if v else np.nan)
    return ys
series=[("harm_dm_llama","anchor (4,5)v(1,2) Llama","#8a8a8a","-o"),
        ("harm_dm_45v12_qwen","anchor (4,5)v(1,2) Qwen","#bbbbbb","-o"),
        ("harm_dm_5v1_qwen","SHARPENED 5-vs-1 (Qwen)","#d1495b","-s"),
        ("harm_dm_5v1_qwen_resid","5-vs-1 residualized (−refusal,−length)","#7b3fa0","-^"),
        ("random_1","random","#cccccc","--o")]
fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
for ax,s,ttl in [(axes[0],"low","low-uplift replies (base 2.75)"),(axes[1],"high","already-harmful replies (base 3.66)")]:
    ax.axhline(h(s,"harm_dm_llama")[0],color="#ddd",lw=1,ls="--")
    for d,lab,c,st in series:
        ax.plot(xa,h(s,d),st,color=c,lw=2.2,ms=6,label=lab)
    ax.set_ylim(1,5); ax.axhline(4,color="#d1495b",lw=0.6,ls=":"); ax.grid(alpha=0.25)
    ax.set_xlabel("steering strength α (α×7.56=‖added vec‖)"); ax.set_title(ttl,fontsize=10)
    if s=="low": ax.set_ylabel("harm uplift (Stage-B, 1–5)"); ax.legend(frameon=False,fontsize=7.6,loc="lower left")
fig.suptitle("Harm-MD variant set — the cleanest null: sharpening labels to the extremes (5-vs-1) and stripping the\n"
             "refusal+length confound both leave uplift flat = baseline = random. (Llama 5-vs-1 unbuildable: only n=2 fives.)",fontsize=10.5,y=1.02)
fig.text(0.5,-0.02,"How to read: at the calibrated dose (α≈0.37) every harm direction sits on baseline and on random. The most length-loaded direction "
    "(sharpened 5-vs-1, cos 0.34 to length) does not steer even RAW — so there is no verbosity artifact for the residualized twin to remove. "
    "At large push all harm arms fall as coherence collapses (degradation), not negative steering; random stays flat.",ha="center",fontsize=7.4,style="italic",wrap=True)
fig.tight_layout(); fig.savefig(f"{OUT}/figP5_harm_md_variants.png",dpi=140,bbox_inches="tight"); print("figP5")
