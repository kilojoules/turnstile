"""fig6 LEFT panel, re-made on goals[50:] (50 held-out JBB harmful prompts) with the
70B JailbreakBench judge (= attack success), replacing the old 40-prompt / lexical-
refusal-rate version.

Refusal-direction data on goals[50:]:  refusal_harm_vs_compliance_v1 (dir="refusal", 40)
  + steer_cvh_ext10 (dir="refusal_dm", 10).   Random control: steer_cvh_matched + fill
  + ext10 (dir="random_1", 50).   Metric: 100 * mean(judge_compliance_unsafe), Llama-3.1-70B JBB.
"""
import json, numpy as np, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/Users/julianquick/portfolio_copy/turnstile"; FIG = f"{R}/figures/causal_steering"
TEAL, GREY = "#1b9e77", "#8a8a8a"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#4d4d4d","figure.facecolor":"white"})
def load(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

REF = load(f"{R}/experiments/refusal_harm_vs_compliance_v1/judged.jsonl") + load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl")
RND = (load(f"{R}/experiments/steer_cvh_matched/judged_llama.jsonl")
       + load(f"{R}/experiments/steer_cvh_fill/judged_llama.jsonl")
       + load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl"))
def curve(rows, dirs):
    by=defaultdict(list)
    for r in rows:
        if r["direction"]=="baseline": by[0.0].append(r)
        elif r["direction"] in dirs: by[r["alpha"]].append(r)
    xs=sorted(by); return xs, [100*np.mean([1 if x.get("judge_compliance_unsafe") else 0 for x in by[a]]) for a in xs]

xr, cr = curve(REF, {"refusal","refusal_dm"})
xn, cn = curve(RND, {"random_1"})
NP = len({r["goal"] for r in REF if r["direction"] in ("refusal","refusal_dm","baseline") and r["alpha"]==0})

fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.plot(xr, cr, "-o", color=TEAL, lw=2.8, ms=7, label="refusal direction")
ax.plot(xn, cn, "--s", color=GREY, lw=2.2, ms=6, label="random direction (equal magnitude)")
ax.axvline(0, color="k", lw=0.7, ls=":"); ax.axhline(2, color="#bbbbbb", lw=0.8, ls="--")
ax.text(0.12, 6, "no steering (α=0):\n2% attack success", fontsize=8.5, color="#666")
ax.annotate("subtract refusal\n⇒ jailbreak", xy=(-1.2, 58), xytext=(-0.55, 72), fontsize=9, color=TEAL, ha="left",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
ax.set_xlabel("steering strength  α    (− = subtract the refusal direction)")
ax.set_ylabel("attack success (%)  —  70B JBB judge (response is unsafe)")
ax.set_ylim(-4, 100); ax.set_xlim(-1.62, 1.62); ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.set_title(f"Subtracting the refusal direction jailbreaks the victim\n"
             f"{NP} held-out JailbreakBench harmful prompts (goals[50:]) · Llama-3.1-70B JBB judge",
             fontsize=11, loc="left")
fig.text(0.01, -0.02,
    "Same 50 single-turn prompts and push size for both arms; only the direction changes. Metric = 70B JBB "
    "attack-success judge (not the lexical refusal-phrase rate). Subtracting the refusal direction (α<0) drives "
    "attack success 2%→60%; a random direction of equal magnitude reaches only ~28% at the largest pushes.",
    fontsize=7.6, color="#555", wrap=True)
fig.tight_layout()
for e in ("png","pdf"):
    fig.savefig(f"{FIG}/fig_refusal_monotonic_70b_goals50.{e}", dpi=200 if e=="png" else None, bbox_inches="tight")
    print(f"wrote {FIG}/fig_refusal_monotonic_70b_goals50.{e}")
print(f"n_prompts={NP}  refusal peak={max(cr):.0f}%  random peak={max(cn):.0f}%")
