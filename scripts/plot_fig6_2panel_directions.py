"""fig6_refusal_monotonic_2panel, re-done to steer the THREE compliance directions
(+ random control) in BOTH panels, lexical is_refusal metric.

LEFT  = HARMFUL (JBB goals[50:], n=50): y = compliance% = 100 - is_refusal%  (non-refusal = jailbreak).
RIGHT = BENIGN (Alpaca harmless[50:90], n=40): y = refusal% = is_refusal%     (over-refusal).
x oriented so +x = TOWARD COMPLIANCE for each direction (refusal_dm plotted at x=-alpha).

Directions (markers): comp_dm_out (o, posterior/training), comp_pre_qwen (s) & comp_pre_llama (^)
(prior/training), refusal_dm (v, prior/task-only), random_1 (X, control).

Data:
  harmful: steer_decoded(+ext10) [comp_*], refusal_harm_vs_compliance_v1(+cvh_ext10) [refusal_dm],
           steer_cvh_matched(+fill+ext10) [random_1].  is_refusal from judged_llama.
  benign : refusal_alpha_sweep_v1 [refusal,random]  +  steer_benign_comp [comp_*]  (both lexical).
"""
import json, os, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/Users/julianquick/portfolio_copy/turnstile"; FIG = f"{R}/figures/causal_steering"
TEAL, GREY = "#1b9e77", "#8a8a8a"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#4d4d4d","figure.facecolor":"white"})
def load(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

# ---------- harmful sources ----------
REF_H = load(f"{R}/experiments/refusal_harm_vs_compliance_v1/judged.jsonl") + load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl")
DEC_H = load(f"{R}/experiments/steer_decoded/judged_llama.jsonl") + load(f"{R}/experiments/steer_decoded_ext10/judged_llama.jsonl")
RND_H = (load(f"{R}/experiments/steer_cvh_matched/judged_llama.jsonl") + load(f"{R}/experiments/steer_cvh_fill/judged_llama.jsonl")
         + load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl"))
# ---------- benign sources ----------
BEN   = [r for r in load(f"{R}/experiments/refusal_alpha_sweep_v1/sweep.jsonl") if r.get("set")=="benign"]
BEN_C = load(f"{R}/experiments/steer_benign_comp/sweep.jsonl")

def curve(rows, dirset, metric, flip=False):
    by = defaultdict(list)
    for r in rows:
        d = r.get("direction")
        if d == "baseline": by[0.0].append(r)
        elif d in dirset: by[r["alpha"]].append(r)
    pts = {}
    for a, rs in by.items():
        ref = np.mean([1 if x.get("is_refusal") else 0 for x in rs])
        pts[(-a if flip else a)] = (100*ref if metric == "refuse" else 100*(1-ref))
    xs = sorted(pts); return xs, [pts[x] for x in xs], len({r.get("prompt_id") for r in sum(by.values(), [])})

# marker, color, {harmful dirset, source} , {benign dirset, source}, flip, label
SPECS = [
    ("o", TEAL, DEC_H, {"comp_dm_out"},   BEN_C, {"comp_dm_out"},   False, "comp_dm_out  (posterior · training corpus)"),
    ("s", TEAL, DEC_H, {"comp_pre_qwen"}, BEN_C, {"comp_pre_qwen"}, False, "comp_pre_qwen (prior · training · Qwen)"),
    ("^", TEAL, DEC_H, {"comp_pre_llama"},BEN_C, {"comp_pre_llama"},False, "comp_pre_llama (prior · training · Llama)"),
    ("v", TEAL, REF_H, {"refusal","refusal_dm"}, BEN, {"refusal"},  True,  "refusal_dm  (prior · task-only)  [x=−α]"),
    ("X", GREY, RND_H, {"random_1"},      BEN, {"random"},          False, "random  (control)"),
]

fig, (aH, aB) = plt.subplots(1, 2, figsize=(14.6, 5.9), sharex=True)
for mk, col, hrows, hset, brows, bset, flip, lab in SPECS:
    xh, yh, nh = curve(hrows, hset, "comply", flip)
    aH.plot(xh, yh, "-" if col==TEAL else "--", marker=mk, color=col, lw=2.3 if col==TEAL else 1.9,
            ms=7 if col==TEAL else 6, mec="white", mew=0.6, zorder=3 if col==TEAL else 6)
    xb, yb, nb = curve(brows, bset, "refuse", flip)
    if xb:
        aB.plot(xb, yb, "-" if col==TEAL else "--", marker=mk, color=col, lw=2.3 if col==TEAL else 1.9,
                ms=7 if col==TEAL else 6, mec="white", mew=0.6, zorder=3 if col==TEAL else 6)

for ax in (aH, aB):
    ax.axvline(0, color="k", lw=0.7, ls=":"); ax.set_xlim(-1.62, 1.62); ax.grid(alpha=0.22); ax.set_ylim(-4, 104)
    ax.set_xlabel("steering strength  α        (oriented so  +α → toward compliance)")
aH.set_ylabel("compliance (%)  =  100 − lexical refusal rate")
aB.set_ylabel("refusal (%)  =  lexical refusal rate")
aH.axhline(2, color="#bbb", lw=0.8, ls="--")
aH.set_title("HARMFUL — JailbreakBench goals[50:] (n=50)\ntoward compliance ⇒ jailbreak", fontsize=11, loc="left")
aB.set_title("BENIGN — Alpaca harmless[50:90] (n=40)\naway from compliance ⇒ over-refusal", fontsize=11, loc="left")
mk_leg = [Line2D([0],[0],color=col,marker=mk,ls="-" if col==TEAL else "--",lw=1.9,ms=7.5,mec="white") for mk,col,*_ in SPECS]
aH.legend(mk_leg, [lab for *_,lab in SPECS], loc="upper left", fontsize=8.4, frameon=False,
          title="steering direction", title_fontsize=8.8)
fig.suptitle("Steering the three compliance directions (+ random) — both panels, lexical refusal metric",
             fontsize=12.5, fontweight="semibold", y=1.02)
fig.text(0.01,-0.02,"Same L16 additive hook (α·REF_NORM·unit) and prompts for every arm; only the direction changes. "
    "Harmful: toward-compliance drives non-refusal (jailbreak). Benign: pushing AWAY from compliance induces refusal of "
    "benign requests. Random (equal magnitude) does little in either until it degrades the model at large ‖α‖.",
    fontsize=7.5, color="#555", wrap=True)
fig.tight_layout(w_pad=2.5)
for stem in ("fig6_refusal_monotonic_2panel", "fig6_refusal_monotonic_2panel_directions"):
    for e in ("png", "pdf"):
        fig.savefig(f"{FIG}/{stem}.{e}", dpi=200 if e == "png" else None, bbox_inches="tight")
        print(f"wrote {FIG}/{stem}.{e}")
print(f"benign comp rows loaded: {len(BEN_C)}  (0 => benign GPU run not landed yet)")
