"""fig6 LEFT panel, re-made on goals[50:] (50 held-out JBB harmful prompts) showing
ALL compliance directions we can steer + the random control, on the LEXICAL metric
(compliance = 100 - is_refusal%). The lexical rate is recorded uniformly for every
run, so it is the common denominator for comparing directions derived different ways.

Orientation: x is signed so +x = "toward compliance" for each direction.
  comp_dm_out / comp_pre_qwen / comp_pre_llama  are fit so +alpha = compliance  -> x = +alpha
  refusal_dm (harmful-vs-benign prior)          opens the gate by SUBTRACTION   -> x = -alpha
  random_1                                       has no compliance side (control) -> x = +alpha (raw)

Caveat drawn on the figure: at large push AWAY from compliance (and for random at
large |alpha|), the victim degrades into incoherent text that also isn't a refusal
*phrase* -> inflated lexical "compliance". The 70B JBB judge scores those cells ~0
(random peaks at 28% attack success there, not 84%). So read the TOWARD-compliance
(right) half as the real signal.

Data (all goals[50:], n=50/cell, Llama generation, is_refusal lexical filter):
  refusal_dm : refusal_harm_vs_compliance_v1 (dir="refusal",40) + steer_cvh_ext10 (dir="refusal_dm",10)
  comp_*     : steer_decoded + steer_decoded_ext10
  random_1   : steer_cvh_matched + steer_cvh_fill + steer_cvh_ext10
"""
import json, numpy as np, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/Users/julianquick/portfolio_copy/turnstile"; FIG = f"{R}/figures/causal_steering"
TEAL, GREY = "#1b9e77", "#8a8a8a"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#4d4d4d","figure.facecolor":"white"})
def load(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

REF = load(f"{R}/experiments/refusal_harm_vs_compliance_v1/judged.jsonl") + load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl")
DEC = load(f"{R}/experiments/steer_decoded/judged_llama.jsonl") + load(f"{R}/experiments/steer_decoded_ext10/judged_llama.jsonl")
RND = (load(f"{R}/experiments/steer_cvh_matched/judged_llama.jsonl")
       + load(f"{R}/experiments/steer_cvh_fill/judged_llama.jsonl")
       + load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl"))

# incoherence rule, calibrated by the coherence-marker workflow (precision 0.97 / recall 0.85)
# against 132 human-style-labeled responses: mr=max consecutive identical-token run,
# tur=token_unique_ratio, ntok=n_tokens. Flags token loops (mr) AND long phrase/line loops (tur+ntok).
def is_incoherent(r):
    c = r.get("coherence", {}) or {}
    mr = c.get("max_repeat", 0); tur = c.get("token_unique_ratio", 1.0); ntok = c.get("n_tokens", 0)
    return (mr >= 7) or (tur < 0.28 and ntok >= 120)

def curve(rows, dirset, flip=False):
    """return (xs sorted, comp%, incoh_frac), x oriented so +x = toward compliance."""
    by = defaultdict(list)
    for r in rows:
        if r["direction"] == "baseline": by[0.0].append(r)
        elif r["direction"] in dirset: by[r["alpha"]].append(r)
    comp = {(-a if flip else a): 100*(1-np.mean([1 if x.get("is_refusal") else 0 for x in by[a]])) for a in by}
    incoh = {(-a if flip else a): np.mean([1 if is_incoherent(x) else 0 for x in by[a]]) for a in by}
    xs = sorted(comp); return xs, [comp[x] for x in xs], [incoh[x] for x in xs]

# marker = how the direction was computed; teal = compliance direction, grey = control
SPECS = [("comp_dm_out",   DEC, {"comp_dm_out"},            "o", TEAL, False, "complied − refused · OUTPUT locus (replay)"),
         ("comp_pre_qwen", DEC, {"comp_pre_qwen"},          "s", TEAL, False, "complied − refused · pre-response · Qwen labels"),
         ("comp_pre_llama",DEC, {"comp_pre_llama"},         "^", TEAL, False, "complied − refused · pre-response · Llama labels"),
         ("refusal_dm",    REF, {"refusal","refusal_dm"},   "v", TEAL, True,  "harmful − benign PRIOR · pre-response  (x = −α)"),
         ("random_1",      RND, {"random_1"},               "X", GREY, False, "random direction (equal magnitude) · control")]

fig, ax = plt.subplots(figsize=(8.6, 5.8))
NP = len({r["goal"] for r in REF if r["direction"] in ("refusal","refusal_dm","baseline") and r["alpha"]==0})
INCOH_MARKED = False
for name, rows, ds, mk, col, flip, lab in SPECS:
    xs, ys, inc = curve(rows, ds, flip)
    ax.plot(xs, ys, "-" if col==TEAL else "--", marker=mk, color=col,
            lw=2.4 if col==TEAL else 1.9, ms=7 if col==TEAL else 6,
            mec="white", mew=0.6, alpha=1.0 if col==TEAL else 0.9, zorder=3 if col==TEAL else 2)
    # overlay a hollow ring on points where the MAJORITY of responses are incoherent (gibberish)
    bad = [(x, y) for x, y, f in zip(xs, ys, inc) if f >= 0.5]
    if bad:
        INCOH_MARKED = True
        ax.scatter([x for x, _ in bad], [y for _, y in bad], s=210, facecolors="none",
                   edgecolors="#c0392b", linewidths=1.8, zorder=5)

ax.axvline(0, color="k", lw=0.7, ls=":"); ax.axhline(2, color="#bbbbbb", lw=0.8, ls="--")
ax.text(0.06, 5.5, "no steering: 2%", fontsize=8.5, color="#777")
# incoherence-artifact callout on the away-from-compliance side
ax.axvspan(-1.62, -0.85, color="#f0a0a0", alpha=0.12, zorder=0)
ax.annotate("push AWAY from compliance at large ‖α‖\n→ incoherent text that also isn't a refusal\nphrase (lexical artifact; 70B judge ≈ 0)",
            xy=(-1.35, 70), xytext=(-1.58, 40), fontsize=7.6, color="#9a4b4b", ha="left", va="top")
ax.annotate("all compliance directions\nopen the gate → ~100%", xy=(1.15, 99), xytext=(0.30, 82),
            fontsize=9, color=TEAL, ha="left", arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.1))

ax.set_xlabel("steering strength  α        (oriented so  +α → toward compliance)")
ax.set_ylabel("compliance (%)  =  100 − lexical refusal-phrase rate")
ax.set_ylim(-4, 104); ax.set_xlim(-1.62, 1.62); ax.grid(alpha=0.22)
ax.set_title("Every way of deriving the compliance direction opens the refusal gate; a random direction doesn't\n"
             f"{NP} held-out JailbreakBench harmful prompts (goals[50:]) · lexical non-refusal metric",
             fontsize=10.8, loc="left")
mk_leg = [Line2D([0],[0],color=col,marker=mk,ls="-" if col==TEAL else "--",lw=1.9,ms=7.5,mec="white") for _,_,_,mk,col,_,_ in SPECS]
leg1 = ax.legend(mk_leg, [lab for *_,lab in SPECS], loc="upper left", fontsize=8.3, frameon=False,
                 title="marker = how the direction was computed", title_fontsize=8.6)
ax.add_artist(leg1)
if INCOH_MARKED:
    ring = Line2D([0],[0], marker="o", ls="none", markerfacecolor="none", markeredgecolor="#c0392b",
                  markeredgewidth=1.8, markersize=13)
    ax.legend([ring], ["majority of responses incoherent (gibberish):\nmax_repeat≥7 or (unique-ratio<0.28 & ≥120 tok)"],
              loc="lower right", fontsize=8.0, frameon=False)
fig.text(0.01, -0.05,
    "Same 50 single-turn harmful prompts and matched push size for every arm; only the direction changes. Metric = lexical "
    "non-refusal (uniformly recorded for all runs), NOT the 70B judge. On the toward-compliance (right) side, all four "
    "compliance directions climb to ~90–100% while random stays ≤24%. Red rings flag α-points where the majority of "
    "responses are degenerate/incoherent (rule calibrated against 132 hand-labeled responses, precision 0.97 / recall 0.85): "
    "these high lexical values are gibberish evading the refusal-phrase filter — the 70B JBB judge scores them near 0 "
    "(e.g. random α=−1.5 is 84% lexical but only 28% attack-success, and is mostly coherent non-compliance, not gibberish).",
    fontsize=7.4, color="#555", wrap=True)
fig.tight_layout()
for e in ("png","pdf"):
    fig.savefig(f"{FIG}/fig_compliance_directions_lexical.{e}", dpi=200 if e=="png" else None, bbox_inches="tight")
    print(f"wrote {FIG}/fig_compliance_directions_lexical.{e}")
print(f"n_prompts={NP}")
