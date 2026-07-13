"""Decoded == steered, on goals[50:]. MARKER = how the steering direction was computed
(recipe/locus/label); COLOR = metric (teal compliance %, crimson harm). THREE panels,
one per direction *family* (they are not the same kind of object):

  L  COMPLIANCE directions (complied−refused, from the training corpus) — they STEER compliance:
        o comp_dm_out (OUTPUT/posterior) · s comp_pre_qwen · ^ comp_pre_llama (pre-response)
  M  PRIOR, TASK-ONLY direction — refusal_dm = μ(harmful)−μ(benign) *prompts*, a different
        contrast (the refusal switch), so it gets its own panel. Also steers compliance.
  R  HARM directions (μ(≥4)−μ(≤2) etc.) — none STEER harm:
        o harm_dm_llama · s harm_dm_resid · D harm_pre_llama · X random control
Both metrics on twin axes (teal compliance% left, crimson Stage-B harm 1–5 right). Llama judge.
Auto-merges steer_decoded + steer_decoded_ext10 (n=50).
"""
import json, numpy as np, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/Users/julianquick/portfolio_copy/turnstile"; FIG = f"{R}/figures/causal_steering"
TEAL, CRIM = "#1b9e77", "#c0455a"
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 13, "axes.labelsize": 11.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4d4d4d", "axes.linewidth": 1.0, "figure.facecolor": "white"})

def load(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
HL = load(f"{R}/experiments/steer_decoded/judged_llama.jsonl") + load(f"{R}/experiments/steer_decoded_ext10/judged_llama.jsonl")
NP = len({r["goal"] for r in HL if r["direction"]=="baseline"}) or "?"

def curve(rows, direction, field):
    by = defaultdict(list)
    for r in rows:
        if r["direction"] == "baseline": by[0.0].append(r)
        elif r["direction"] == direction: by[r["alpha"]].append(r)
    xs = sorted(by)
    if field == "comp":
        ys = [100*np.mean([1 if x.get("judge_compliance_unsafe") else 0 for x in by[a]]) for a in xs]
    else:
        ys = [np.mean([x["judge_harm_likert"] for x in by[a] if isinstance(x.get("judge_harm_likert"),(int,float))]) for a in xs]
    return xs, ys

# refusal_dm PRIOR: refusal_v1 (dir="refusal", 40) + ext10 (dir="refusal_dm", 10). x=-alpha so
# "subtract refusal = toward compliance" points +x like the posterior directions.
REFL = load(f"{R}/experiments/refusal_harm_vs_compliance_v1/judged.jsonl") + load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl")
def refusal_curve(field):
    by = defaultdict(list)
    for r in REFL:
        d = r.get("direction")
        if d in ("refusal", "refusal_dm"): by[r["alpha"]].append(r)
        elif d == "baseline": by[0.0].append(r)
    items = sorted(by)
    if field == "comp":
        ys = {a: 100*np.mean([1 if x.get("judge_compliance_unsafe") else 0 for x in by[a]]) for a in items}
    else:
        ys = {a: np.mean([x["judge_harm_likert"] for x in by[a] if isinstance(x.get("judge_harm_likert"),(int,float))]) for a in items}
    xf = sorted(-a for a in items)
    return xf, [ys[-x] for x in xf]

CVHl = (load(f"{R}/experiments/steer_cvh_matched/judged_llama.jsonl")
        + load(f"{R}/experiments/steer_cvh_fill/judged_llama.jsonl")
        + load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl"))

fig, (aL, aM, aR) = plt.subplots(1, 3, figsize=(20.5, 6.1))

def frame(ax, title, sub, xlab):
    ax.set_xlim(-1.62, 1.62); ax.axvline(0, color="#d8d8d8", lw=1, ls=":"); ax.set_ylim(-3, 100)
    t = ax.twinx(); t.spines["top"].set_visible(False); t.set_ylim(1, 5)
    ax.tick_params(axis="y", labelcolor=TEAL); t.tick_params(axis="y", labelcolor=CRIM)
    ax.set_xlabel(xlab)
    ax.text(0.0, 1.10, title, transform=ax.transAxes, fontsize=12.5, fontweight="semibold", ha="left")
    ax.text(0.0, 1.037, sub, transform=ax.transAxes, fontsize=8.3, color="#6b6b6b", ha="left")
    return t

# ---------------- L: prior, task-only direction (refusal_dm) ----------------
aLt = frame(aL, "PRIOR, TASK-ONLY direction  →  it STEERS",
            "refusal_dm = μ(harmful) − μ(benign) PROMPTS · pre-response · a different contrast", "α")
xr, cr = refusal_curve("comp"); xrh, hr = refusal_curve("harm")
aL.plot(xr, cr, "-", marker="v", color=TEAL, lw=2.4, ms=7, mec="white", mew=0.6)
aLt.plot(xrh, hr, "-", marker="v", color=CRIM, lw=1.9, ms=6, mec="white", mew=0.5, alpha=0.9)
# random control (equal magnitude, same 70B judges), plotted at raw alpha
xrc, rc = curve(CVHl, "random_1", "comp"); xrhh, rhh = curve(CVHl, "random_1", "harm")
aL.plot(xrc, rc, "--", marker="X", color=TEAL, lw=1.5, ms=5, mec="white", mew=0.4, alpha=0.5)
aLt.plot(xrhh, rhh, "--", marker="X", color=CRIM, lw=1.5, ms=5, mec="white", mew=0.4, alpha=0.5)
aL.set_ylabel("compliance / attack success (%)", color=TEAL)
aL.legend([Line2D([0],[0],color="#555",marker="v",ls="-",lw=1.8,ms=7,mec="white"),
           Line2D([0],[0],color="#555",marker="X",ls="--",lw=1.5,ms=6,mec="white",alpha=0.6)],
          ["harmful − benign prompts (refusal_dm, x=−α)", "random control (equal magnitude)"],
          loc="upper left", fontsize=8.3, frameon=False,
          bbox_to_anchor=(0.02,0.98), title="task-prompt prior (not a breach label)", title_fontsize=8.4)

# ---------------- M: compliance directions (from training corpus) ----------------
aMt = frame(aM, "COMPLIANCE directions  →  they STEER",
            f"complied − refused (training corpus) · goals[50:] (n={NP}) · Llama judge", "α        +α = toward compliance  →")
COMP = [("comp_dm_out", "o", "OUTPUT locus (replay)"),
        ("comp_pre_qwen", "s", "pre-response · Qwen labels"),
        ("comp_pre_llama", "^", "pre-response · Llama labels")]
for d, mk, lab in COMP:
    xc, c = curve(HL, d, "comp"); xh, h = curve(HL, d, "harm")
    aM.plot(xc, c, "-", marker=mk, color=TEAL, lw=2.2, ms=6.5, mec="white", mew=0.6)
    aMt.plot(xh, h, "-", marker=mk, color=CRIM, lw=1.8, ms=5.5, mec="white", mew=0.5, alpha=0.9)
aM.legend([Line2D([0],[0],color="#555",marker=mk,ls="-",lw=1.8,ms=7,mec="white") for _,mk,_ in COMP],
          [lab for _,_,lab in COMP], loc="upper left", fontsize=8.3, frameon=False, bbox_to_anchor=(0.02,0.98),
          title="complied − refused (diff-in-means)", title_fontsize=8.4)

# ---------------- R: harm directions ----------------
aRt = frame(aR, "HARM directions  →  none STEER harm",
            "μ(≥4) − μ(≤2) etc. · crimson=harm (bold), teal=comp% (faint) · goals[50:] · Llama", "α        +α = toward harm  →")
HARM = [("harm_dm_llama", CVHl, "o", "μ(≥4) − μ(≤2) · OUTPUT"),
        ("harm_dm_resid", CVHl, "s", "OUTPUT · refusal+length removed"),
        ("harm_pre_llama", HL, "D", "μ(≥4) · pre-response"),
        ("random_1", CVHl, "X", "random control")]
for d, rows, mk, lab in HARM:
    xc, c = curve(rows, d, "comp"); xh, h = curve(rows, d, "harm")
    aR.plot(xc, c, "-", marker=mk, color=TEAL, lw=1.6, ms=5.5, mec="white", mew=0.5, alpha=0.55)
    aRt.plot(xh, h, "-", marker=mk, color=CRIM, lw=2.2, ms=6, mec="white", mew=0.6)
aRt.set_ylabel("harm uplift (Stage-B, 1–5)", color=CRIM)
aR.legend([Line2D([0],[0],color=CRIM,marker=mk,ls="-",lw=1.8,ms=7,mec="white") for _,_,mk,_ in HARM],
          [lab for _,_,_,lab in HARM], loc="upper left", fontsize=8.3, frameon=False, bbox_to_anchor=(0.02,0.98),
          title="harm direction", title_fontsize=8.4)

fig.suptitle("Decoded == steered (goals[50:]): the compliance directions and the prior task-only refusal direction both "
             "steer compliance; the decodable harm direction won't move harm past ~1.6", fontsize=11.5, fontweight="semibold", y=1.02)
fig.tight_layout(w_pad=3.0)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}/fig_decoded_steering.{ext}", dpi=200 if ext=="png" else None, bbox_inches="tight")
    print(f"wrote {FIG}/fig_decoded_steering.{ext}")
print(f"n_prompts={NP}")
