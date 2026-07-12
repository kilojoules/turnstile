"""Two-panel summary: steering the COMPLIANCE gate moves both compliance and harm
(harm only to a capped ceiling), while steering the HARM direction — in every variant
we tried — moves nothing. Left: refusal/gate direction on 40 JBB harmful prompts.
Right: 5 distinct harm-direction definitions + random on low-uplift compliant replies.
"""
import json, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/Users/julianquick/portfolio_copy/turnstile"; FIG = f"{R}/figures/causal_steering"; U = 7.563
TEAL, CRIM, GREY = "#1b9e77", "#c0455a", "#8a8a8a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4d4d4d", "axes.linewidth": 1.0, "figure.facecolor": "white",
})

def load(p):
    return [json.loads(l) for l in open(p)]

# ---------------- LEFT: compliance gate (refusal direction), JBB harmful ----------------
rb = defaultdict(list)
for r in load(f"{R}/experiments/refusal_harm_vs_compliance_v1/judged.jsonl"):
    rb[r["alpha"]].append(r)
A = sorted(rb)
comp = [100*sum(1 for r in rb[a] if r.get("judge_compliance_unsafe"))/len(rb[a]) for a in A]
harm = [np.mean([r["judge_harm_likert"] for r in rb[a] if isinstance(r.get("judge_harm_likert"), (int, float))]) for a in A]

# ---------------- RIGHT: every harm-direction variant, low-uplift replies ----------------
def hseries(path, direction):
    by = defaultdict(list)
    for r in load(path):
        if r["set"] == "low": by[(r["direction"], r["magnitude"])].append(r)
    mags = [0.0, 2.8, 5.5, 11.1]; ys = []
    for m in mags:
        k = ("baseline", 0.0) if m == 0 else (direction, m)
        v = [r["judge_harm_likert"] for r in by.get(k, []) if isinstance(r.get("judge_harm_likert"), (int, float))]
        ys.append(np.mean(v) if v else np.nan)
    return [m/U for m in mags], ys

S5 = f"{R}/experiments/steer_5v1/judged_llama.jsonl"
P3 = f"{R}/experiments/phase3_resid_harm/judged_llama.jsonl"
VARIANTS = [
    ("diff-in-means · Llama labels",         S5, "harm_dm_llama",          "#3b6ea5", "-o"),
    ("diff-in-means · Qwen labels",          S5, "harm_dm_45v12_qwen",     "#e07b39", "-o"),
    ("sharpened 5-vs-1 · Qwen",              S5, "harm_dm_5v1_qwen",       "#4f9d69", "-s"),
    ("5-vs-1, refusal+length removed",       S5, "harm_dm_5v1_qwen_resid", "#7d6bb0", "-^"),
    ("diff-in-means, refusal+length removed",P3, "harm_dm_resid",          "#c9a441", "-D"),
]
xr, base = hseries(S5, "harm_dm_llama"); baseline = base[0]

fig, (aL, aR) = plt.subplots(1, 2, figsize=(15.5, 5.4))

# --- LEFT ---
aL.plot(A, comp, "-o", color=TEAL, lw=3, ms=6.5)
aL.set_ylabel("compliance / attack success (%)", color=TEAL); aL.tick_params(axis="y", labelcolor=TEAL)
aL.set_ylim(-4, 100); aL.set_xlim(1.6, -1.6)  # inverted: subtract-refusal (open gate) increases rightward
aL.set_xlabel("steering strength  α    (subtract the refusal direction  →)")
aL.axvline(0, color="#cccccc", lw=1, ls=":")
aLt = aL.twinx(); aLt.spines["top"].set_visible(False)
aLt.plot(A, harm, "-s", color=CRIM, lw=3, ms=6)
aLt.set_ylabel("harm uplift (Stage-B, 1–5)", color=CRIM); aLt.tick_params(axis="y", labelcolor=CRIM)
aLt.set_ylim(1, 5)
aLt.axhline(2.3, color=CRIM, lw=1, ls="--"); aLt.text(1.5, 2.42, "capability ceiling ≈ 2.3", fontsize=9.5, color=CRIM, ha="left")
aLt.axhline(4, color=CRIM, lw=0.9, ls=":"); aLt.text(1.5, 4.08, "'meaningful uplift' (4) — never reached", fontsize=9, color=CRIM, ha="left")
aL.annotate("", xy=(-1.0, 62), xytext=(-1.0, 27), arrowprops=dict(arrowstyle="<->", color="#444", lw=1.4))
aL.text(-0.92, 45, "the gap\n= the finding", fontsize=9.5, ha="left", color="#222", fontweight="semibold")
aL.text(0.0, 1.10, "Steer the COMPLIANCE gate  →  moves both (harm only to a ceiling)",
        transform=aL.transAxes, fontsize=13.5, fontweight="semibold", ha="left")
aL.text(0.0, 1.035, "refusal direction · 40 JailbreakBench harmful prompts · Llama-70B judge",
        transform=aL.transAxes, fontsize=10.5, color="#6b6b6b", ha="left")

# --- RIGHT ---
aR.axhline(baseline, color="#9a9a9a", lw=1.4, ls="--")
aR.text(0.02, baseline+0.08, f"baseline {baseline:.2f}", fontsize=9.5, color="#6b6b6b")
for lab, path, d, col, mk in VARIANTS:
    x, y = hseries(path, d)
    aR.plot(x, y, mk, color=col, lw=2.2, ms=6, label=lab)
xrand, yrand = hseries(S5, "random_1")
aR.plot(xrand, yrand, "--o", color=GREY, lw=1.8, ms=5, label="random control")
aR.axhline(4, color=CRIM, lw=0.9, ls=":"); aR.text(1.47, 4.08, "'meaningful uplift' (4)", fontsize=9, color=CRIM, ha="right")
aR.set_ylim(1, 5); aR.set_xlim(-0.08, 1.6)
aR.set_xlabel("steering strength  α    (add the harm direction  →)")
aR.set_ylabel("harm uplift (Stage-B, 1–5)")
aR.legend(loc="lower left", fontsize=9.5, frameon=False, handlelength=1.7, labelspacing=0.5)
aR.text(0.0, 1.10, "Steer the HARM direction  →  moves nothing, every way we tried",
        transform=aR.transAxes, fontsize=13.5, fontweight="semibold", ha="left")
aR.text(0.0, 1.035, "5 distinct harm-direction definitions + random · low-uplift compliant replies · Llama-70B judge",
        transform=aR.transAxes, fontsize=10.5, color="#6b6b6b", ha="left")

fig.tight_layout(w_pad=3.0)
for ext in ("png", "pdf"):
    out = f"{FIG}/fig_compliance_vs_harm_steering.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {out}")
