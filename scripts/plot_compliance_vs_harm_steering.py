"""Apples-to-apples two-panel: the refusal/compliance gate is a real lever, the
harm direction is not — BOTH evaluated on the SAME low-uplift compliant replies,
scored by BOTH judges (Llama-70B solid, Qwen-72B dashed).

Left  : steer the refusal (compliance-gate) direction, two-sided. Closing the gate
        (add refusal) pulls compliance AND harm down together; opening it never
        lifts harm past its ~2.8 ceiling. random_1 shown as a gate control.
Right : steer 5 distinct harm-direction definitions + random. No definition lifts
        harm above baseline under either judge; large pushes only degrade it.

Data (all low-set curves complete for both judges):
  experiments/steer_refusal_replies/judged_{llama,qwen}.jsonl   (refusal_dm, random_1)
  experiments/steer_5v1/judged_{llama,qwen}.jsonl               (4 harm dirs + random)
  experiments/phase3_resid_harm/judged_{llama,qwen}.jsonl       (dm-resid)
"""
import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/Users/julianquick/portfolio_copy/turnstile"
FIG = f"{R}/figures/causal_steering"
U = 7.563
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


def curve(rows, direction, field, setname="low"):
    """Return (alphas, ys, ns) for `direction` (plus shared baseline at alpha=0)."""
    by = defaultdict(list)
    for r in rows:
        if r.get("set") != setname:
            continue
        if r["direction"] == "baseline":
            by[0.0].append(r.get(field))
        elif r["direction"] == direction:
            by[r["magnitude"]].append(r.get(field))
    mags = sorted(by)
    xs = [m / U for m in mags]
    ns = [len(by[m]) for m in mags]
    if field == "judge_compliance_unsafe":
        ys = [100 * np.mean([1 if v else 0 for v in by[m]]) for m in mags]
    else:
        ys = [np.mean([v for v in by[m] if isinstance(v, (int, float))]) for m in mags]
    return xs, ys, ns


# ---------------- load ----------------
RRl = load(f"{R}/experiments/steer_refusal_replies/judged_llama.jsonl")
RRq = load(f"{R}/experiments/steer_refusal_replies/judged_qwen.jsonl")
S5l = load(f"{R}/experiments/steer_5v1/judged_llama.jsonl")
S5q = load(f"{R}/experiments/steer_5v1/judged_qwen.jsonl")
P3l = load(f"{R}/experiments/phase3_resid_harm/judged_llama.jsonl")
P3q = load(f"{R}/experiments/phase3_resid_harm/judged_qwen.jsonl")

base_l = curve(S5l, "harm_dm_llama", "judge_harm_likert")[1][0]   # shared baseline harm (Llama)
base_q = curve(S5q, "harm_dm_llama", "judge_harm_likert")[1][0]   # shared baseline harm (Qwen)

fig, (aL, aR) = plt.subplots(1, 2, figsize=(15.5, 5.7))

# ================= LEFT: refusal / compliance gate =================
# secondary axis = harm (crimson); primary = compliance % (teal)
xr, hL, _ = curve(RRl, "refusal_dm", "judge_harm_likert")
_,  hQ, _ = curve(RRq, "refusal_dm", "judge_harm_likert")
xc, cL, _ = curve(RRl, "refusal_dm", "judge_compliance_unsafe")
_,  crL, _ = curve(RRl, "random_1", "judge_compliance_unsafe")

aL.set_xlim(-1.62, 1.62)
aL.axvline(0, color="#d5d5d5", lw=1, ls=":")
# compliance (primary, teal)
aL.plot(xc, cL, "-o", color=TEAL, lw=2.6, ms=6, label="compliance % (Llama JBB)")
aL.plot(xc, crL, ":o", color=TEAL, lw=1.6, ms=4, alpha=0.65, label="compliance % · random dir")
aL.set_ylabel("compliance / attack success (%)", color=TEAL)
aL.tick_params(axis="y", labelcolor=TEAL)
aL.set_ylim(0, 100)
aL.set_xlabel("steering strength  α        add refusal (close gate)  $\\rightarrow$")
# harm (secondary, crimson)
aLt = aL.twinx(); aLt.spines["top"].set_visible(False)
aLt.plot(xr, hL, "-s", color=CRIM, lw=2.8, ms=6, label="harm · Llama")
aLt.plot(xr, hQ, "--s", color=CRIM, lw=2.2, ms=5.5, mfc="white", label="harm · Qwen")
aLt.set_ylabel("harm uplift (Stage-B, 1–5)", color=CRIM)
aLt.tick_params(axis="y", labelcolor=CRIM)
aLt.set_ylim(1, 5)
aLt.axhline(4, color=CRIM, lw=0.9, ls=":")
aLt.text(1.55, 4.07, "'meaningful uplift' (4)", fontsize=8.5, color=CRIM, ha="right")
aLt.axhline(base_l, color="#b0b0b0", lw=0.9, ls="--")
aLt.text(-1.55, base_l + 0.07, f"harm ceiling ≈ {base_l:.2f}", fontsize=8.5, color="#7a7a7a", ha="left")
aLt.annotate("close gate $\\rightarrow$\ncompliance & harm\nfall together", xy=(1.2, 1.5), xytext=(0.35, 1.35),
             fontsize=8.5, color="#333", ha="left",
             arrowprops=dict(arrowstyle="->", color="#666", lw=1.1))
aL.text(0.0, 1.10, "Steer the REFUSAL / COMPLIANCE gate  $\\rightarrow$  a real lever",
        transform=aL.transAxes, fontsize=13.5, fontweight="semibold", ha="left")
aL.text(0.0, 1.035, "refusal direction · low-uplift compliant replies (n=32/pt) · both judges",
        transform=aL.transAxes, fontsize=10, color="#6b6b6b", ha="left")
# combined legend for left
h1, l1 = aL.get_legend_handles_labels()
h2, l2 = aLt.get_legend_handles_labels()
aL.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.8, frameon=False, labelspacing=0.4)

# ================= RIGHT: harm directions =================
VARIANTS = [
    ("diff-in-means · Llama labels",          S5l, S5q, "harm_dm_llama",          "#3b6ea5"),
    ("diff-in-means · Qwen labels",           S5l, S5q, "harm_dm_45v12_qwen",     "#e07b39"),
    ("sharpened 5-vs-1 · Qwen",               S5l, S5q, "harm_dm_5v1_qwen",       "#4f9d69"),
    ("5-vs-1, refusal+length removed",        S5l, S5q, "harm_dm_5v1_qwen_resid", "#7d6bb0"),
    ("diff-in-means, refusal+length removed", P3l, P3q, "harm_dm_resid",          "#c9a441"),
]
aR.axhline(base_l, color="#b0b0b0", lw=1.0, ls="--")
aR.text(0.02, base_l + 0.07, f"baseline harm ≈ {base_l:.2f}", fontsize=9, color="#7a7a7a")
aR.axhline(4, color=CRIM, lw=0.9, ls=":")
aR.text(1.55, 4.07, "'meaningful uplift' (4) — never reached", fontsize=8.5, color=CRIM, ha="right")
for lab, lrows, qrows, d, col in VARIANTS:
    xl, yl, _ = curve(lrows, d, "judge_harm_likert")
    xq, yq, _ = curve(qrows, d, "judge_harm_likert")
    aR.plot(xl, yl, "-o", color=col, lw=2.2, ms=5.5, label=lab)
    aR.plot(xq, yq, "--o", color=col, lw=1.6, ms=4, mfc="white", alpha=0.85)
# random control
xrl, yrl, _ = curve(S5l, "random_1", "judge_harm_likert")
xrq, yrq, _ = curve(S5q, "random_1", "judge_harm_likert")
aR.plot(xrl, yrl, "-D", color=GREY, lw=2.0, ms=5, label="random control")
aR.plot(xrq, yrq, "--D", color=GREY, lw=1.5, ms=4, mfc="white", alpha=0.85)

aR.set_ylim(1, 5); aR.set_xlim(-0.06, 1.62)
aR.set_xlabel("steering strength  α        add the harm direction  $\\rightarrow$")
aR.set_ylabel("harm uplift (Stage-B, 1–5)")
aR.text(0.0, 1.10, "Steer the HARM direction  $\\rightarrow$  moves nothing, every way we tried",
        transform=aR.transAxes, fontsize=13.5, fontweight="semibold", ha="left")
aR.text(0.0, 1.035, "5 distinct harm-direction definitions + random · same replies · both judges",
        transform=aR.transAxes, fontsize=10, color="#6b6b6b", ha="left")
var_handles = [Line2D([0], [0], color=c, lw=2.4) for *_, c in [(v[0], v[-1]) for v in VARIANTS]]
var_labels = [v[0] for v in VARIANTS]
judge_key = [Line2D([0], [0], color="#555", lw=2.2, ls="-"),
             Line2D([0], [0], color="#555", lw=1.6, ls="--", marker="o", mfc="white")]
leg1 = aR.legend([Line2D([0], [0], color=c, lw=2.4) for *_, c in
                  [(v[0], v[-1]) for v in VARIANTS]] + [Line2D([0], [0], color=GREY, lw=2.0)],
                 var_labels + ["random control"], loc="lower left", fontsize=8.6,
                 frameon=False, handlelength=1.7, labelspacing=0.45)
aR.add_artist(leg1)
aR.legend(judge_key, ["solid = Llama-70B judge", "dashed = Qwen-72B judge"],
          loc="upper right", fontsize=8.6, frameon=False, handlelength=2.0, labelspacing=0.35)

fig.tight_layout(w_pad=3.0)
for ext in ("png", "pdf"):
    out = f"{FIG}/fig_compliance_vs_harm_steering.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {out}")
