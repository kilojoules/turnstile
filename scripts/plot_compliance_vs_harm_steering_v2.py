"""Updated (regime-aware) compliance-vs-harm steering figure — FIRST DRAFT.

Incorporates the regime-dependence finding: compliance steering is a real lever
ONLY where the refusal gate is closed (baseline-refusing prompts). Where the
attacker's context already opened it (breach replies), the lever is spent. And
harm never clears its ceiling under ANY direction.

LEFT  : compliance % under refusal steering, TWO regimes overlaid —
        closed-gate (single-turn harmful JBB prompts, baseline refuses) rises 2.5->62%,
        open-gate (already-complied replies) stays flat. Harm (secondary axis, both judges)
        rises only to a ~2.3 ceiling even when the gate is fully opened.
LEFT data : refusal_harm_vs_compliance_v1 (Llama judged.jsonl + refusal_judged_qwen.jsonl),
            steer_refusal_replies (low set).
RIGHT : harm uplift under 5 harm-direction definitions + random on the replies, both judges —
        flat; no lever. (existing harm-null panel)
"""
import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/Users/julianquick/portfolio_copy/turnstile"; FIG = f"{R}/figures/causal_steering"; U = 7.563
TEAL, CRIM, GREY = "#1b9e77", "#c0455a", "#8a8a8a"
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4d4d4d", "axes.linewidth": 1.0, "figure.facecolor": "white"})


def load(p):
    return [json.loads(l) for l in open(p)]


def by_alpha(rows, field, akey="alpha"):
    d = defaultdict(list)
    for r in rows:
        d[r[akey]].append(r)
    xs = sorted(d)
    if field == "judge_compliance_unsafe":
        ys = [100 * np.mean([1 if x.get(field) else 0 for x in d[a]]) for a in xs]
    else:
        ys = [np.mean([x[field] for x in d[a] if isinstance(x.get(field), (int, float))]) for a in xs]
    return xs, ys


def reply_curve(rows, direction, field):
    d = defaultdict(list)
    for r in rows:
        if r.get("set") != "low": continue
        if r["direction"] == "baseline": d[0.0].append(r.get(field))
        elif r["direction"] == direction: d[r["magnitude"] / U].append(r.get(field))
    xs = sorted(d)
    if field == "judge_compliance_unsafe":
        ys = [100 * np.mean([1 if v else 0 for v in d[a]]) for a in xs]
    else:
        ys = [np.mean([v for v in d[a] if isinstance(v, (int, float))]) for a in xs]
    return xs, ys


HL = load(f"{R}/experiments/refusal_harm_vs_compliance_v1/judged.jsonl")
HQ = load(f"{R}/experiments/refusal_harm_vs_compliance_v1/refusal_judged_qwen.jsonl")
RRl = load(f"{R}/experiments/steer_refusal_replies/judged_llama.jsonl")

fig, (aL, aR) = plt.subplots(1, 2, figsize=(15.5, 5.8))

# ============ LEFT: compliance gate, two regimes ============
aL.set_xlim(1.62, -1.62)                       # inverted: subtract-refusal (open) -> right
aL.axvline(0, color="#d8d8d8", lw=1, ls=":")
xc, cc = by_alpha(HL, "judge_compliance_unsafe")           # closed-gate compliance
xo, co = reply_curve(RRl, "refusal_dm", "judge_compliance_unsafe")  # open-gate compliance
aL.plot(xc, cc, "-o", color=TEAL, lw=3, ms=6.5, label="compliance · gate CLOSED (harmful prompts)")
aL.plot(xo, co, ":o", color=TEAL, lw=1.9, ms=4.5, alpha=0.7, label="compliance · gate OPEN (complied replies)")
aL.set_ylabel("compliance / attack success (%)", color=TEAL); aL.tick_params(axis="y", labelcolor=TEAL)
aL.set_ylim(-3, 100)
aL.set_xlabel("steering strength  α        subtract the refusal direction (open gate)  " + r"$\rightarrow$")
# harm on secondary axis (closed-gate regime, both judges)
aLt = aL.twinx(); aLt.spines["top"].set_visible(False)
xh, hh = by_alpha(HL, "judge_harm_likert")
xhq, hhq = by_alpha(HQ, "judge_harm_likert")
aLt.plot(xh, hh, "-s", color=CRIM, lw=2.6, ms=5.5, label="harm · Llama (harmful prompts)")
aLt.plot(xhq, hhq, "--s", color=CRIM, lw=2.0, ms=5, mfc="white", label="harm · Qwen (harmful prompts)")
aLt.set_ylabel("harm uplift (Stage-B, 1–5)", color=CRIM); aLt.tick_params(axis="y", labelcolor=CRIM)
aLt.set_ylim(1, 5)
aLt.axhline(4, color=CRIM, lw=0.9, ls=":"); aLt.text(-1.55, 4.08, "'meaningful uplift' (4) — never reached", fontsize=8.5, color=CRIM, ha="left")
peak = max(hh)
aLt.axhline(peak, color="#c98", lw=0.9, ls="--"); aLt.text(1.55, peak + 0.08, f"harm ceiling ≈ {peak:.1f}", fontsize=8.5, color="#b07", ha="left")
aL.text(0.0, 1.10, "Compliance steering bites only where the gate is closed",
        transform=aL.transAxes, fontsize=13.5, fontweight="semibold", ha="left")
aL.text(0.0, 1.035, "refusal direction · baseline-REFUSING harmful prompts vs already-COMPLIED replies · gate opened → harm caps at ~2.3",
        transform=aL.transAxes, fontsize=9, color="#6b6b6b", ha="left")
h1, l1 = aL.get_legend_handles_labels(); h2, l2 = aLt.get_legend_handles_labels()
aL.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8.6, frameon=False, labelspacing=0.4)

# ============ RIGHT: harm directions on replies (both judges) ============
S5l = load(f"{R}/experiments/steer_5v1/judged_llama.jsonl"); S5q = load(f"{R}/experiments/steer_5v1/judged_qwen.jsonl")
P3l = load(f"{R}/experiments/phase3_resid_harm/judged_llama.jsonl"); P3q = load(f"{R}/experiments/phase3_resid_harm/judged_qwen.jsonl")
base_l = reply_curve(S5l, "harm_dm_llama", "judge_harm_likert")[1][0]
VAR = [("diff-in-means · Llama labels", S5l, S5q, "harm_dm_llama", "#3b6ea5"),
       ("diff-in-means · Qwen labels", S5l, S5q, "harm_dm_45v12_qwen", "#e07b39"),
       ("sharpened 5-vs-1 · Qwen", S5l, S5q, "harm_dm_5v1_qwen", "#4f9d69"),
       ("5-vs-1, refusal+length removed", S5l, S5q, "harm_dm_5v1_qwen_resid", "#7d6bb0"),
       ("diff-in-means, refusal+length removed", P3l, P3q, "harm_dm_resid", "#c9a441")]
aR.axhline(base_l, color="#b0b0b0", lw=1.0, ls="--"); aR.text(0.02, base_l + 0.07, f"baseline ≈ {base_l:.2f}", fontsize=9, color="#7a7a7a")
aR.axhline(4, color=CRIM, lw=0.9, ls=":"); aR.text(1.55, 4.08, "'meaningful uplift' (4)", fontsize=8.5, color=CRIM, ha="right")
for lab, lr, qr, d, col in VAR:
    xl, yl = reply_curve(lr, d, "judge_harm_likert"); xq, yq = reply_curve(qr, d, "judge_harm_likert")
    aR.plot(xl, yl, "-o", color=col, lw=2.2, ms=5.5, label=lab)
    aR.plot(xq, yq, "--o", color=col, lw=1.5, ms=4, mfc="white", alpha=0.8)
xrl, yrl = reply_curve(S5l, "random_1", "judge_harm_likert"); xrq, yrq = reply_curve(S5q, "random_1", "judge_harm_likert")
aR.plot(xrl, yrl, "-D", color=GREY, lw=2.0, ms=5, label="random control")
aR.plot(xrq, yrq, "--D", color=GREY, lw=1.4, ms=4, mfc="white", alpha=0.8)
aR.set_ylim(1, 5); aR.set_xlim(-0.06, 1.62)
aR.set_xlabel("steering strength  α        add the harm direction  " + r"$\rightarrow$")
aR.set_ylabel("harm uplift (Stage-B, 1–5)")
aR.text(0.0, 1.10, "No harm direction is a lever, under either judge",
        transform=aR.transAxes, fontsize=13.5, fontweight="semibold", ha="left")
aR.text(0.0, 1.035, "5 harm-direction definitions + random · compliant replies · solid=Llama-70B, dashed=Qwen-72B",
        transform=aR.transAxes, fontsize=9, color="#6b6b6b", ha="left")
vh = [Line2D([0], [0], color=c, lw=2.3) for *_, c in [(v[0], v[-1]) for v in VAR]] + [Line2D([0], [0], color=GREY, lw=2)]
aR.legend(vh, [v[0] for v in VAR] + ["random control"], loc="lower left", fontsize=8.4, frameon=False, labelspacing=0.4)

fig.tight_layout(w_pad=3.0)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}/fig_compliance_vs_harm_steering_v2.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {FIG}/fig_compliance_vs_harm_steering_v2.{ext}")
