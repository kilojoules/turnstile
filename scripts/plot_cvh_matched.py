"""Apples-to-apples: steer COMPLIANCE vs steer HARM, from the IDENTICAL baseline.

Same 40 baseline-refusing JBB prompts, same start (comp ~2.5%, harm ~1.07), same
dual y-axes (compliance % + harm 1-5), both judges. Left panel sweeps the refusal
(compliance) direction; right panel sweeps the harm direction. x-axis of each panel
is "steering strength toward the named target" so both read left->right the same way.

Left  data: experiments/refusal_harm_vs_compliance_v1 (judged.jsonl + refusal_judged_qwen.jsonl)
Right data: experiments/steer_cvh_matched (judged_llama.jsonl + judged_qwen.jsonl)  [harm_dm_llama]
"""
import json, os
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/Users/julianquick/portfolio_copy/turnstile"; FIG = f"{R}/figures/causal_steering"
TEAL, CRIM = "#1b9e77", "#c0455a"
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4d4d4d", "axes.linewidth": 1.0, "figure.facecolor": "white"})


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def curve(rows, field, xkey, xsign=1.0, direction=None, base_dir="baseline"):
    """Return (xs, ys) aggregated by signed alpha; includes baseline at x=0."""
    d = defaultdict(list)
    for r in rows:
        if direction is not None:
            if r.get("direction") == base_dir:
                d[0.0].append(r.get(field)); continue
            if r.get("direction") != direction:
                continue
        d[xsign * r[xkey]].append(r.get(field))
    xs = sorted(d)
    if field == "judge_compliance_unsafe":
        ys = [100 * np.mean([1 if v else 0 for v in d[a]]) for a in xs]
    else:
        ys = [np.mean([v for v in d[a] if isinstance(v, (int, float))]) for a in xs]
    return xs, ys


# ---- data ----
def keepdir(rows, dirs):
    return [r for r in rows if r.get("direction") in dirs]

# ext10 = the 10 extra held-out prompts goals[90:100] (refusal + harm dirs at matched grids)
E10L = load(f"{R}/experiments/steer_cvh_ext10/judged_llama.jsonl")
E10Q = load(f"{R}/experiments/steer_cvh_ext10/judged_qwen.jsonl")
CDIRS = {"refusal_dm", "baseline"}                                   # feed the compliance panel
HDIRS = {"harm_dm_llama", "harm_dm_resid", "random_1", "baseline"}   # feed the harm panel

CL = load(f"{R}/experiments/refusal_harm_vs_compliance_v1/judged.jsonl") + keepdir(E10L, CDIRS)       # compliance, Llama (40 + 10)
CQ = load(f"{R}/experiments/refusal_harm_vs_compliance_v1/refusal_judged_qwen.jsonl") + keepdir(E10Q, CDIRS)  # Qwen
HL = load(f"{R}/experiments/steer_cvh_matched/judged_llama.jsonl") + \
     load(f"{R}/experiments/steer_cvh_fill/judged_llama.jsonl") + keepdir(E10L, HDIRS)                # harm, Llama (40 + fill + 10)
HQ = load(f"{R}/experiments/steer_cvh_matched/judged_qwen.jsonl") + \
     load(f"{R}/experiments/steer_cvh_fill/judged_qwen.jsonl") + keepdir(E10Q, HDIRS)                 # Qwen
print(f"compliance-dir rows L/Q = {len(CL)}/{len(CQ)}  |  harm-dir rows L/Q = {len(HL)}/{len(HQ)}")
if not HL or not HQ:
    print("harm-dir judged data not present yet -> waiting"); raise SystemExit

HARM_DIR = "harm_dm_llama"
fig, (aL, aR) = plt.subplots(1, 2, figsize=(15.0, 5.7), sharey=False)


def panel(ax, comp_rows_l, comp_rows_q, harm_rows_l, harm_rows_q, xkey, xsign, direction,
          title, subtitle, xlabel):
    ax.set_xlim(-1.62, 1.62); ax.axvline(0, color="#d8d8d8", lw=1, ls=":")
    # compliance % (primary, teal) — both judges
    xc, cl = curve(comp_rows_l, "judge_compliance_unsafe", xkey, xsign, direction)
    _,  cq = curve(comp_rows_q, "judge_compliance_unsafe", xkey, xsign, direction)
    ax.plot(xc, cl, "-o", color=TEAL, lw=2.7, ms=6, label="compliance · Llama")
    ax.plot(xc, cq, "--o", color=TEAL, lw=1.8, ms=4.5, mfc="white", alpha=0.85, label="compliance · Qwen")
    ax.set_ylabel("compliance / attack success (%)", color=TEAL); ax.tick_params(axis="y", labelcolor=TEAL)
    ax.set_ylim(-3, 100); ax.set_xlabel(xlabel)
    # harm (secondary, crimson) — both judges
    at = ax.twinx(); at.spines["top"].set_visible(False)
    xh, hl = curve(harm_rows_l, "judge_harm_likert", xkey, xsign, direction)
    _,  hq = curve(harm_rows_q, "judge_harm_likert", xkey, xsign, direction)
    at.plot(xh, hl, "-s", color=CRIM, lw=2.5, ms=5.5, label="harm · Llama")
    at.plot(xh, hq, "--s", color=CRIM, lw=1.8, ms=5, mfc="white", alpha=0.85, label="harm · Qwen")
    at.set_ylabel("harm uplift (Stage-B, 1–5)", color=CRIM); at.tick_params(axis="y", labelcolor=CRIM)
    at.set_ylim(1, 5)
    at.axhline(4, color=CRIM, lw=0.9, ls=":"); at.text(-1.55, 4.08, "'meaningful uplift' (4)", fontsize=8.5, color=CRIM, ha="left")
    ax.text(0.0, 1.10, title, transform=ax.transAxes, fontsize=13.5, fontweight="semibold", ha="left")
    ax.text(0.0, 1.035, subtitle, transform=ax.transAxes, fontsize=9, color="#6b6b6b", ha="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = at.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.6, frameon=False, labelspacing=0.35,
              bbox_to_anchor=(0.02, 0.98))
    return at


# LEFT: compliance direction. x = toward-compliance strength = -alpha_refusal.
atL = panel(aL, CL, CQ, CL, CQ, "alpha", -1.0, None,
      "Steer the COMPLIANCE direction (refusal)",
      "40 baseline-refusing JBB prompts · both judges · x>0 = subtract refusal (toward compliance)",
      "steering strength toward compliance  →")
# RIGHT: harm direction. x = toward-harm strength = +alpha.
atR = panel(aR, HL, HQ, HL, HQ, "alpha", 1.0, HARM_DIR,
      "Steer the HARM direction",
      "SAME 40 prompts · SAME baseline · both judges · x>0 = add harm direction",
      "steering strength toward harm  →")
# random-direction harm as a reference on the harm panel: the harm dir is no better than noise
xr, hr = curve(HL, "judge_harm_likert", "alpha", 1.0, "random_1")
atR.plot([abs(x) for x in xr if x], [hr[i] for i, x in enumerate(xr) if x], ":", color="#8a8a8a", lw=1.6, alpha=0.9)
atR.plot(xr, hr, "d", color="#8a8a8a", ms=4, alpha=0.7, label="_")
atR.text(1.5, 1.95, "random push\n(↑harm MORE than\nthe harm dir)", fontsize=7.5, color="#666", ha="right", va="top")
# shared "max harm reached by ANY steering" ceiling (= what the compliance dir achieves)
for at in (atL, atR):
    at.axhline(2.30, color="#b07", lw=0.8, ls=(0, (4, 3)), alpha=0.6)
atL.text(-1.55, 2.36, "max harm any steering reaches ≈ 2.3", fontsize=8, color="#b07", ha="left")

fig.suptitle("Same prompts, same baseline (2.5% compliant / harm 1.07): steering harm raises harm LESS than steering compliance — or even a random push",
             fontsize=12, fontweight="semibold", y=1.02)
fig.tight_layout(w_pad=3.2)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}/fig_cvh_matched.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {FIG}/fig_cvh_matched.{ext}")

# sanity: baselines must match across the two datasets
b_comp = np.mean([1 if r.get("judge_compliance_unsafe") else 0 for r in CL if r["alpha"] == 0]) * 100
b_harm = np.mean([r["judge_harm_likert"] for r in HL if r.get("direction") == "baseline"])
b_comp_h = np.mean([1 if r.get("judge_compliance_unsafe") else 0 for r in HL if r.get("direction") == "baseline"]) * 100
print(f"BASELINE check — compliance dir set: comp={b_comp:.1f}%  |  harm dir set: comp={b_comp_h:.1f}% harm={b_harm:.2f}")
