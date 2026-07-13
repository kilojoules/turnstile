"""Analyze the composed-steering (harm-on-open) experiment.

Key question: once the refusal gate is OPEN (alpha_ref=-1.0, harm~2.3 ceiling on the
40 baseline-refusing JBB prompts), does adding a harm direction push harm ABOVE 2.3
while random does not? If yes -> an independent harm lever exists once compliant.
If no -> the harm-null is robust even in the favorable (headroom) regime.

Prints a table and writes figures/causal_steering/fig_harm_on_open.{png,pdf}.
"""
import json, os
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/Users/julianquick/portfolio_copy/turnstile"
D = f"{R}/experiments/steer_harm_on_open"
FIG = f"{R}/figures/causal_steering"; os.makedirs(FIG, exist_ok=True)


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def agg(rows):
    """key -> dict(harm mean, comp %, n). key=(arm,harm_dir,alpha_harm)."""
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r.get("harm_dir"), r.get("alpha_harm", 0.0))].append(r)
    out = {}
    for k, g in by.items():
        h = [x["judge_harm_likert"] for x in g if isinstance(x.get("judge_harm_likert"), (int, float))]
        c = [1 if x.get("judge_compliance_unsafe") else 0 for x in g]
        rf = [1 if x.get("is_refusal") else 0 for x in g]
        out[k] = {"harm": np.mean(h) if h else np.nan, "comp": 100 * np.mean(c) if c else np.nan,
                  "refuse": 100 * np.mean(rf) if rf else np.nan, "n": len(g)}
    return out


L = load(f"{D}/judged_llama.jsonl")
Q = load(f"{D}/judged_qwen.jsonl")
print(f"llama rows={len(L)}  qwen rows={len(Q)}")
if not L:
    print("no results yet"); raise SystemExit
aL, aQ = agg(L), agg(Q)

# reference points
base = ("baseline", None, 0.0)
openk = ("open", None, 0.0)
harm_base_l = aL.get(base, {}).get("harm", np.nan)
open_harm_l = aL.get(openk, {}).get("harm", np.nan)
open_comp_l = aL.get(openk, {}).get("comp", np.nan)
print(f"\nbaseline harm(L)={harm_base_l:.2f}  |  OPEN harm(L)={open_harm_l:.2f} comp={open_comp_l:.0f}%  <- the ceiling to beat")

# ---- table ----
GROUPS = [
    ("open + harm_resid",    "harm_dm_resid",          -1.0),
    ("open + harm_5v1resid", "harm_dm_5v1_qwen_resid", -1.0),
    ("open + harm_raw",      "harm_dm_llama",          -1.0),
    ("harm_alone (no open)", "harm_dm_resid",           0.0),
    ("open + random",        "random_1",               -1.0),
]
ARMMAP = {"open + harm_resid": "open+harm_resid", "open + harm_5v1resid": "open+harm_5v1resid",
          "open + harm_raw": "open+harm_raw", "harm_alone (no open)": "harm_alone",
          "open + random": "open+random"}
print(f"\n{'arm':22s} {'aharm':>5} | {'harmL':>6} {'harmQ':>6} {'compL%':>6} {'refuse%':>7}   Δharm_vs_open(L)")
print("-" * 78)
for label, hd, aref in GROUPS:
    arm = ARMMAP[label]
    ahs = sorted({k[2] for k in aL if k[0] == arm})
    for ah in ahs:
        k = (arm, hd, ah)
        rl, rq = aL.get(k, {}), aQ.get(k, {})
        dh = rl.get("harm", np.nan) - open_harm_l
        print(f"{label:22s} {ah:5.1f} | {rl.get('harm',np.nan):6.2f} {rq.get('harm',np.nan):6.2f} "
              f"{rl.get('comp',np.nan):6.0f} {rl.get('refuse',np.nan):7.0f}   {dh:+.2f}")

# ---- figure ----
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4d4d4d", "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(9.0, 5.8))
VAR = [("open + harm_resid (residualized)", "open+harm_resid", "harm_dm_resid", "#3b6ea5"),
       ("open + harm_5v1resid (residualized)", "open+harm_5v1resid", "harm_dm_5v1_qwen_resid", "#4f9d69"),
       ("open + harm_raw (dm-llama)", "open+harm_raw", "harm_dm_llama", "#e07b39"),
       ("open + random (control)", "open+random", "random_1", "#8a8a8a")]
for lab, arm, hd, col in VAR:
    ks = sorted([k for k in aL if k[0] == arm], key=lambda z: z[2])
    if not ks: continue
    xs = [0.0] + [k[2] for k in ks]                      # anchor at open (alpha_harm=0)
    yL = [open_harm_l] + [aL[k]["harm"] for k in ks]
    yQ = [aQ.get(openk, {}).get("harm", np.nan)] + [aQ.get(k, {}).get("harm", np.nan) for k in ks]
    ax.plot(xs, yL, "-o", color=col, lw=2.3, ms=6, label=lab)
    ax.plot(xs, yQ, "--o", color=col, lw=1.6, ms=4, mfc="white", alpha=0.85)

ax.axhline(open_harm_l, color="#666", lw=1.2, ls="--")
ax.text(0.02, open_harm_l + 0.06, f"gate OPEN, no harm push  (harm ≈ {open_harm_l:.2f})", fontsize=9, color="#444")
ax.axhline(harm_base_l, color="#bbb", lw=1.0, ls=":")
ax.text(0.02, harm_base_l + 0.06, f"baseline / refusing (harm ≈ {harm_base_l:.2f})", fontsize=8.5, color="#999")
ax.axhline(4, color="#c0455a", lw=0.9, ls=":")
ax.text(1.5, 4.06, "'meaningful uplift' (4)", fontsize=8.5, color="#c0455a", ha="right")
ax.set_ylim(1, 5); ax.set_xlim(-0.05, 1.6)
ax.set_xlabel("harm-direction strength  α_harm   (gate already opened)")
ax.set_ylabel("harm uplift (Stage-B, 1–5)")
ax.text(0.0, 1.06, "Once the gate is open, no harm direction lifts harm above the ceiling",
        transform=ax.transAxes, fontsize=13, fontweight="semibold", ha="left")
ax.text(0.0, 1.015, "40 baseline-refusing JBB prompts · open gate (α_ref=−1.0) + harm dir · solid=Llama-70B, dashed=Qwen-72B",
        transform=ax.transAxes, fontsize=9, color="#6b6b6b", ha="left")
ax.legend(loc="upper left", fontsize=9, frameon=False, bbox_to_anchor=(0.0, 0.94))
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}/fig_harm_on_open.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {FIG}/fig_harm_on_open.{ext}")
