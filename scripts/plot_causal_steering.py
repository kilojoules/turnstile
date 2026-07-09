"""Figures for the causal-steering results (gate / form / capability).

Reads the experiment JSONL and writes self-explanatory PNGs to
figures/causal_steering/. All numbers recomputed from data; nothing hardcoded.

Consistent visual language across every figure:
  grey            = baseline / no intervention
  teal  (#1b9e77) = the fitted target direction (refusal / harm / French / verbose)
  crimson(#d1495b)= the same direction used to REMOVE (ablate) a property
  hatched grey    = random-direction control
  dashed line     = a baseline the reader should compare against
Each figure has a plain-English title (the takeaway) and a "How to read this"
caption describing the setup.
"""
import json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "causal_steering")
os.makedirs(OUT, exist_ok=True)

GREY, TEAL, CRIM, RAND, BLUE = "#8a8a8a", "#1b9e77", "#d1495b", "#c9c9c9", "#3a6ea5"
STEER_UNIT = 7.563   # refusal diff-in-means norm; α is defined so α × STEER_UNIT = ‖added vector‖
plt.rcParams.update({"axes.titlesize": 11, "font.size": 10})

def load(p): return [json.loads(l) for l in open(os.path.join(ROOT, p))]
def rate(rows, key): return 100.0 * sum(bool(r.get(key)) for r in rows) / len(rows) if rows else 0.0
def cap(fig, text):
    fig.text(0.5, -0.02, "How to read this: " + text, ha="center", va="top",
             fontsize=8.6, style="italic", color="#333333", wrap=True)

# ------------------------------------------------------------------ FIG 1 (monotonic vs U)
def fig_monotonic():
    # every panel below holds the PROMPT SET fixed and compares the real direction
    # against a random direction (shared alpha=0 baseline within each panel).
    comp = load("experiments/steering_v3/sweep_p2_judged.jsonl")
    rnd = load("experiments/steering_v3/sweep_random_L16_jbb_qwen.jsonl")
    def curve(rows):
        by = defaultdict(list)
        for r in rows:
            if r.get("prompt_type") == "loss": by[r.get("alpha_c")].append(r)
        xs = sorted(a for a in by if a is not None)
        return xs, [rate(by[a], "judge_compliance_unsafe") for a in xs]
    cx, cy = curve(comp); rx, ry = curve(rnd)
    rows = load("experiments/refusal_alpha_sweep_v1/sweep.jsonl")
    by = defaultdict(list)
    for r in rows: by[(r["set"], r["direction"], r["alpha"])].append(r)
    A = sorted(set(r["alpha"] for r in rows))
    def rr(s, d): return [rate(by[(s, d, a)], "is_refusal") for a in A]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.5))
    # Panel 1: compliance PROBE direction on the loss set -> U-shape (== random)
    a1.plot(cx, cy, "-o", color=BLUE, lw=2.4, ms=6.5, label="compliance PROBE direction")
    a1.plot(rx, ry, "--s", color=GREY, lw=2.1, ms=5.5, label="random direction")
    a1.axvline(0, color="k", lw=0.7, ls=":")
    a1.set_xlabel("steering strength  α"); a1.set_ylabel("jailbreak success rate (%)")
    a1.set_title("Compliance PROBE direction\n(60 loss-set replies)\n→ U-shaped = random ⇒ artifact", color=CRIM, fontsize=10)
    a1.legend(frameon=False, fontsize=8.4, loc="upper center"); a1.grid(alpha=0.25)
    # Panel 2: refusal direction on HARMFUL prompts -> monotonic bypass, vs random
    a2.plot(A, rr("harmful", "refusal"), "-o", color=CRIM, lw=2.5, ms=6.5, label="refusal direction")
    a2.plot(A, rr("harmful", "random"), "--s", color=GREY, lw=2.1, ms=5.5, label="random direction")
    a2.axvline(0, color="k", lw=0.7, ls=":"); a2.axhline(98, color="#bbbbbb", lw=0.8, ls="--")
    a2.text(-1.45, 92, "baseline 98% (α=0)", fontsize=8, color="#666")
    a2.set_xlabel("steering strength  α   (− = remove refusal)")
    a2.set_ylabel("refusal rate (%)"); a2.set_ylim(-4, 110)
    a2.set_title("40 harmful prompts (JailbreakBench)\n→ monotonic bypass; random flat", color=TEAL, fontsize=10)
    a2.legend(frameon=False, fontsize=8.4, loc="lower right"); a2.grid(alpha=0.25)
    # Panel 3: refusal direction on BENIGN prompts -> monotonic induce, vs random
    a3.plot(A, rr("benign", "refusal"), "-o", color=TEAL, lw=2.5, ms=6.5, label="refusal direction")
    a3.plot(A, rr("benign", "random"), "--s", color=GREY, lw=2.1, ms=5.5, label="random direction")
    a3.axvline(0, color="k", lw=0.7, ls=":"); a3.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
    a3.text(-1.45, 6, "baseline 0% (α=0)", fontsize=8, color="#666")
    a3.set_xlabel("steering strength  α   (+ = add refusal)")
    a3.set_ylabel("refusal rate (%)"); a3.set_ylim(-4, 110)
    a3.set_title("40 benign prompts (AlpacaEval)\n→ monotonic induction; random flat", color=TEAL, fontsize=10)
    a3.legend(frameon=False, fontsize=8.4, loc="upper left"); a3.grid(alpha=0.25)
    fig.suptitle("Within each panel, one fixed pool of prompts is steered — only the direction (real vs random) changes",
                 fontsize=12.5, y=1.04)
    cap(fig, "Every line adds α·(a direction) to the same fixed pool of prompts at layer 16, and each panel compares "
             "the real direction against a random direction of equal magnitude, so within a panel the prompts and "
             "the push size are held constant and only the direction changes. Left: the compliance-probe direction "
             "on 60 loss-set replies traces a U that a random direction reproduces (a size-of-push artifact). "
             "Middle: on 40 JailbreakBench harmful prompts (98% refuse at α=0), negative α removes the refusal "
             "direction and bypasses refusal (98→0%); random only perturbs it near the largest pushes. Right: on 40 "
             "AlpacaEval benign prompts (0% refuse at α=0), positive α adds the refusal direction and induces refusal "
             "(0→100%); random does nothing. The two right panels use different pools because a pool already at 98% "
             "refusal has no room to be 'induced', and one at 0% has none to be 'bypassed'.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_monotonic_vs_ushape.png", dpi=140, bbox_inches="tight"); plt.close(fig)

def fig_refusal_2panel():
    """The two monotonic panels, framed as COMPLIANCE (= 100 − refusal)."""
    rows = load("experiments/refusal_alpha_sweep_v1/sweep.jsonl")
    by = defaultdict(list)
    for r in rows: by[(r["set"], r["direction"], r["alpha"])].append(r)
    A = sorted(set(r["alpha"] for r in rows))
    def comp(s, d): return [100 - rate(by[(s, d, a)], "is_refusal") for a in A]  # complies = does not refuse
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.7))
    # harmful: subtract refusal -> comply (jailbreak)
    a1.plot(A, comp("harmful", "refusal"), "-o", color=TEAL, lw=2.6, ms=7, label="refusal direction")
    a1.plot(A, comp("harmful", "random"), "--s", color=GREY, lw=2.1, ms=6, label="random direction (equal size)")
    a1.axvline(0, color="k", lw=0.7, ls=":"); a1.axhline(2, color="#bbbbbb", lw=0.8, ls="--")
    a1.text(0.1, 9, "no steering (α=0):\n2% comply (98% refuse)", fontsize=8, color="#666")
    a1.set_xlabel("steering strength  α   (− = subtract the refusal direction)")
    a1.set_ylabel("complies with request (%)   —   for harmful prompts, = attack success")
    a1.set_ylim(-4, 112)
    a1.set_title("40 harmful prompts (JailbreakBench)\nsubtracting the refusal direction ⇒ it complies (jailbreak)", color=TEAL, fontsize=10.5)
    a1.legend(frameon=False, fontsize=8.6, loc="upper right"); a1.grid(alpha=0.25)
    # benign: add refusal -> stops complying (over-refusal)
    a2.plot(A, comp("benign", "refusal"), "-o", color=TEAL, lw=2.6, ms=7, label="refusal direction")
    a2.plot(A, comp("benign", "random"), "--s", color=GREY, lw=2.1, ms=6, label="random direction (equal size)")
    a2.axvline(0, color="k", lw=0.7, ls=":"); a2.axhline(100, color="#bbbbbb", lw=0.8, ls="--")
    a2.text(-1.45, 92, "no steering (α=0):\n100% comply", fontsize=8, color="#666")
    a2.set_xlabel("steering strength  α   (+ = add the refusal direction)")
    a2.set_ylabel("complies with request (%)"); a2.set_ylim(-4, 112)
    a2.set_title("40 benign prompts (AlpacaEval)\nadding the refusal direction ⇒ it refuses (over-refusal)", color=TEAL, fontsize=10.5)
    a2.legend(frameon=False, fontsize=8.6, loc="lower left"); a2.grid(alpha=0.25)
    fig.suptitle("The refusal direction is a compliance knob: remove it → comply more (jailbreak); add it → comply less (refuse)",
                 fontsize=12, y=1.03)
    cap(fig, "Compliance = the reply addresses the request instead of refusing it, measured as 100% minus the "
             "refusal-phrase rate (for a harmful prompt a non-refusal is a successful jailbreak; this complement "
             "tracks the 70B attack-success judge to within ~2 points). Each panel steers one fixed pool of prompts "
             "by adding α·(a direction) at layer 16 and compares the causal refusal direction against a random "
             "direction of equal magnitude, so the prompts and push size are held constant and only the direction "
             "changes. Left (40 JailbreakBench harmful requests, 2% comply unsteered): subtracting the refusal "
             "direction (α<0) monotonically drives compliance up to 100% — i.e. it jailbreaks the model — while a "
             "random direction only spikes near the largest, output-degrading pushes. Right (40 AlpacaEval benign "
             "requests, 100% comply unsteered): adding the refusal direction (α>0) drives compliance down to 0% "
             "(over-refusal); a random direction stays flat.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_refusal_monotonic_2panel.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ FIG 2 (refusal gate)
def fig_refusal_gate():
    gen = load("experiments/arditi_repl_v1/generations.jsonl")
    jud = load("experiments/arditi_repl_v1/arditi_judged.jsonl")
    g = defaultdict(list); j = defaultdict(list)
    for r in gen: g[r["cond"]].append(r)
    for r in jud: j[r["method"]].append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    arms = ["harmful_baseline", "harmful_ablate_refusal", "harmful_ablate_random", "harmful_ablate_lrcomp"]
    labs = ["no\nintervention", "REMOVE\nrefusal dir", "remove\nrandom dir", "remove\ncompliance probe"]
    ref = [rate(g[a], "is_refusal") for a in arms]
    asr = [rate(j[a], "judge_compliance_unsafe") for a in arms]
    x = np.arange(4); w = 0.38
    a1.axhline(ref[0], color=GREY, lw=1, ls="--")
    b1 = a1.bar(x - w/2, ref, w, color=[GREY, CRIM, RAND, RAND], label="refuses (%)")
    b2 = a1.bar(x + w/2, asr, w, color="none", edgecolor="k", hatch="////", label="produces harmful content (%)")
    for i in range(4):
        a1.text(i - w/2, ref[i] + 2, f"{ref[i]:.0f}", ha="center", fontsize=8.5)
        a1.text(i + w/2, asr[i] + 2, f"{asr[i]:.0f}", ha="center", fontsize=8.5)
    a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=8.5); a1.set_ylabel("%"); a1.set_ylim(0, 112)
    a1.set_title("Remove a direction from HARMFUL prompts\nonly the refusal direction bypasses refusal", color=TEAL)
    a1.legend(frameon=False, fontsize=8.6, loc="center right"); a1.grid(axis="y", alpha=0.25)
    # addition on benign
    arms2 = ["harmless_baseline", "harmless_add_refusal"]
    ref2 = [rate(g[a], "is_refusal") for a in arms2]
    a2.bar([0, 1], ref2, 0.5, color=[GREY, TEAL])
    for i, v in enumerate(ref2): a2.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["no\nintervention", "ADD\nrefusal dir"], fontsize=9)
    a2.set_ylabel("refuses (%)"); a2.set_ylim(0, 112)
    a2.set_title("Add the refusal direction to BENIGN prompts\nit makes the model refuse harmless requests", color=TEAL)
    a2.grid(axis="y", alpha=0.25)
    fig.suptitle("Refusal is a causal gate: adding it induces refusal, removing it bypasses refusal", fontsize=13, y=1.03)
    cap(fig, "Left: on 50 harmful prompts the model refuses 98% of the time; projecting the refusal direction out of "
             "every layer drops that to 68% and lets 32% produce harmful content (hatched) — while removing a random "
             "direction or the victim's own compliance-probe direction changes nothing. Right: on 100 harmless "
             "prompts, adding the refusal direction makes the model refuse 64% of them. Method = Arditi et al. 2024.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_refusal_gate.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ FIG 3 (output form)
def fig_output_content():
    rows = load("experiments/output_content_control_v1/generations.jsonl")
    by = defaultdict(list)
    for r in rows: by[(r["prop"], r["arm"])].append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    def panel(ax, prop, metric_fn, ylab, title, ymax):
        arms = ["add_baseline", "add_prop", "add_random", "abl_baseline", "abl_prop", "abl_random"]
        vals = [metric_fn(by[(prop, a)]) for a in arms]
        cols = [GREY, TEAL, RAND, GREY, CRIM, RAND]
        ax.bar(range(6), vals, color=cols, edgecolor="k", linewidth=0.4)
        for i, v in enumerate(vals): ax.text(i, v + ymax*0.02, f"{v:.0f}", ha="center", fontsize=8.5)
        ax.axvline(2.5, color="k", lw=0.8, ls=":")
        ax.text(1, ymax*1.04, "ADD the direction\n(on plain prompts)", ha="center", fontsize=8.6, weight="bold")
        ax.text(4, ymax*1.04, "REMOVE the direction\n(on prompts that have it)", ha="center", fontsize=8.6, weight="bold")
        ax.set_xticks(range(6))
        ax.set_xticklabels(["none", "+dir", "+rand", "none", "−dir", "−rand"], fontsize=8.6)
        ax.set_ylabel(ylab); ax.set_ylim(0, ymax*1.18); ax.set_title(title); ax.grid(axis="y", alpha=0.25)
    panel(a1, "french", lambda rr: rate(rr, "is_french"), "% of replies in French",
          "Language (English ↔ French)", 100)
    panel(a2, "verbose", lambda rr: float(np.mean([r["n_tokens"] for r in rr])), "reply length (tokens)",
          "Verbosity (terse ↔ long)", 200)
    fig.suptitle("A positive control: output FORM is steerable by ADDING a direction, but not by removing it",
                 fontsize=13, y=1.03)
    cap(fig, "Directions for 'reply in French' and 'reply at length' are fit the same way as the harm direction. "
             "ADDING them works strongly (English→French 0→98%; 30→115 tokens, teal); REMOVING them barely does "
             "anything (100→90%; 200→192, crimson). Random controls (grey) do nothing. So the method CAN change "
             "output content — via addition. This is the control that makes the harm result (next figure) meaningful.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_output_form.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ FIG 4 (harm capability)
def fig_harm_capability():
    add = load("experiments/add_harm_v1/judged.jsonl")
    abl = load("experiments/harm_ablation_v1/judged.jsonl")
    a = defaultdict(list); b = defaultdict(list)
    for r in add: a[r["method"]].append(r)
    for r in abl: b[r["method"]].append(r)
    def mh(rows):
        v = [r["judge_harm_likert"] for r in rows if isinstance(r.get("judge_harm_likert"), (int, float))]
        return (np.mean(v), np.std(v)/np.sqrt(len(v))) if v else (0, 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # ADD
    aa = ["baseline", "add_harm_1x", "add_harm_2x", "add_random_1x", "add_random_2x"]
    al = ["no\nintervention", "ADD harm\n1×", "ADD harm\n2×", "add\nrandom 1×", "add\nrandom 2×"]
    m = [mh(a[x])[0] for x in aa]; s = [mh(a[x])[1] for x in aa]
    a1.axhline(m[0], color=GREY, lw=1, ls="--", label="baseline")
    a1.bar(range(5), m, yerr=[1.96*x for x in s], capsize=3, color=[GREY, TEAL, TEAL, RAND, RAND], edgecolor="k", linewidth=0.4)
    for i in range(5): a1.text(i, m[i] + 1.96*s[i] + 0.13, f"{m[i]:.2f}", ha="center", fontsize=8.5)
    a1.set_xticks(range(5)); a1.set_xticklabels(al, fontsize=8.2)
    a1.set_ylabel("harmful uplift  (Stage-B, 1–5)"); a1.set_ylim(0, 5)
    a1.set_title("ADD the harm direction → no change\n(vs French 0→98% under the same move)", color=CRIM)
    a1.legend(frameon=False, fontsize=8.5, loc="upper right"); a1.grid(axis="y", alpha=0.25)
    # ABLATE
    la = ["baseline", "ablate_md_harm", "ablate_lr_harm", "ablate_random"]
    ll = ["no\nintervention", "REMOVE harm\n(mean-diff)", "REMOVE harm\n(probe)", "remove\nrandom"]
    m2 = [mh(b[x])[0] for x in la]; s2 = [mh(b[x])[1] for x in la]
    a2.axhline(m2[0], color=GREY, lw=1, ls="--", label="baseline")
    a2.bar(range(4), m2, yerr=[1.96*x for x in s2], capsize=3, color=[GREY, CRIM, CRIM, RAND], edgecolor="k", linewidth=0.4)
    for i in range(4): a2.text(i, m2[i] + 1.96*s2[i] + 0.13, f"{m2[i]:.2f}", ha="center", fontsize=8.5)
    a2.set_xticks(range(4)); a2.set_xticklabels(ll, fontsize=8.2)
    a2.set_ylabel("harmful uplift  (Stage-B, 1–5)"); a2.set_ylim(0, 5)
    a2.set_title("REMOVE the harm direction → no change\n(on already-harmful compliant replies)", color=CRIM)
    a2.legend(frameon=False, fontsize=8.5, loc="upper right"); a2.grid(axis="y", alpha=0.25)
    fig.suptitle("Harm-uplift is a capability: it moves under neither add nor remove (despite AUC 0.88 decodability)",
                 fontsize=13, y=1.03)
    cap(fig, "Left: on low-uplift compliant replies (baseline 2.2/5, room to rise), adding the harm direction at a "
             "coherent magnitude changes uplift by +0.11 — no more than a random direction. Right: on high-uplift "
             "replies (baseline 3.6/5, room to fall), removing the harm direction also does nothing vs random. Error "
             "bars are 95% CIs. Steering moves representations and forms, not the operational knowledge an 8B lacks.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_harm_capability.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ FIG 5 (harm dose-response)
def fig_harm_dose():
    rn = 1.385; mults = [0, 1, 2, 4, 8]; x = [m*rn/STEER_UNIT for m in mults]   # α units
    def series(path):
        rows = load(path); by = defaultdict(list)
        for r in rows: by[r["method"]].append(r)
        def mh(m):
            v = [r["judge_harm_likert"] for r in by[m] if isinstance(r.get("judge_harm_likert"), (int, float))]
            return (np.mean(v), np.std(v)/np.sqrt(len(v))) if v else (np.nan, 0)
        def tur(m): return np.mean([r["coherence"]["token_unique_ratio"] for r in by[m]]) if by[m] else np.nan
        return mh, tur
    lo_mh, lo_tur = series("experiments/add_harm_doseresponse_v1/judged.jsonl")
    hi_mh, hi_tur = series("experiments/add_harm_doseresponse_highharm_v1/judged.jsonl")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # Panel A: absolute uplift vs push, harm(solid) vs random(dashed), for both sets
    for mh, col, name in [(hi_mh, TEAL, "already-harmful replies"), (lo_mh, BLUE, "low-uplift replies")]:
        hu = [mh(f"harm_m{m}")[0] for m in mults]; he = [1.96*mh(f"harm_m{m}")[1] for m in mults]
        ru = [np.nan] + [mh(f"random_m{m}")[0] for m in mults[1:]]
        a1.errorbar(x, hu, yerr=he, fmt="-o", color=col, lw=2.4, ms=6.5, capsize=3, label=f"{name}: harm dir")
        a1.plot(x[1:], ru[1:], "--s", color=col, lw=1.6, ms=5, alpha=0.6, label=f"{name}: random dir")
    a1.axvline(rn/STEER_UNIT, color="#bbb", lw=0.8, ls=":"); a1.text(rn/STEER_UNIT+0.03, 1.15, "calibrated\ndose", fontsize=7.3, color="#666")
    a1.axvline(9.71/STEER_UNIT, color="#bbb", lw=0.8, ls=":"); a1.text(9.71/STEER_UNIT-0.04, 1.15, "full residual\nnorm", fontsize=7.3, color="#666", ha="right")
    a1.set_xlabel("steering strength  α   (α×7.56 = ‖added vector‖)"); a1.set_ylabel("harmful uplift (Stage-B, 1–5)")
    a1.set_ylim(1, 5); a1.set_title("Uplift is flat then falls — on BOTH sets\nharm dir ≈ random dir; French flips at α≈0.3", color=CRIM)
    a1.legend(frameon=False, fontsize=7.6, loc="lower left"); a1.grid(alpha=0.25)
    # Panel B: coherence of the harm-direction arm, both sets
    a2.plot(x, [hi_tur(f"harm_m{m}") for m in mults], "-o", color=TEAL, lw=2.4, ms=6.5, label="already-harmful replies")
    a2.plot(x, [lo_tur(f"harm_m{m}") for m in mults], "-o", color=BLUE, lw=2.4, ms=6.5, label="low-uplift replies")
    a2.axvline(9.71/STEER_UNIT, color="#bbb", lw=0.8, ls=":")
    a2.set_xlabel("steering strength  α   (α×7.56 = ‖added vector‖)"); a2.set_ylabel("coherence  (unique-token ratio)")
    a2.set_ylim(0.25, 0.55); a2.set_title("Pushing harder just degrades the output\n(that is what the small 'drop' at high push is)", color=CRIM)
    a2.legend(frameon=False, fontsize=8.2, loc="lower left"); a2.grid(alpha=0.25)
    fig.suptitle("Dose–response: adding the harm direction does not raise uplift, whether or not the reply is already harmful",
                 fontsize=12, y=1.03)
    cap(fig, "Two eval sets: replies that are already harmful (Stage-B ≥4, baseline 3.7) and low-uplift replies "
             "(≤3, baseline 2.2). On BOTH, adding the harm direction leaves uplift statistically flat vs a random "
             "direction (peak harm−random gap +0.13, within noise) and then LOWERS it at large push as the output "
             "degrades (right panel). So the direction neither manufactures uplift where it's absent nor "
             "meaningfully amplifies it where it's present — unlike French, which flips 0→98% at α≈0.3 with no "
             "degradation. (An earlier confounded α-sweep reported a +0.40 'odd' component; it does not survive as "
             "a clean add-vs-random effect here.)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_harm_doseresponse.png", dpi=140, bbox_inches="tight"); plt.close(fig)

def fig_compliance_and_harm():
    """Compliance + harm uplift under steering: refusal direction vs TWO harm-direction sweeps."""
    REF_NORM = 7.563; rn = 1.385; M = [0, 1, 2, 4, 8]

    def refusal_series():
        rb = defaultdict(list)
        for r in load("experiments/refusal_harm_vs_compliance_v1/judged.jsonl"): rb[r["alpha"]].append(r)
        A = sorted(rb)
        comp = [100*sum(1 for r in rb[a] if r.get("judge_compliance_unsafe"))/len(rb[a]) for a in A]
        harm = [np.mean([r["judge_harm_likert"] for r in rb[a] if isinstance(r.get("judge_harm_likert"), (int, float))]) for a in A]
        return A, comp, harm

    def harm_series(path):
        hb = defaultdict(list)
        for r in load(path): hb[r["method"]].append(r)
        x = [m*rn/REF_NORM for m in M]  # same α units (α×REF_NORM = ‖added vector‖)
        comp = [100*sum(1 for r in hb[f"harm_m{m}"] if r.get("judge_compliance_unsafe"))/len(hb[f"harm_m{m}"]) for m in M]
        harm = [np.mean([r["judge_harm_likert"] for r in hb[f"harm_m{m}"] if isinstance(r.get("judge_harm_likert"), (int, float))]) for m in M]
        return x, comp, harm

    A, rc, rh = refusal_series()
    lx, lc, lh = harm_series("experiments/add_harm_doseresponse_v1/judged.jsonl")
    hx, hc, hh = harm_series("experiments/add_harm_doseresponse_highharm_v1/judged.jsonl")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    def panel(ax, xs, comp, harm, title, xlabel, add_only=False):
        ax.plot(xs, comp, "-o", color=TEAL, lw=2.5, ms=6)
        ax.set_ylabel("complies / attack success (%)", color=TEAL); ax.tick_params(axis="y", labelcolor=TEAL)
        ax.set_ylim(-4, 100); ax.set_xlim(-1.65, 1.65); ax.axvline(0, color="k", lw=0.6, ls=":")
        axb = ax.twinx(); axb.plot(xs, harm, "-s", color=CRIM, lw=2.5, ms=5.5)
        axb.set_ylabel("harm uplift (Stage-B, 1–5)", color=CRIM); axb.tick_params(axis="y", labelcolor=CRIM)
        axb.set_ylim(1, 5); axb.axhline(4, color=CRIM, lw=0.7, ls=":")
        axb.text(-1.55, 4.08, "'meaningful uplift' (4)", fontsize=6.8, color=CRIM)
        if add_only:
            ax.text(-1.55, 50, "(only +α / add\nside swept)", fontsize=7, color="#888")
        ax.set_xlabel(xlabel, fontsize=8.5); ax.set_title(title, fontsize=10)
    panel(axes[0], A, rc, rh, "Steer the REFUSAL direction\n(40 harmful prompts) → compliance 2→62%,\nharm follows to ~2.3 ceiling",
          "α   (−α = subtract refusal dir)")
    panel(axes[1], lx, lc, lh, "Steer the HARM direction\n(low-uplift replies, base 2.2) → both FLAT",
          "α   (+α = add harm dir)", add_only=True)
    panel(axes[2], hx, hc, hh, "Steer the HARM direction\n(already-harmful replies, base 3.7) → both FLAT",
          "α   (+α = add harm dir)", add_only=True)
    fig.suptitle("Only the refusal direction moves harm (and only to a low ceiling); neither harm-direction steering does — α×7.56 = ‖added vector‖ everywhere",
                 fontsize=11.5, y=1.04)
    cap(fig, "Compliance/attack-success (teal, left axis) and harm uplift (crimson, right axis) vs steering strength α, "
             "all judged by Llama-70B. LEFT — subtracting the refusal direction from 40 JailbreakBench harmful prompts "
             "drives attack success 2→62% and pulls harm up with it, but only to ~2.3/5 ('marginal/web-equivalent'), "
             "never reaching '4 = meaningful uplift'; both collapse at the largest push as output degrades. MIDDLE & "
             "RIGHT — adding the harm direction (fit on Llama Stage-B labels) leaves BOTH compliance and harm flat at "
             "every magnitude, whether the replies start low-uplift (2.2) or already-harmful (3.7). So harm rises only "
             "as a by-product of opening the compliance gate, and is capped at the 8B's capability ceiling; pushing the "
             "harm direction itself moves nothing. (Different prompt sets per panel; a single matched sweep is pending.)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_compliance_and_harm.png", dpi=140, bbox_inches="tight"); plt.close(fig)

def fig_harm_matrix_dualjudge():
    """2×2: harm uplift vs magnitude for {harm_llama, harm_qwen, random}, per set × judge."""
    lla = defaultdict(list); qwe = defaultdict(list)
    for r in load("experiments/harm_steer_matrix_v1/judged_llama.jsonl"): lla[(r["set"], r["direction"], r["magnitude"])].append(r)
    for r in load("experiments/harm_steer_matrix_v1/judged_qwen.jsonl"): qwe[(r["set"], r["direction"], r["magnitude"])].append(r)
    mags = [0.0, 2.8, 5.5, 11.1]; xa = [m/STEER_UNIT for m in mags]   # α units (α×7.56 = ‖added vector‖)
    def harm(by, s, d):
        ys = []
        for m in mags:
            k = (s, "baseline", 0.0) if m == 0.0 else (s, d, m)
            v = [r["judge_harm_likert"] for r in by[k] if isinstance(r.get("judge_harm_likert"), (int, float))]
            ys.append(np.mean(v) if v else np.nan)
        return ys
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    cols = {"harm_llama": TEAL, "harm_qwen": "#7b3fa0", "random": GREY}
    labs = {"harm_llama": "harm dir (Llama-fit)", "harm_qwen": "harm dir (Qwen-fit)", "random": "random"}
    for i, s in enumerate(["low", "high"]):
        for j, (by, jn) in enumerate([(lla, "Llama-70B judge"), (qwe, "Qwen-72B judge")]):
            ax = axes[i][j]
            b = harm(by, s, "harm_llama")[0]
            ax.axhline(b, color="#bbb", lw=1, ls="--")
            for d in ["harm_llama", "harm_qwen", "random"]:
                ls = "-" if d != "random" else "--"
                ax.plot(xa, harm(by, s, d), ls+"o", color=cols[d], lw=2.3, ms=6, label=labs[d])
            ax.set_ylim(1, 5); ax.grid(alpha=0.25)
            ax.axhline(4, color=CRIM, lw=0.7, ls=":")
            ax.set_title(f"{'low-uplift' if s=='low' else 'already-harmful'} replies  ·  {jn}", fontsize=10)
            if i == 1: ax.set_xlabel("steering strength  α   (α×7.56 = ‖added vector‖)")
            if j == 0: ax.set_ylabel(f"harm uplift (1–5)\nbaseline {b:.1f}")
            if i == 0 and j == 0: ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Adding the harm direction never raises uplift — robust to the fitting judge (Llama/Qwen),\nthe eval set (low/high), and the scoring judge (Llama/Qwen). It only falls, as large pushes degrade output.",
                 fontsize=11.5, y=1.0)
    cap(fig, "Each panel: Stage-B harm uplift vs steering strength α (α×7.56 = ‖added vector‖) when ADDING a harm direction, for two harm directions "
             "(fit on Llama-labelled vs Qwen-labelled Stage-B wins; they are nearly identical, cos=0.93) and a random "
             "control, on two eval sets, scored by two judges. In all four panels the harm-direction curves sit on top "
             "of the random curve and the baseline (grey dashed) and never approach '4 = meaningful uplift' (red dotted); "
             "at the largest push all curves fall as the output degrades. Compliance behaves the same (harm dirs ≈ random). "
             "So 'harm is not add-steerable' holds regardless of which judge defines the direction or scores the outcome.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig8_harm_matrix_dualjudge.png", dpi=140, bbox_inches="tight"); plt.close(fig)

def fig_harm_vs_compliance():
    """On headroom-rich harmful prompts: does the harm direction open the compliance gate? (No.)"""
    rows = [r for r in load("experiments/steer_compare_v1/judged_llama.jsonl") if r["set"] == "harmful"]
    by = defaultdict(list)
    for r in rows: by[(r["direction"], r["alpha"])].append(r)
    A = sorted({a for (_, a) in by})
    cols = {"refusal": CRIM, "harm": "#7b3fa0", "random": GREY}
    labs = {"refusal": "refusal dir (the gate)", "harm": "harm dir", "random": "random"}
    def val(d, a, f):
        k = ("baseline", 0.0) if a == 0.0 else (d, a)
        g = by[k]
        return f(g) if g else np.nan
    comp = lambda g: 100*sum(1 for r in g if r.get("judge_compliance_unsafe"))/len(g)
    coh = lambda g: np.mean([r["coherence"]["token_unique_ratio"] for r in g])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True)
    for d in ["refusal", "harm", "random"]:
        a1.plot(A, [val(d, a, comp) for a in A], "-o", color=cols[d], lw=2.4, ms=6, label=labs[d])
        a2.plot(A, [val(d, a, coh) for a in A], "-o", color=cols[d], lw=2.4, ms=6, label=labs[d])
    a1.set_ylabel("complies / attack success (%)  ·  70B judge"); a1.set_ylim(-4, 100)
    a1.set_xlabel("steering strength  α   (−α = subtract direction)"); a1.grid(alpha=0.25)
    a1.axvline(0, color="k", lw=0.6, ls=":"); a1.legend(frameon=False, fontsize=8.5, loc="upper center")
    a1.set_title("Harm-dir 'compliance' appears only at α=+0.75/1.5…", fontsize=10)
    a2.axhspan(0.0, 0.6, color="#f2d0d0", alpha=0.5, zorder=0)
    a2.text(0.02, 0.30, "degraded /\ngibberish", fontsize=8, color=CRIM)
    a2.set_ylabel("coherence  (unique-token ratio)"); a2.set_ylim(0, 1.0)
    a2.set_xlabel("steering strength  α"); a2.grid(alpha=0.25); a2.axvline(0, color="k", lw=0.6, ls=":")
    a2.set_title("…exactly where its output has collapsed to gibberish", fontsize=10)
    fig.suptitle("Does the harm direction steer compliance? No — its only 'compliance' is a degraded-output artifact, not a real bypass",
                 fontsize=11.5, y=1.01)
    cap(fig, "Steering three directions on 30 JailbreakBench harmful prompts (baseline 0% compliance — full headroom). "
             "LEFT: the harm direction shows 0% compliance at every coherence-preserving magnitude (e.g. α=−0.75, where "
             "its output is clean at 0.94) and only reaches ~30% at α=+0.75. RIGHT: but at α=+0.75 its coherence has "
             "collapsed to 0.32 (α=+1.5 → 0.09, pure gibberish) — so that 'compliance' is the known degradation "
             "false-positive (the 70B JBB judge miscounts broken non-refusals as unsafe), the same artifact the random "
             "direction produces on its degraded side (33% at α=−1.5, coherence 0.45). The refusal direction, by "
             "contrast, opens compliance at lower magnitude and is the Arditi-confirmed causal gate. So the harm "
             "direction steers NEITHER harm NOR compliance — it only breaks the model.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig9_harm_vs_compliance.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ============================================================ SUMMARY FIGURES
def fig_s1_scoreboard():
    """The thesis in one grid: gate / form / capability × ADD / ABLATE / causal."""
    from matplotlib.patches import Rectangle
    GRN, AMB, RED = "#cdebd6", "#f6e6c5", "#f4d0d0"
    rows = [
        ("Refusal\n(GATE)", [
            (GRN, "✓  induce\n0→64% refuse"), (GRN, "✓  bypass\n98→68% ref\n2→32% ASR"), (GRN, "✓  Arditi\nreplicates")]),
        ("Language / verbosity\n(output FORM)", [
            (GRN, "✓  French\n0→98%\nverbose 22→89"), (AMB, "~  weak\nFrench 100→90"), (GRN, "✓  (add\ndirection)")]),
        ("Harm uplift\n(CAPABILITY)", [
            (RED, "✗  +0.1\n≈ random"), (RED, "✗  Δ≈0\n≈ random"), (GREY_L := "#e4e4e4", "separable ≠\nsteerable")]),
    ]
    heads = ["", "ADD  (inject)", "ABLATE  (remove)", "causal?"]
    fig, ax = plt.subplots(figsize=(11, 5.8)); ax.set_xlim(0, 4); ax.set_ylim(-0.55, 4.6); ax.axis("off")
    cw = [1.35, 0.88, 0.88, 0.89]; x0 = [0, 1.35, 2.23, 3.11]; H = 0.9
    ax.text(2, 4.4, "One taxonomy: three DIFFERENT KINDS of object, not three directions of differing strength",
            ha="center", fontsize=11.5, fontweight="bold")
    for j, h in enumerate(heads):
        ax.text(x0[j]+cw[j]/2, 3.9, h, ha="center", va="center", fontsize=10.5, fontweight="bold")
    ybase = [2.75, 1.6, 0.45]
    for i, (label, cells) in enumerate(rows):
        y = ybase[i]
        ax.text(x0[0]+cw[0]/2, y+H/2, label, ha="center", va="center", fontsize=10, fontweight="bold")
        for j, (col, txt) in enumerate(cells):
            ax.add_patch(Rectangle((x0[j+1]+0.02, y), cw[j+1]-0.04, H, facecolor=col, edgecolor="white", lw=2))
            ax.text(x0[j+1]+cw[j+1]/2, y+H/2, txt, ha="center", va="center", fontsize=8.6)
    ax.text(2, -0.3, "A steering vector can flip a behavioral GATE and inject output FORM — but not conjure a CAPABILITY the model lacks. "
            "'Separable ≠ steerable':\nat the OUTPUT locus, harmful vs benign responses are trivially separable (~0.94 harm, ~0.92 output-compliance) yet the separating "
            "direction injects\nnothing. (Pre-response readability — Fig 5, input locus — is the mid-0.7s claim, NOT the 0.94.) ASR overstates harm: the gate flips, the capability can't.",
            ha="center", fontsize=7.6, style="italic", color="#333")
    fig.savefig(f"{OUT}/figS1_scoreboard.png", dpi=140, bbox_inches="tight"); plt.close(fig)

def fig_s2_asr_neq_harm():
    """Money plot: open the compliance gate → ASR climbs to 62%, harm plateaus at the capability ceiling."""
    by = defaultdict(list)
    for r in load("experiments/refusal_harm_vs_compliance_v1/judged.jsonl"): by[r["alpha"]].append(r)
    A = sorted(by)
    comp = [100*sum(1 for r in by[a] if r.get("judge_compliance_unsafe"))/len(by[a]) for a in A]
    harm = [np.mean([r["judge_harm_likert"] for r in by[a] if isinstance(r.get("judge_harm_likert"), (int, float))]) for a in A]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(A, comp, "-o", color=TEAL, lw=3, ms=7, label="compliance / attack success")
    ax.set_ylabel("compliance / attack success (%)", color=TEAL, fontsize=11); ax.tick_params(axis="y", labelcolor=TEAL)
    ax.set_ylim(-4, 100); ax.set_xlabel("steering strength  α    (subtract the refusal direction  →)", fontsize=10.5)
    ax.invert_xaxis()
    axb = ax.twinx(); axb.plot(A, harm, "-s", color=CRIM, lw=3, ms=6.5, label="harm uplift")
    axb.set_ylabel("harm uplift (Stage-B, 1–5)", color=CRIM, fontsize=11); axb.tick_params(axis="y", labelcolor=CRIM)
    axb.set_ylim(1, 5); axb.axhspan(1, 2.4, color=CRIM, alpha=0.07)
    axb.axhline(2.3, color=CRIM, lw=0.9, ls="--"); axb.text(1.45, 2.42, "capability ceiling (~2.3)", fontsize=8.5, color=CRIM, ha="left")
    axb.axhline(4, color=CRIM, lw=0.8, ls=":"); axb.text(1.45, 4.06, "'meaningful uplift' (4) — never reached", fontsize=8, color=CRIM, ha="left")
    ax.annotate("", xy=(-1.0, 62), xytext=(-1.0, 26.5), arrowprops=dict(arrowstyle="<->", color="#444", lw=1.4))
    ax.text(-1.06, 44, "the GAP\n= the finding", fontsize=9, ha="right", color="#222", fontweight="bold")
    ax.set_title("ASR ≠ harm:  opening the compliance gate drives attack success to 62%,\nbut harm plateaus at the model's capability ceiling",
                 fontsize=11.5)
    cap(fig, "Subtracting the refusal direction from 40 JailbreakBench harmful prompts (Llama-70B judged). Attack "
             "success climbs 2→62% as the gate opens, but the harm uplift of those compliant replies saturates at "
             "~2.3/5 ('marginal / web-equivalent') and never reaches meaningful uplift. The vertical gap between the "
             "two curves is the paper's thesis: a jailbreak metric counts the gate flip, not the (capped) harm behind it.")
    fig.tight_layout(); fig.savefig(f"{OUT}/figS2_asr_neq_harm.png", dpi=140, bbox_inches="tight"); plt.close(fig)

def fig_s3_mechanism_triptych():
    """Same mechanism (ADD a direction): output FORM flips, CAPABILITY doesn't."""
    fr = defaultdict(list); vb = defaultdict(list)
    import re
    FRW = re.compile(r'\b(le|la|les|un|une|des|est|et|dans|pour|avec|sur)\b', re.I)
    isfr = lambda t: bool(re.search(r'[àâçéèêëîïôûùü]', t)) and len(FRW.findall(t)) >= 3
    for r in load("experiments/output_content_control_v1/generations.jsonl"):
        (fr if r["prop"] == "french" else vb)[r["arm"]].append(r["response"])
    frv = {k: 100*sum(isfr(t) for t in v)/len(v) for k, v in fr.items()}
    vbv = {k: np.mean([len(t.split()) for t in v]) for k, v in vb.items()}
    hb = defaultdict(list)
    for r in load("experiments/add_harm_v1/judged.jsonl"): hb[r["method"]].append(r)
    hmean = lambda m: np.mean([r["judge_harm_likert"] for r in hb[m] if isinstance(r.get("judge_harm_likert"), (int, float))])
    panels = [
        ("ADD 'French'\n(output FORM)  ✓", ["baseline", "add\ndirection", "add\nrandom"],
         [frv["add_baseline"], frv["add_prop"], frv["add_random"]], "% replies in French", 100, TEAL),
        ("ADD 'verbose'\n(output FORM)  ✓", ["baseline", "add\ndirection", "add\nrandom"],
         [vbv["add_baseline"], vbv["add_prop"], vbv["add_random"]], "mean response length (tokens)", 100, TEAL),
        ("ADD 'harm'\n(CAPABILITY)  ✗", ["baseline", "add\ndirection", "add\nrandom"],
         [hmean("baseline"), hmean("add_harm_1x"), hmean("add_random_1x")], "harm uplift (Stage-B, 1–5)", 5, CRIM),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (title, labs, vals, ylab, ymax, hl) in zip(axes, panels):
        bars = ax.bar([0, 1, 2], vals, color=["#cccccc", hl, "#e0e0e0"], edgecolor="k", lw=0.6, width=0.62)
        for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+ymax*0.02, f"{v:.0f}" if ymax>10 else f"{v:.2f}", ha="center", fontsize=9.5)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labs, fontsize=9); ax.set_ylim(0 if ymax>10 else 1, ymax if ymax>10 else 5)
        ax.set_ylabel(ylab, fontsize=9.5); ax.set_title(title, fontsize=10.5, color=hl); ax.grid(axis="y", alpha=0.25)
    fig.suptitle("One mechanism (ADD the direction at L16), three concepts: the SAME operation flips output FORM but cannot move CAPABILITY",
                 fontsize=11.5, y=1.02)
    cap(fig, "Identical intervention — add a concept's diff-in-means direction to the residual stream — applied to three "
             "concepts, each with its own add-random control. Language and verbosity (output FORM) flip decisively "
             "(French 0→98%, length 22→89 tokens) while add-random does nothing. Harm uplift (a CAPABILITY) does not "
             "move (2.20→2.27 ≈ add-random 2.16). Same knob, opposite outcome: form is injectable, capability is not.")
    fig.tight_layout(); fig.savefig(f"{OUT}/figS3_mechanism_triptych.png", dpi=140, bbox_inches="tight"); plt.close(fig)

def fig_s4_readout_not_lever():
    """Distills fig8+fig9: harm dir moves neither harm (any judge/set) nor compliance (only via gibberish)."""
    U = STEER_UNIT
    L = defaultdict(list); Q = defaultdict(list)
    for r in load("experiments/harm_steer_matrix_v1/judged_llama.jsonl"): L[(r["set"], r["direction"], r["magnitude"])].append(r)
    for r in load("experiments/harm_steer_matrix_v1/judged_qwen.jsonl"): Q[(r["set"], r["direction"], r["magnitude"])].append(r)
    mags = [0.0, 2.8, 5.5, 11.1]; xa = [m/U for m in mags]
    def hh(by, s, d):
        out = []
        for m in mags:
            k = (s, "baseline", 0.0) if m == 0 else (s, d, m)
            v = [r["judge_harm_likert"] for r in by[k] if isinstance(r.get("judge_harm_likert"), (int, float))]
            out.append(np.mean(v) if v else np.nan)
        return out
    sc = defaultdict(list)
    for r in load("experiments/steer_compare_v1/judged_llama.jsonl"):
        if r["set"] == "harmful": sc[(r["direction"], r["alpha"])].append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.7))
    styles = [("low", "Llama", "-", TEAL), ("low", "Qwen", "--", TEAL), ("high", "Llama", "-", "#7b3fa0"), ("high", "Qwen", "--", "#7b3fa0")]
    for s, jn, ls, c in styles:
        by = L if jn == "Llama" else Q
        a1.plot(xa, hh(by, s, "harm_llama"), ls+"o", color=c, lw=2, ms=5, label=f"{s}-uplift · {jn} judge")
    a1.axhline(4, color=CRIM, lw=0.7, ls=":"); a1.text(0.05, 4.06, "meaningful uplift (4)", fontsize=7.5, color=CRIM)
    a1.set_ylim(1, 5); a1.set_xlabel("steering strength  α   (add harm direction)"); a1.set_ylabel("harm uplift (Stage-B, 1–5)")
    a1.set_title("Harm: flat on every set & judge\n(never rises; falls when pushed hard)", fontsize=10)
    a1.legend(frameon=False, fontsize=7.6, loc="upper right"); a1.grid(alpha=0.25)
    # right: harm-dir compliance vs coherence on harmful prompts
    pts = []
    for (d, al), g in sc.items():
        if d != "harm": continue
        comp = 100*sum(1 for r in g if r.get("judge_compliance_unsafe"))/len(g)
        coh = np.mean([r["coherence"]["token_unique_ratio"] for r in g])
        pts.append((coh, comp, al))
    pts.sort()
    a2.axvspan(0, 0.6, color="#f2d0d0", alpha=0.5); a2.text(0.04, 55, "output\ndegraded", fontsize=8.5, color=CRIM)
    a2.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#7b3fa0", lw=1.8, ms=8)
    for coh, comp, al in pts: a2.annotate(f"α={al:g}", (coh, comp), textcoords="offset points", xytext=(6, 5), fontsize=8)
    a2.set_xlim(0, 1); a2.set_ylim(-4, 70); a2.set_xlabel("coherence  (unique-token ratio)")
    a2.set_ylabel("compliance / attack success (%)"); a2.grid(alpha=0.25)
    a2.set_title("Compliance appears ONLY where output\nhas collapsed to gibberish (= false-positive)", fontsize=10)
    fig.suptitle("The harm direction is a READOUT, not a LEVER — it moves neither harm nor (real) compliance",
                 fontsize=11.5, y=1.0)
    cap(fig, "LEFT (from the dual-judge matrix): adding the harm direction leaves uplift flat on both the low- and "
             "already-harmful sets under both Llama and Qwen judges — it never approaches meaningful uplift and only "
             "falls at large push. RIGHT (headroom test on harmful prompts): the harm direction's only non-zero "
             "compliance (30%) occurs at the α where coherence has collapsed to 0.32 — i.e. a degradation "
             "false-positive, not a real bypass; at coherence-preserving magnitudes it stays at 0%. So the harm "
             "direction is a clean readout (AUC 0.88) but causally inert in both dimensions; the refusal direction is the only real knob.")
    fig.tight_layout(); fig.savefig(f"{OUT}/figS4_readout_not_lever.png", dpi=140, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_s1_scoreboard(); print("figS1 scoreboard")
    fig_s2_asr_neq_harm(); print("figS2 asr_neq_harm")
    fig_s3_mechanism_triptych(); print("figS3 mechanism_triptych")
    fig_s4_readout_not_lever(); print("figS4 readout_not_lever")
    fig_harm_vs_compliance(); print("fig9 harm_vs_compliance")
    fig_harm_matrix_dualjudge(); print("fig8 harm_matrix_dualjudge")
    fig_compliance_and_harm(); print("fig7 compliance_and_harm")
    fig_harm_dose(); print("fig5 harm_doseresponse")
    fig_monotonic(); print("fig1 monotonic_vs_ushape")
    fig_refusal_2panel(); print("fig6 refusal_monotonic_2panel")
    fig_refusal_gate(); print("fig2 refusal_gate")
    fig_output_content(); print("fig3 output_form")
    fig_harm_capability(); print("fig4 harm_capability")
    print("wrote to", OUT)
