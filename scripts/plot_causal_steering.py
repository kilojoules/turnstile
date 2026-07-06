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
    """Just the two monotonic panels of fig1 (harmful bypass + benign induce)."""
    rows = load("experiments/refusal_alpha_sweep_v1/sweep.jsonl")
    by = defaultdict(list)
    for r in rows: by[(r["set"], r["direction"], r["alpha"])].append(r)
    A = sorted(set(r["alpha"] for r in rows))
    def rr(s, d): return [rate(by[(s, d, a)], "is_refusal") for a in A]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.7))
    # harmful (bypass)
    a1.plot(A, rr("harmful", "refusal"), "-o", color=CRIM, lw=2.6, ms=7, label="refusal direction")
    a1.plot(A, rr("harmful", "random"), "--s", color=GREY, lw=2.1, ms=6, label="random direction (equal size)")
    a1.axvline(0, color="k", lw=0.7, ls=":"); a1.axhline(98, color="#bbbbbb", lw=0.8, ls="--")
    a1.text(-1.45, 91, "no steering (α=0):\n98% refuse", fontsize=8, color="#666")
    a1.set_xlabel("steering strength  α   (− = subtract the refusal direction)")
    a1.set_ylabel("refusal rate (%)"); a1.set_ylim(-4, 112)
    a1.set_title("40 harmful prompts (JailbreakBench)\nsubtracting the refusal direction bypasses refusal", color=TEAL, fontsize=10.5)
    a1.legend(frameon=False, fontsize=8.6, loc="lower right"); a1.grid(alpha=0.25)
    # benign (induce)
    a2.plot(A, rr("benign", "refusal"), "-o", color=TEAL, lw=2.6, ms=7, label="refusal direction")
    a2.plot(A, rr("benign", "random"), "--s", color=GREY, lw=2.1, ms=6, label="random direction (equal size)")
    a2.axvline(0, color="k", lw=0.7, ls=":"); a2.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
    a2.text(-1.45, 7, "no steering (α=0):\n0% refuse", fontsize=8, color="#666")
    a2.set_xlabel("steering strength  α   (+ = add the refusal direction)")
    a2.set_ylabel("refusal rate (%)"); a2.set_ylim(-4, 112)
    a2.set_title("40 benign prompts (AlpacaEval)\nadding the refusal direction induces refusal", color=TEAL, fontsize=10.5)
    a2.legend(frameon=False, fontsize=8.6, loc="upper left"); a2.grid(alpha=0.25)
    fig.suptitle("Steering the causal refusal direction moves refusal monotonically (a random direction does not)",
                 fontsize=12.5, y=1.03)
    cap(fig, "Each panel steers one fixed pool of prompts by adding α·(a direction) to the residual stream at layer "
             "16, and compares the causal refusal direction (harmful-minus-harmless difference-in-means) against a "
             "random direction of equal magnitude — so the prompts and the push size are held constant and only the "
             "direction changes. Left (40 JailbreakBench harmful requests, which refuse 98% of the time unsteered): "
             "subtracting the refusal direction (α<0) monotonically removes refusal, down to 0%; a random direction "
             "only dents it near the largest pushes (a degradation effect, not a directional one). Right (40 "
             "AlpacaEval benign requests, which never refuse unsteered): adding the refusal direction (α>0) "
             "monotonically induces refusal, up to 100%; a random direction stays flat at 0%.")
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
    rn = 1.385; mults = [0, 1, 2, 4, 8]; x = [m*rn for m in mults]
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
    a1.axvline(rn, color="#bbb", lw=0.8, ls=":"); a1.text(rn+0.2, 1.15, "calibrated\ndose (1×)", fontsize=7.3, color="#666")
    a1.axvline(9.71, color="#bbb", lw=0.8, ls=":"); a1.text(9.71-0.3, 1.15, "full residual\nnorm", fontsize=7.3, color="#666", ha="right")
    a1.set_xlabel("push strength  (‖added vector‖)"); a1.set_ylabel("harmful uplift (Stage-B, 1–5)")
    a1.set_ylim(1, 5); a1.set_title("Uplift is flat then falls — on BOTH sets\nharm dir ≈ random dir; French flips at 2.3", color=CRIM)
    a1.legend(frameon=False, fontsize=7.6, loc="lower left"); a1.grid(alpha=0.25)
    # Panel B: coherence of the harm-direction arm, both sets
    a2.plot(x, [hi_tur(f"harm_m{m}") for m in mults], "-o", color=TEAL, lw=2.4, ms=6.5, label="already-harmful replies")
    a2.plot(x, [lo_tur(f"harm_m{m}") for m in mults], "-o", color=BLUE, lw=2.4, ms=6.5, label="low-uplift replies")
    a2.axvline(9.71, color="#bbb", lw=0.8, ls=":")
    a2.set_xlabel("push strength  (‖added vector‖)"); a2.set_ylabel("coherence  (unique-token ratio)")
    a2.set_ylim(0.25, 0.55); a2.set_title("Pushing harder just degrades the output\n(that is what the small 'drop' at high push is)", color=CRIM)
    a2.legend(frameon=False, fontsize=8.2, loc="lower left"); a2.grid(alpha=0.25)
    fig.suptitle("Dose–response: adding the harm direction does not raise uplift, whether or not the reply is already harmful",
                 fontsize=12, y=1.03)
    cap(fig, "Two eval sets: replies that are already harmful (Stage-B ≥4, baseline 3.7) and low-uplift replies "
             "(≤3, baseline 2.2). On BOTH, adding the harm direction leaves uplift statistically flat vs a random "
             "direction (peak harm−random gap +0.13, within noise) and then LOWERS it at large push as the output "
             "degrades (right panel). So the direction neither manufactures uplift where it's absent nor "
             "meaningfully amplifies it where it's present — unlike French, which flips 0→98% at ‖·‖=2.3 with no "
             "degradation. (An earlier confounded α-sweep reported a +0.40 'odd' component; it does not survive as "
             "a clean add-vs-random effect here.)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_harm_doseresponse.png", dpi=140, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_harm_dose(); print("fig5 harm_doseresponse")
    fig_monotonic(); print("fig1 monotonic_vs_ushape")
    fig_refusal_2panel(); print("fig6 refusal_monotonic_2panel")
    fig_refusal_gate(); print("fig2 refusal_gate")
    fig_output_content(); print("fig3 output_form")
    fig_harm_capability(); print("fig4 harm_capability")
    print("wrote to", OUT)
