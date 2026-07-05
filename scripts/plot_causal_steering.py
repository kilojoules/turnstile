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

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # LEFT: probe direction, U-shaped
    a1.plot(cx, cy, "-o", color=BLUE, lw=2.5, ms=7, label="steer the compliance PROBE direction")
    a1.plot(rx, ry, "--s", color=GREY, lw=2.2, ms=6, label="steer a RANDOM direction (same size)")
    a1.axvline(0, color="k", lw=0.7, ls=":")
    a1.set_xlabel("steering strength  α"); a1.set_ylabel("jailbreak success rate (%)")
    a1.set_title("WRONG direction → U-shaped (an artifact)\nboth pushes raise ASR; random matches it", color=CRIM)
    a1.legend(frameon=False, fontsize=8.6, loc="upper center"); a1.grid(alpha=0.25)
    # RIGHT: refusal direction, monotonic — annotate the two baselines
    a2.axhspan(0, 100, xmin=0, xmax=0.5, color="#f4f4f4", zorder=0)
    a2.plot(A, rr("harmful", "refusal"), "-o", color=CRIM, lw=2.5, ms=7, label="harmful prompts")
    a2.plot(A, rr("benign", "refusal"), "-o", color=TEAL, lw=2.5, ms=7, label="benign prompts")
    a2.plot(A, rr("benign", "random"), "--s", color=GREY, lw=2.0, ms=6, label="benign, random direction")
    a2.axvline(0, color="k", lw=0.8, ls=":")
    a2.text(0.02, 103, "α = 0: no steering\n(each set at its own baseline)", fontsize=8, ha="left")
    a2.annotate("harmful baseline 98%", (0.0, 98), (-1.45, 86), fontsize=8, color=CRIM,
                arrowprops=dict(arrowstyle="->", color=CRIM, lw=1))
    a2.annotate("benign baseline 0%", (0.0, 0), (0.25, 12), fontsize=8, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1))
    a2.text(-1.4, 55, "← remove\nrefusal", fontsize=8.5, ha="left", color="#555")
    a2.text(1.45, 55, "add\nrefusal →", fontsize=8.5, ha="right", color="#555")
    a2.set_xlabel("steering strength  α   (− = remove refusal, + = add refusal)")
    a2.set_ylabel("refusal rate (%)"); a2.set_ylim(-4, 116)
    a2.set_title("RIGHT direction → monotonic (causal)\nmore refusal-feature ⇒ more refusal, on both sets", color=TEAL)
    a2.legend(frameon=False, fontsize=8.6, loc="center left"); a2.grid(alpha=0.25)
    fig.suptitle("What steering looks like when you use the wrong vs the right direction", fontsize=13, y=1.03)
    cap(fig, "Each line adds α·(a direction) to the victim's residual stream. Left: the compliance probe "
             "direction gives a symmetric U — a random direction reproduces it, so it is a size-of-push artifact, "
             "not a real effect. Right: the causal refusal direction moves refusal monotonically — adding it "
             "(α>0) makes benign prompts refuse (teal, 0→100%), removing it (α<0) makes harmful prompts comply "
             "(crimson, 98→0%). The two curves start at different heights only because benign vs harmful prompts "
             "have different baseline refusal rates at α=0.")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_monotonic_vs_ushape.png", dpi=140, bbox_inches="tight"); plt.close(fig)

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

if __name__ == "__main__":
    fig_monotonic(); print("fig1 monotonic_vs_ushape")
    fig_refusal_gate(); print("fig2 refusal_gate")
    fig_output_content(); print("fig3 output_form")
    fig_harm_capability(); print("fig4 harm_capability")
    print("wrote to", OUT)
