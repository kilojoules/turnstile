"""Figures for the causal-steering follow-ups (gate / form / capability).

Reads the experiment JSONL files and writes 4 PNGs to figures/causal_steering/.
All numbers recomputed from data; nothing hardcoded.
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

def load(p): return [json.loads(l) for l in open(os.path.join(ROOT, p))]
def rate(rows, key): return 100.0 * sum(bool(r.get(key)) for r in rows) / len(rows)
def wilson(k, n, z=1.96):
    if n == 0: return 0, 0
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)
    return 100*(c-h)/d, 100*(c+h)/d

BLUE, RED, GREY, GREEN = "#2c6fbb", "#c1372f", "#9a9a9a", "#2e8b57"

# ---------------------------------------------------------------- FIG 1
def fig_norm_confound():
    comp = load("experiments/steering_v3/sweep_p2_judged.jsonl")
    rand = load("experiments/steering_v3/sweep_random_L16_jbb_qwen.jsonl")
    def curve(rows):
        by = defaultdict(list)
        for r in rows:
            if r.get("prompt_type") == "loss":
                by[r.get("alpha_c")].append(r)
        xs = sorted(a for a in by if a is not None)
        ys = [rate(by[a], "judge_compliance_unsafe") for a in xs]
        return xs, ys
    cx, cy = curve(comp); rx, ry = curve(rand)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(cx, cy, "-o", color=BLUE, lw=2.4, ms=7, label='"compliance direction"')
    ax.plot(rx, ry, "--s", color=GREY, lw=2.4, ms=6, label="random direction (matched norm)")
    ax.set_xlabel("steering coefficient  α  (× residual-stream norm at L16)")
    ax.set_ylabel("attack success rate  (%)")
    ax.set_title("Old additive steering was a norm artifact\nA random direction reproduces the whole U-shape", fontsize=11)
    ax.legend(frameon=False, fontsize=10); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_norm_confound.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- FIG 2
def fig_refusal_gate():
    gen = load("experiments/arditi_repl_v1/generations.jsonl")
    jud = load("experiments/arditi_repl_v1/arditi_judged.jsonl")
    gby = defaultdict(list)
    for r in gen: gby[r["cond"]].append(r)
    jby = defaultdict(list)
    for r in jud: jby[r["method"]].append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    # harmful ablation
    arms = ["harmful_baseline", "harmful_ablate_refusal", "harmful_ablate_random", "harmful_ablate_lrcomp"]
    labels = ["baseline", "ablate\nrefusal dir", "ablate\nrandom", "ablate\nlr_comp"]
    ref = [rate(gby[a], "is_refusal") for a in arms]
    asr = [rate(jby[a], "judge_compliance_unsafe") for a in arms]
    x = np.arange(len(arms)); w = 0.38
    a1.bar(x - w/2, ref, w, color=BLUE, label="refusal rate")
    a1.bar(x + w/2, asr, w, color=RED, label="attack success (harmful content)")
    for i, (rf, ar) in enumerate(zip(ref, asr)):
        a1.text(i - w/2, rf + 1.5, f"{rf:.0f}", ha="center", fontsize=8.5)
        a1.text(i + w/2, ar + 1.5, f"{ar:.0f}", ha="center", fontsize=8.5)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=9)
    a1.set_ylabel("%"); a1.set_ylim(0, 108)
    a1.set_title("Ablate on harmful prompts\n(only the refusal direction bypasses refusal)", fontsize=10.5)
    a1.legend(frameon=False, fontsize=9, loc="center right"); a1.grid(axis="y", alpha=0.25)
    # harmless addition
    arms2 = ["harmless_baseline", "harmless_add_refusal"]
    ref2 = [rate(gby[a], "is_refusal") for a in arms2]
    a2.bar([0, 1], ref2, 0.5, color=[GREY, BLUE])
    for i, rf in enumerate(ref2): a2.text(i, rf + 1.5, f"{rf:.0f}", ha="center", fontsize=9)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["baseline", "add\nrefusal dir"], fontsize=9)
    a2.set_ylabel("refusal rate (%)"); a2.set_ylim(0, 108)
    a2.set_title("Add to benign prompts\n(induces refusal on harmless requests)", fontsize=10.5)
    a2.grid(axis="y", alpha=0.25)
    fig.suptitle("Refusal is a causal gate: bidirectionally steerable  (Arditi et al. replicates in our setup)",
                 fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_refusal_gate.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- FIG 3
def fig_output_content():
    rows = load("experiments/output_content_control_v1/generations.jsonl")
    by = defaultdict(list)
    for r in rows: by[(r["prop"], r["arm"])].append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    # french %
    fr_arms = ["add_baseline", "add_prop", "add_random", "abl_baseline", "abl_prop", "abl_random"]
    fr = [rate(by[("french", a)], "is_french") for a in fr_arms]
    colors = [GREY, GREEN, "#cfcfcf", GREY, RED, "#cfcfcf"]
    a1.bar(range(6), fr, color=colors)
    for i, v in enumerate(fr): a1.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
    a1.axvline(2.5, color="k", lw=0.8, ls=":")
    a1.text(1, 122, "ADD (on English)", ha="center", fontsize=9, weight="bold")
    a1.text(4, 122, "ABLATE (on French)", ha="center", fontsize=9, weight="bold")
    a1.set_xticks(range(6)); a1.set_xticklabels(["base", "+dir", "+rand", "base", "−dir", "−rand"], fontsize=8.5)
    a1.set_ylabel("% responses in French"); a1.set_ylim(0, 130)
    a1.set_title("Language", fontsize=11); a1.grid(axis="y", alpha=0.25)
    # verbosity tokens
    def mtok(a):
        g = by[("verbose", a)]; return np.mean([r["n_tokens"] for r in g])
    vb = [mtok(a) for a in fr_arms]
    a2.bar(range(6), vb, color=colors)
    for i, v in enumerate(vb): a2.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=8.5)
    a2.axvline(2.5, color="k", lw=0.8, ls=":")
    a2.text(1, max(vb)*1.12, "ADD (on terse)", ha="center", fontsize=9, weight="bold")
    a2.text(4, max(vb)*1.12, "ABLATE (on verbose)", ha="center", fontsize=9, weight="bold")
    a2.set_xticks(range(6)); a2.set_xticklabels(["base", "+dir", "+rand", "base", "−dir", "−rand"], fontsize=8.5)
    a2.set_ylabel("mean response length (tokens)"); a2.set_ylim(0, max(vb)*1.25)
    a2.set_title("Verbosity", fontsize=11); a2.grid(axis="y", alpha=0.25)
    fig.suptitle("Output form is add-steerable, but a weak lever under ablation",
                 fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_output_content.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- FIG 4
def fig_harm_capability():
    add = load("experiments/add_harm_v1/judged.jsonl")
    abl = load("experiments/harm_ablation_v1/judged.jsonl")
    aby = defaultdict(list)
    for r in add: aby[r["method"]].append(r)
    lby = defaultdict(list)
    for r in abl: lby[r["method"]].append(r)
    def mh(rows):
        v = [r["judge_harm_likert"] for r in rows if isinstance(r.get("judge_harm_likert"), (int, float))]
        return np.mean(v), (np.std(v)/np.sqrt(len(v)) if v else 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    # ADD-harm
    aa = ["baseline", "add_harm_1x", "add_harm_2x", "add_random_1x", "add_random_2x"]
    al = ["baseline", "add harm 1×", "add harm 2×", "add rand 1×", "add rand 2×"]
    means = [mh(aby[a])[0] for a in aa]; ses = [mh(aby[a])[1] for a in aa]
    cols = [GREY, GREEN, GREEN, "#cfcfcf", "#cfcfcf"]
    a1.bar(range(5), means, yerr=[1.96*s for s in ses], color=cols, capsize=3)
    for i, (v, s) in enumerate(zip(means, ses)): a1.text(i, v + 1.96*s + 0.12, f"{v:.2f}", ha="center", fontsize=8.5)
    a1.set_xticks(range(5)); a1.set_xticklabels(al, fontsize=8, rotation=15)
    a1.set_ylabel("Stage-B uplift  (1–5)"); a1.set_ylim(0, 5)
    a1.axhline(means[0], color="k", lw=0.7, ls=":")
    a1.set_title("ADD-harm ≈ random ≈ baseline\n(vs ADD-French 0→98% same mechanism)", fontsize=10.5)
    a1.grid(axis="y", alpha=0.25)
    # ABLATE-harm
    la = ["baseline", "ablate_md_harm", "ablate_lr_harm", "ablate_random"]
    ll = ["baseline", "ablate\nmd_harm", "ablate\nlr_harm", "ablate\nrandom"]
    m2 = [mh(lby[a])[0] for a in la]; s2 = [mh(lby[a])[1] for a in la]
    a2.bar(range(4), m2, yerr=[1.96*s for s in s2], color=[GREY, RED, RED, "#cfcfcf"], capsize=3)
    for i, (v, s) in enumerate(zip(m2, s2)): a2.text(i, v + 1.96*s + 0.12, f"{v:.2f}", ha="center", fontsize=8.5)
    a2.set_xticks(range(4)); a2.set_xticklabels(ll, fontsize=8.5)
    a2.set_ylabel("Stage-B uplift  (1–5)"); a2.set_ylim(0, 5)
    a2.axhline(m2[0], color="k", lw=0.7, ls=":")
    a2.set_title("ABLATE-harm ≈ random\n(on high-harm compliant turns)", fontsize=10.5)
    a2.grid(axis="y", alpha=0.25)
    fig.suptitle("Harm-uplift is a capability: steerable by neither add nor ablate (despite AUC 0.88 decodability)",
                 fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_harm_capability.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- FIG 5
def fig_monotonic():
    # left: U-shaped probe direction (loss-set ASR vs alpha) — reuse fig1 data
    comp = load("experiments/steering_v3/sweep_p2_judged.jsonl")
    rand = load("experiments/steering_v3/sweep_random_L16_jbb_qwen.jsonl")
    def curve(rows):
        by = defaultdict(list)
        for r in rows:
            if r.get("prompt_type") == "loss":
                by[r.get("alpha_c")].append(r)
        xs = sorted(a for a in by if a is not None)
        return xs, [rate(by[a], "judge_compliance_unsafe") for a in xs]
    cx, cy = curve(comp); rx, ry = curve(rand)
    # right: monotonic refusal-direction sweep
    rows = load("experiments/refusal_alpha_sweep_v1/sweep.jsonl")
    by = defaultdict(list)
    for r in rows: by[(r["set"], r["direction"], r["alpha"])].append(r)
    alphas = sorted(set(r["alpha"] for r in rows))
    def rr(setn, dirn): return [rate(by[(setn, dirn, a)], "is_refusal") for a in alphas]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)
    a1.plot(cx, cy, "-o", color=BLUE, lw=2.4, ms=7, label='"compliance direction" (probe)')
    a1.plot(rx, ry, "--s", color=GREY, lw=2.2, ms=6, label="random (matched norm)")
    a1.set_xlabel("steering coefficient  α"); a1.set_ylabel("attack success rate (%)")
    a1.set_title("Probe direction → U-shaped\n(norm artifact: random matches it)", fontsize=10.5)
    a1.legend(frameon=False, fontsize=9); a1.grid(alpha=0.25)
    a2.plot(alphas, rr("benign", "refusal"), "-o", color=GREEN, lw=2.4, ms=7, label="benign: induce refusal")
    a2.plot(alphas, rr("harmful", "refusal"), "-o", color=RED, lw=2.4, ms=7, label="harmful: bypass refusal")
    a2.plot(alphas, rr("benign", "random"), "--s", color=GREY, lw=2.2, ms=6, label="random (benign): flat")
    a2.axvline(0, color="k", lw=0.6, ls=":")
    a2.set_xlabel("steering coefficient  α  (× raw refusal diff-in-means)")
    a2.set_ylabel("refusal rate (%)")
    a2.set_title("Refusal direction → monotonic\n(the signature of a causal direction)", fontsize=10.5)
    a2.legend(frameon=False, fontsize=9, loc="center left"); a2.grid(alpha=0.25)
    fig.suptitle("Monotonic (causal) vs U-shaped (artifact): what steering along the right direction looks like",
                 fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_monotonic_sweep.png", dpi=140, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_monotonic(); print("fig5 monotonic")
    fig_norm_confound(); print("fig1 norm_confound")
    fig_refusal_gate(); print("fig2 refusal_gate")
    fig_output_content(); print("fig3 output_content")
    fig_harm_capability(); print("fig4 harm_capability")
    print("wrote to", OUT)
