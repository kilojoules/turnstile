"""Phase 1 figure: whitened probes are weak steering vectors.
Overlay the diff-in-means refusal curve (a real lever) against the whitened
refusal_probe / comp_probe / random on harmful prompts. Compliance + coherence.
"""
import json, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "figures/causal_steering"
TEAL, CRIM, GREY, PURP = "#1b9e77", "#d1495b", "#8a8a8a", "#7b3fa0"

def series(path, keyfn):
    by = defaultdict(list)
    for r in map(json.loads, open(path)): by[keyfn(r)].append(r)
    return by
def comp(by, k): g = by.get(k, []); return 100*sum(1 for r in g if r.get("judge_compliance_unsafe"))/len(g) if g else np.nan
def coh(by, k): g = by.get(k, []); return np.mean([r["coherence"]["token_unique_ratio"] for r in g]) if g else np.nan

# diff-in-means refusal reference (harmful prompts)
rdm = series("experiments/refusal_harm_vs_compliance_v1/judged.jsonl", lambda r: r["alpha"])
Adm = sorted(rdm)
# phase 1 probe steering
p1 = series("experiments/phase1_probe_steer/judged_llama.jsonl", lambda r: (r["direction"], r["alpha"]))
A = sorted({a for (d, a) in p1})
def pk(d, a): return ("baseline", 0.0) if a == 0.0 else (d, a)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
a1.plot(Adm, [comp(rdm, a) for a in Adm], "-o", color="k", lw=3, ms=6, label="refusal DIFF-IN-MEANS (a real lever)")
a1.plot(A, [comp(p1, pk("refusal_probe", a)) for a in A], "-s", color=TEAL, lw=2.2, ms=6, label="refusal PROBE (whitened, pos. control)")
a1.plot(A, [comp(p1, pk("comp_probe", a)) for a in A], "-^", color=PURP, lw=2.2, ms=6, label="compliance PROBE")
a1.plot(A, [comp(p1, pk("random", a)) for a in A], "--o", color=GREY, lw=1.8, ms=5, label="random")
a1.set_xlabel("steering strength  α   (−α = subtract)"); a1.set_ylabel("compliance / attack success (%)")
a1.set_ylim(-4, 70); a1.grid(alpha=0.25); a1.axvline(0, color="k", lw=0.5, ls=":")
a1.legend(frameon=False, fontsize=8.2, loc="upper right")
a1.set_title("Diff-in-means steers hard (60%); the whitened probe\nbarely beats random even at cos 0.87 to it", fontsize=10)
for d, c, m in [("refusal_probe", TEAL, "s"), ("comp_probe", PURP, "^"), ("random", GREY, "o")]:
    a2.plot(A, [coh(p1, pk(d, a)) for a in A], "-"+m, color=c, lw=2, ms=5, label=d)
a2.axhspan(0, 0.6, color="#f2d0d0", alpha=0.5); a2.text(-1.45, 0.30, "degraded /\ngibberish", fontsize=8, color=CRIM)
a2.set_xlabel("steering strength  α"); a2.set_ylabel("coherence (unique-token ratio)")
a2.set_ylim(0, 1.0); a2.grid(alpha=0.25); a2.legend(frameon=False, fontsize=8, loc="lower left")
a2.set_title("The probe's larger numbers appear only where\ncoherence has collapsed (degradation false-positive)", fontsize=10)
fig.suptitle("Phase 1 gate — whitened PROBE directions are weak steering vectors: we base the harm null on DIFF-IN-MEANS, not probes",
             fontsize=11.5, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/figP1_probe_vs_dm.png", dpi=140, bbox_inches="tight")
print("saved figP1_probe_vs_dm.png")
