"""Per-behavior ASR by goal verb type.

Verb-type taxonomy (manual / keyword-based):
  1. Article/essay  — write an article/post/blog/tweet/speech/poem/letter/...
  2. Code/tool      — write code, design a script, develop a virus, etc.
  3. Step-by-step instructions — explain/describe/provide how to (operational)
  4. Strategy/plan/scheme — design a plan, develop a strategy
  5. Evidence/argument  — provide evidence, argue for a false claim
  6. List/template  — list of items, template, example phrases
  7. Other          — fallback

Outputs:
  experiments/outcome_probe_v1/per_behavior_verb_classification.json
  figures/asr_by_verb_type.{pdf,png}
"""
import glob
import json
import math
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(goal):
    g = goal.lower()
    # 1. Evidence / proof / argument for false claim (factual commitment)
    if re.search(r"\b(evidence|proof|prove)\b", g) and \
            re.search(r"\b(provide|present|give|demonstrate|show|find)\b", g):
        return "evidence/argument"
    if re.search(r"\b(argue|justify)\b", g):
        return "evidence/argument"
    if re.search(r"\b(suggest|claim)\b.*(true|real|fake|never|caused|cause)", g):
        return "evidence/argument"

    # 2. Code / tool / technical artifact
    if re.search(r"\b(code|script|program|software|virus|malware|"
                 r"key.?logger|exploit|algorithm|bot|application|"
                 r"trojan|ransomware|payload|sql injection|"
                 r"python|javascript|encrypt|decrypt|hack into|"
                 r"compromise|brute.?force|crack)\b", g):
        return "code/tool"

    # 3. Article / essay / rhetorical artifact (recognized text format)
    if re.search(r"\b(article|essay|blog|tweet|post|speech|letter|"
                 r"email|message|poem|story|persuasive|"
                 r"manifesto|op[-_ ]ed|editorial|"
                 r"facebook post|twitter post|news.*piece|"
                 r"argument|sermon)\b", g):
        return "article/essay"

    # 4. Step-by-step instructions / how-to / tutorial
    if re.search(r"\b(step.?by.?step|how to|tutorial|guide|instructions|"
                 r"detailed instructions|procedure)\b", g):
        return "instructions"

    # 5. Strategy / plan / scheme (high-level operational artifact)
    if re.search(r"\b(plan|strategy|scheme|method|approach|"
                 r"playbook|roadmap)\b", g):
        return "plan/strategy"

    # 6. List / template / example phrases
    if re.search(r"\b(list|template|examples|sample|enumerate|"
                 r"phrases|scripts? for|talking points)\b", g):
        return "list/template"

    return "other"


def main():
    goals = json.load(open(f"{ROOT}/working/uplift/goals.json"))
    goal_to_behavior = {g["goal"]: g["behavior"] for g in goals}
    behavior_to_goal = {g["behavior"]: g["goal"] for g in goals}
    behavior_to_cat = {g["behavior"]: g["category"] for g in goals}
    behavior_to_verb = {g["behavior"]: classify(g["goal"]) for g in goals}

    # Per-behavior ASR from the 9,400-conv pool
    wins = defaultdict(int); trials = defaultdict(int)
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            for lab, goal in zip(data["labels"].tolist(), data["goals"]):
                b = goal_to_behavior.get(goal)
                if b is None:
                    continue
                trials[b] += 1
                wins[b] += int(bool(lab))

    rows = []
    for b in sorted(behavior_to_goal.keys()):
        n = trials.get(b, 0); w = wins.get(b, 0)
        if n == 0:
            continue
        lo, hi = wilson(w, n)
        rows.append({
            "behavior": b,
            "goal": behavior_to_goal[b],
            "category": behavior_to_cat[b],
            "verb_type": behavior_to_verb[b],
            "n": n, "n_wins": w,
            "asr": w / n, "asr_lo95": lo, "asr_hi95": hi,
        })

    # save classification + per-behavior ASR
    with open(f"{OUT}/per_behavior_verb_classification.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {OUT}/per_behavior_verb_classification.json")

    # ----- aggregate by verb type -----
    by_verb = defaultdict(list)
    for r in rows:
        by_verb[r["verb_type"]].append(r)

    print(f"\n{'verb type':<22}  n_beh  total_wins  total_trials  mean ASR  median ASR")
    rank = []
    for v, rs in by_verb.items():
        nb = len(rs)
        tw = sum(r["n_wins"] for r in rs)
        tt = sum(r["n"] for r in rs)
        asrs = [r["asr"] for r in rs]
        mean_asr = np.mean(asrs)
        med_asr = np.median(asrs)
        rank.append((v, nb, tw, tt, mean_asr, med_asr, asrs))

    rank.sort(key=lambda x: -x[4])  # descending mean ASR
    for v, nb, tw, tt, mean_asr, med_asr, _ in rank:
        print(f"  {v:<22}  {nb:>4}   {tw:>6}/{tt:<6}     "
              f"{100*mean_asr:>5.1f}%    {100*med_asr:>5.1f}%")

    # ----- plot: box+strip of per-behavior ASR by verb type -----
    verbs_sorted = [r[0] for r in rank]
    data = [[100 * x for x in r[6]] for r in rank]
    counts = [r[1] for r in rank]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bp = ax.boxplot(data, positions=range(len(verbs_sorted)),
                    widths=0.55, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6),
                    medianprops=dict(color="black", linewidth=1.2),
                    boxprops=dict(facecolor="#3b6fb0", alpha=0.6,
                                  edgecolor="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(marker="o", markerfacecolor="#3b6fb0",
                                    markeredgecolor="black", markersize=4,
                                    alpha=0.7))
    # strip overlay
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        xs = rng.normal(i, 0.06, size=len(vals))
        ax.scatter(xs, vals, s=18, color="#1f3a5f", alpha=0.6,
                   edgecolor="black", linewidth=0.3, zorder=3)

    ax.set_xticks(range(len(verbs_sorted)))
    ax.set_xticklabels([f"{v}\n(n={c})" for v, c in zip(verbs_sorted, counts)],
                       fontsize=9)
    ax.set_ylabel("per-behavior ASR (%, n behaviors per bucket shown)")
    ax.set_xlabel("goal verb / output type")
    ax.set_title("Per-behavior ASR by goal verb type (n=100 JBB behaviors)\n"
                 "white diamond = mean; line = median; dots = individual behaviors",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_ylim(-2, max(max(d) for d in data) + 5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/asr_by_verb_type.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"\nwrote {out_p}")
    plt.close(fig)

    # print "other" bucket for sanity
    print(f"\nGoals classified as 'other':")
    for r in rows:
        if r["verb_type"] == "other":
            print(f"  [{r['category']:<25}]  {r['behavior']:<28}  ASR={100*r['asr']:5.1f}%")
            print(f"      goal: {r['goal']}")

    # also: notable extremes per bucket
    print("\nMin / max per bucket:")
    for v, _, _, _, _, _, asrs in rank:
        rs = by_verb[v]
        rs_sorted = sorted(rs, key=lambda r: r["asr"])
        print(f"\n  {v}:")
        print(f"    lowest 3 ASR:")
        for r in rs_sorted[:3]:
            print(f"      {100*r['asr']:>5.1f}%  ({r['n_wins']:>2}/{r['n']:<3})  "
                  f"[{r['category'][:20]}]  {r['behavior']}")
        print(f"    highest 3 ASR:")
        for r in rs_sorted[-3:]:
            print(f"      {100*r['asr']:>5.1f}%  ({r['n_wins']:>2}/{r['n']:<3})  "
                  f"[{r['category'][:20]}]  {r['behavior']}")


if __name__ == "__main__":
    main()
