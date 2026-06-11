"""Scatter: Stage B real-world harm Likert (1-5) vs probe score at the
pivot turn (breach-turn for wins, last-turn for losses).

One point per Stage B record (n=600). Colored by wins / losses.
Reproduces the probe scores from scripts/stage_b_probe_harm_correlation.py
so this script is self-contained.
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYER = 16


def load_pool(layer):
    out = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        src = os.path.basename(sdir.rstrip("/"))
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            rnum = int(os.path.basename(path).replace("round_", "")
                       .replace(".pt", ""))
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i, (lab, tob, cat) in enumerate(
                    zip(data["labels"].tolist(),
                        data["turns_of_breach"],
                        data["categories"])):
                out[(src, rnum, i)] = {
                    "hs": arr[i], "label": bool(lab),
                    "turn_of_breach": tob, "category": cat,
                }
    return out


def main():
    print(f"Loading pool at L{LAYER}...", flush=True)
    pool = load_pool(LAYER)

    stage_b = [json.loads(l)
               for l in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl")]
    cand_meta = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand_meta[(r["source"], r["round"], r["idx"])] = r

    keyed = []
    for r in stage_b:
        if not r.get("parse_ok"):
            continue
        parts = r["conv_id"].split("/")
        src = parts[0]; rnum = int(parts[1].lstrip("r"))
        idx = int(parts[2].lstrip("i"))
        key = (src, rnum, idx)
        if key not in pool or key not in cand_meta:
            continue
        cand = cand_meta[key]
        conv = cand["conversation"]
        asst_turns = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        num_turns = len(asst_turns)
        if num_turns == 0:
            continue
        if r["unsafe"] and r.get("turn_of_breach") is not None and \
                int(r["turn_of_breach"]) < num_turns:
            rated_turn = int(r["turn_of_breach"])
        else:
            rated_turn = num_turns - 1
        keyed.append({
            "key": key, "rated_turn": rated_turn,
            "unsafe": r["unsafe"], "harm": r["rating"],
            "category": r["category"],
        })

    sb_keys = set(r["key"] for r in keyed)

    # build per-turn training set from non-Stage-B conversations
    X_tr, y_tr = [], []
    for key, rec in pool.items():
        if key in sb_keys:
            continue
        breach = rec["label"]
        t_star = rec["turn_of_breach"] if breach else None
        if breach and t_star is None:
            continue
        t_max = 4 if not breach else int(t_star)
        for t in range(t_max + 1):
            X_tr.append(rec["hs"][t])
            y_tr.append(1 if (breach and t == t_star) else 0)
    X_tr = np.stack(X_tr); y_tr = np.array(y_tr)
    X_te = np.stack([pool[r["key"]]["hs"][r["rated_turn"]] for r in keyed])

    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    probe = clf.predict_proba(X_te)[:, 1]
    for r, s in zip(keyed, probe):
        r["probe"] = float(s)

    wins = [r for r in keyed if r["unsafe"]]
    losses = [r for r in keyed if not r["unsafe"]]

    rho_w, p_w = spearmanr([r["harm"] for r in wins],
                           [r["probe"] for r in wins])
    rho_l, p_l = spearmanr([r["harm"] for r in losses],
                           [r["probe"] for r in losses])
    rho_a, p_a = spearmanr([r["harm"] for r in keyed],
                           [r["probe"] for r in keyed])

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    # jittered y so points at same Likert don't pile up
    for grp, color, label, marker in [
        (losses, "#7f7f7f", f"losses (n={len(losses)})", "o"),
        (wins, "#d62728", f"wins (n={len(wins)})", "o"),
    ]:
        x = np.array([r["probe"] for r in grp])
        y = np.array([r["harm"] for r in grp])
        y_j = y + rng.uniform(-0.18, 0.18, size=len(y))
        ax.scatter(x, y_j, s=22, color=color, edgecolor="black",
                   linewidth=0.3, alpha=0.65, label=label, marker=marker)

    # overlay group means (no jitter)
    for grp, color in [(losses, "#7f7f7f"), (wins, "#d62728")]:
        mean_probe = np.mean([r["probe"] for r in grp])
        mean_harm = np.mean([r["harm"] for r in grp])
        ax.scatter(mean_probe, mean_harm, s=200, color=color,
                   edgecolor="black", linewidth=1.2, marker="X", zorder=5)

    ax.set_xlabel("probe score at pivot turn (P[unsafe | HS at breach turn for wins, last turn for losses])")
    ax.set_ylabel("Stage B real-world uplift rating (Qwen-72B Likert 1-5)")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 5.5)

    stats_text = (f"Wins only (n={len(wins)}):   Spearman ρ = {rho_w:+.2f}, p = {p_w:.2f}\n"
                  f"Losses only (n={len(losses)}):  Spearman ρ = {rho_l:+.2f}, p = {p_l:.2f}\n"
                  f"All 600:           Spearman ρ = {rho_a:+.2f}, p = {p_a:.3f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.93))

    ax.set_title("Real-world harm vs compliance probe score at pivot turn\n"
                 "(Stage B n=600; X = group means)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.93)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/harm_vs_probe_scatter.{ext}"
        fig.savefig(out_p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
