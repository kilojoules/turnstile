"""Per-layer cosine similarity between the compliance probe direction and
the harm probe direction.

For each layer L:
  - Fit linear LR for compliance (per-turn rows from non-SB convs)
  - Fit linear LR for harm (289 SB wins, high-harm vs low-harm)
  - Normalize each coef vector; compute cosine.

Also reports the random-cosine floor (~1/sqrt(d)) for context.

Output:
  experiments/outcome_probe_v1/probe_cosine_per_layer.json
  figures/probe_cosine_per_layer.{pdf,png}
"""
import gc
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4


def load_sb():
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    out = []
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        parts = r["conv_id"].split("/")
        if len(parts) != 3:
            continue
        key = (parts[0], int(parts[1].lstrip("r")), int(parts[2].lstrip("i")))
        if key not in cand:
            continue
        conv = cand[key]["conversation"]
        asst_turns = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        if not asst_turns:
            continue
        if r["unsafe"] and r.get("turn_of_breach") is not None and \
                int(r["turn_of_breach"]) < len(asst_turns):
            rated_turn = int(r["turn_of_breach"])
        else:
            rated_turn = len(asst_turns) - 1
        out.append({"key": key, "rated_turn": rated_turn,
                    "unsafe": r["unsafe"], "harm": r["rating"]})
    return out


def load_pool_layer(layer):
    out = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        src = os.path.basename(sdir.rstrip("/"))
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            rnum = int(os.path.basename(path).replace("round_", "")
                       .replace(".pt", ""))
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i, (lab, tob) in enumerate(
                    zip(data["labels"].tolist(),
                        data["turns_of_breach"])):
                out[(src, rnum, i)] = {"hs": arr[i], "label": bool(lab),
                                       "turn_of_breach": tob}
    return out


def main():
    sb = load_sb()
    sb_keys = set(r["key"] for r in sb)
    wins = [r for r in sb if r["unsafe"]]
    y_harm = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)
    print(f"SB: {len(sb)} records, wins={len(wins)}, "
          f"harm>=4: {int(y_harm.sum())}", flush=True)

    results = {"layers": LAYERS,
               "random_floor_d4096": 1 / math.sqrt(4096),
               "per_layer": {}}

    for L in LAYERS:
        print(f"\nLayer L{L}...", flush=True)
        pool = load_pool_layer(L)

        # compliance training
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

        clf_c = LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=2000, solver="lbfgs")
        clf_c.fit(X_tr, y_tr)
        w_c = clf_c.coef_.ravel()

        # harm training (on SB wins)
        X_h = np.stack([pool[r["key"]]["hs"][r["rated_turn"]] for r in wins])
        clf_h = LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=2000, solver="lbfgs")
        clf_h.fit(X_h, y_harm)
        w_h = clf_h.coef_.ravel()

        cos = float((w_c @ w_h) / (np.linalg.norm(w_c)
                                    * np.linalg.norm(w_h)))
        nrm_c = float(np.linalg.norm(w_c))
        nrm_h = float(np.linalg.norm(w_h))
        results["per_layer"][f"L{L}"] = {
            "cosine": cos,
            "norm_compliance_w": nrm_c, "norm_harm_w": nrm_h,
            "n_train_comp_rows": len(y_tr),
            "n_train_comp_pos": int(y_tr.sum()),
        }
        print(f"  cos(w_comp, w_harm) = {cos:+.4f}  "
              f"(|w_c|={nrm_c:.2f}, |w_h|={nrm_h:.2f}, "
              f"train rows comp={len(y_tr)})", flush=True)

        del pool, X_tr, y_tr, X_h
        gc.collect()

    with open(f"{OUT}/probe_cosine_per_layer.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}/probe_cosine_per_layer.json")

    # plot
    cosines = [results["per_layer"][f"L{L}"]["cosine"] for L in LAYERS]
    floor = 1 / math.sqrt(4096)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(LAYERS, cosines, marker="o", markersize=7, linewidth=1.8,
            color="#3b6fb0", label="cos(compliance probe, harm probe)")
    ax.fill_between(LAYERS, [-floor]*len(LAYERS), [floor]*len(LAYERS),
                    color="gray", alpha=0.18,
                    label=f"random-direction floor ±1/√d = ±{floor:.4f}")
    ax.axhline(0, color="black", linewidth=0.6)

    for L, c in zip(LAYERS, cosines):
        ax.text(L, c + 0.012, f"{c:+.3f}", ha="center", fontsize=8.5)

    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_xlabel("Llama-3.1-8B-Instruct layer")
    ax.set_ylabel("cosine similarity of probe coefficient vectors")
    ax.set_title("Compliance-probe ↔ harm-probe cosine, per layer\n"
                 "(compliance probe trained on 9,400-conv pool per-turn; "
                 "harm probe on 289 SB wins)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92)

    # set ylim with some headroom
    ymin = min(cosines) - 0.05; ymax = max(cosines) + 0.05
    ax.set_ylim(min(ymin, -0.05), max(ymax, 0.4))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/probe_cosine_per_layer.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
