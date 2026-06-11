"""Fit linear probes at L16 (compliance, harm) and save the L2-normalized
coefficient vectors as `.pt` files for use as steering directions.

  v_comp_L16.pt: shape [4096], compliance probe (per-turn label on 9,400 pool
                 minus Stage B convs).  Positive direction = "predicts breach".
  v_harm_L16.pt: shape [4096], harm probe (289 Stage B wins, Likert >=4 vs <=3).
                 Positive direction = "predicts high uplift".

Also saves an empirical estimate of ||h_L16|| (mean residual norm at L16,
last-token-of-prompt position) for the steering alpha scaling.
"""
import gc
import glob
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/steering_v3"
LAYER = 16
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
    norms = []
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i, (lab, tob) in enumerate(
                    zip(data["labels"].tolist(),
                        data["turns_of_breach"])):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_", "")
                           .replace(".pt", ""))
                key = (src, rnum, i)
                out[key] = {"hs": arr[i], "label": bool(lab),
                            "turn_of_breach": tob}
                # measure norms at the breach turn (or last turn if no breach)
                if bool(lab) and tob is not None:
                    norms.append(float(np.linalg.norm(arr[i, int(tob)])))
                else:
                    norms.append(float(np.linalg.norm(arr[i, -1])))
    return out, np.array(norms)


def main():
    os.makedirs(OUT, exist_ok=True)
    sb = load_sb()
    sb_keys = set(r["key"] for r in sb)
    wins = [r for r in sb if r["unsafe"]]
    y_harm = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)
    print(f"SB: {len(sb)} records, wins={len(wins)} "
          f"(high>={HARM_THRESH}: {int(y_harm.sum())})", flush=True)

    print(f"\nLoading pool at L{LAYER}...", flush=True)
    pool, h_norms = load_pool_layer(LAYER)
    print(f"  {len(pool)} pool records, {len(h_norms)} norm measurements",
          flush=True)
    h_med = float(np.median(h_norms))
    h_mean = float(np.mean(h_norms))
    print(f"  ||h_L{LAYER}||: median={h_med:.2f}, mean={h_mean:.2f}, "
          f"std={float(np.std(h_norms)):.2f}", flush=True)

    # Compliance probe: per-turn rows from non-SB convs
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
    w_c = clf_c.coef_.ravel().astype(np.float32)
    v_c = w_c / np.linalg.norm(w_c)
    print(f"\nCompliance probe: train rows={len(y_tr)}, "
          f"pos={int(y_tr.sum())}, ||w||={float(np.linalg.norm(w_c)):.2f}",
          flush=True)

    # Harm probe: SB wins HS at rated turn
    X_h = np.stack([pool[r["key"]]["hs"][r["rated_turn"]] for r in wins])
    clf_h = LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=2000, solver="lbfgs")
    clf_h.fit(X_h, y_harm)
    w_h = clf_h.coef_.ravel().astype(np.float32)
    v_h = w_h / np.linalg.norm(w_h)
    print(f"Harm probe: train rows={len(y_harm)}, "
          f"pos={int(y_harm.sum())}, ||w||={float(np.linalg.norm(w_h)):.2f}",
          flush=True)

    cos = float(v_c @ v_h)
    print(f"\ncos(v_comp, v_harm) = {cos:+.4f} (floor: ±{1/np.sqrt(4096):.4f})",
          flush=True)

    torch.save(torch.tensor(v_c), f"{OUT}/v_comp_L{LAYER}.pt")
    torch.save(torch.tensor(v_h), f"{OUT}/v_harm_L{LAYER}.pt")
    meta = {
        "layer": LAYER,
        "harm_thresh": HARM_THRESH,
        "n_train_comp_rows": len(y_tr),
        "n_train_comp_pos": int(y_tr.sum()),
        "n_train_harm_rows": len(wins),
        "n_train_harm_pos": int(y_harm.sum()),
        "raw_norm_w_comp": float(np.linalg.norm(w_c)),
        "raw_norm_w_harm": float(np.linalg.norm(w_h)),
        "cos_v_comp_v_harm": cos,
        "h_L16_norm_median": h_med,
        "h_L16_norm_mean": h_mean,
        "h_L16_norm_std": float(np.std(h_norms)),
        "sign_convention": "positive direction = 'predicts unsafe' (comp) "
                           "or 'predicts high harm' (harm)",
    }
    with open(f"{OUT}/steering_directions_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {OUT}/v_comp_L{LAYER}.pt, v_harm_L{LAYER}.pt, "
          f"steering_directions_meta.json")


if __name__ == "__main__":
    main()
