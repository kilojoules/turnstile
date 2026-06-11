"""Phase 2.3: sign check on Llama-relabeled LR-harm and MD-harm directions.

Loads new harm directions from directions_llama/, projects the harm-probe
training residuals on them, confirms that the Llama-Likert≥4 (positive) class
has more-positive mean projection than the Likert<4 class. Flips on disk if
sign is inverted.
"""
import gc
import glob
import json
import os
import numpy as np
import torch

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
DIRS_LLAMA = f"{ROOT}/experiments/steering_v3/layer_sweep/directions_llama"
SCORES = f"{ROOT}/working/uplift/stage_b_scores_llama.jsonl"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4


def load_harm(layer):
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    wins = []
    for line in open(SCORES):
        r = json.loads(line)
        if not r.get("parse_ok") or not r.get("unsafe"):
            continue
        parts = r["conv_id"].split("/")
        if len(parts) != 3:
            continue
        key = (parts[0], int(parts[1].lstrip("r")), int(parts[2].lstrip("i")))
        if key not in cand:
            continue
        conv = cand[key]["conversation"]
        asst = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        if not asst:
            continue
        if r.get("turn_of_breach") is not None and int(r["turn_of_breach"]) < len(asst):
            rated = int(r["turn_of_breach"])
        else:
            rated = len(asst) - 1
        wins.append({"key": key, "rated_turn": rated, "harm": r["rating"]})
    win_keys = {r["key"] for r in wins}
    hs_by_key = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False, map_location="cpu")
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i in range(len(data["labels"])):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_","").replace(".pt",""))
                key = (src, rnum, i)
                if key in win_keys:
                    hs_by_key[key] = arr[i]
            del data, arr; gc.collect()
    wins = [w for w in wins if w["key"] in hs_by_key]
    X = np.stack([hs_by_key[w["key"]][w["rated_turn"]] for w in wins])
    y = np.array([w["harm"] >= HARM_THRESH for w in wins], dtype=int)
    return X, y


def main():
    print(f"{'L':>3}  {'method':<10}  {'pos_proj_mean':>15}  {'neg_proj_mean':>15}  "
          f"{'Δ (pos-neg)':>14}  status")
    print("-" * 84)
    issues = []
    for L in LAYERS:
        X, y = load_harm(L)
        for method in ["lr_harm", "md_harm"]:
            path = f"{DIRS_LLAMA}/v_{method}_L{L}.pt"
            v = torch.load(path, weights_only=False).float().numpy()
            mu = X.mean(0)
            proj = (X - mu) @ v
            pos_m = float(proj[y == 1].mean())
            neg_m = float(proj[y == 0].mean())
            delta = pos_m - neg_m
            status = "OK" if delta > 0 else "**FLIPPED — fixing**"
            if delta <= 0:
                # auto-flip on disk
                v_fixed = -v
                torch.save(torch.tensor(v_fixed.astype(np.float32)), path)
                issues.append((L, method, delta))
            print(f"L{L:<2}  {method:<10}  {pos_m:+15.4f}  {neg_m:+15.4f}  "
                  f"{delta:+14.4f}  {status}")
        del X, y; gc.collect()
    print()
    if issues:
        print(f"Flipped on disk: {len(issues)} directions:")
        for L, m, d in issues:
            print(f"  L{L} {m}  Δ was {d:+.4f}, now reversed")
    else:
        print("All 18 Llama-fit harm directions have positive class > negative. No flips.")


if __name__ == "__main__":
    main()
