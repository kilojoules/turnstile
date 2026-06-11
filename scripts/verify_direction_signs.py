"""Empirical sign check on all 36 (concept × method × layer) directions.

For each direction v we report:
  mean projection (x - μ) · v  on the positive class vs negative class.

If positive-class mean is LESS than negative-class mean, the direction is
sign-flipped relative to its label convention.

Concept positive class:
  compliance: breach turn (y=1)
  harm:       Stage-B Likert ≥ 4
"""
import gc, glob, json, os
import numpy as np
import torch

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
DIRS = f"{ROOT}/experiments/steering_v3/layer_sweep/directions"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4


def load_compliance(layer, sb_keys):
    X, y = [], []
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            labels = data["labels"].tolist()
            tobs = data["turns_of_breach"]
            for i in range(len(labels)):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_","").replace(".pt",""))
                if (src, rnum, i) in sb_keys:
                    continue
                breach = bool(labels[i])
                t_star = tobs[i] if breach else None
                if breach and t_star is None:
                    continue
                t_max = 4 if not breach else int(t_star)
                for t in range(t_max + 1):
                    X.append(arr[i, t])
                    y.append(1 if (breach and t == t_star) else 0)
            del data, arr; gc.collect()
    return np.stack(X), np.array(y)


def load_harm(layer):
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    wins = []
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok") or not r.get("unsafe"): continue
        parts = r["conv_id"].split("/")
        if len(parts) != 3: continue
        key = (parts[0], int(parts[1].lstrip("r")), int(parts[2].lstrip("i")))
        if key not in cand: continue
        conv = cand[key]["conversation"]
        asst = [i for i, t in enumerate(conv) if t["role"]=="assistant"]
        if not asst: continue
        if r.get("turn_of_breach") is not None and int(r["turn_of_breach"]) < len(asst):
            rated = int(r["turn_of_breach"])
        else:
            rated = len(asst)-1
        wins.append({"key": key, "rated_turn": rated, "harm": r["rating"]})
    win_keys = {r["key"] for r in wins}
    hs_by_key = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i in range(len(data["labels"])):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_","").replace(".pt",""))
                k = (src, rnum, i)
                if k in win_keys:
                    hs_by_key[k] = arr[i]
            del data, arr; gc.collect()
    X = np.stack([hs_by_key[r["key"]][r["rated_turn"]] for r in wins])
    y = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)
    return X, y


def main():
    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))

    print(f"{'L':>3}  {'method':<10}  {'pos_proj_mean':>15}  {'neg_proj_mean':>15}  "
          f"{'Δ (pos-neg)':>14}  status")
    print("-" * 84)
    issues = []
    for L in LAYERS:
        Xc, yc = load_compliance(L, sb_keys)
        Xh, yh = load_harm(L)

        for method, X, y in [
            ("lr_comp", Xc, yc), ("md_comp", Xc, yc),
            ("lr_harm", Xh, yh), ("md_harm", Xh, yh),
        ]:
            v = torch.load(f"{DIRS}/v_{method}_L{L}.pt", weights_only=False).float().numpy()
            mu = X.mean(0)
            proj = (X - mu) @ v
            pos_m = float(proj[y == 1].mean())
            neg_m = float(proj[y == 0].mean())
            delta = pos_m - neg_m
            status = "OK" if delta > 0 else "**FLIPPED**"
            if delta <= 0:
                issues.append((L, method, delta))
            print(f"L{L:<2}  {method:<10}  {pos_m:+15.4f}  {neg_m:+15.4f}  "
                  f"{delta:+14.4f}  {status}")
        del Xc, yc, Xh, yh; gc.collect()

    print()
    if issues:
        print(f"{len(issues)} directions flipped:")
        for L, m, d in issues:
            print(f"  L{L} {m}  Δ={d:+.4f}")
    else:
        print("All 36 directions have positive class > negative class. No sign flips.")


if __name__ == "__main__":
    main()
