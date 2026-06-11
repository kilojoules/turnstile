"""Proper null distribution for v_d (the disagreement axis).

For each layer, repeat 1000 times:
  - draw 41 wins uniformly at random from the 289 wins
  - randomly split into groups of size 22 and 19
  - compute v_d_null = normalize( μ(22) - μ(19) )
  - record cos(v_d_null, v_md_comp_L)

Report:
  - null mean, sd, and 95% / 99% quantiles of |cos|
  - observed cos(v_d, v_md_comp) and its one-sided percentile in the null
  - whether the observed value is in the right tail

The null is "what would a random 22/19 split of 41 wins look like in terms of
MD-direction alignment with comp?" — small-N MD is noisy and its random-pair
floor is wider than the full-pool 4096-dim ±1/√4096 floor.
"""
import gc
import glob
import json
import os
import numpy as np
import torch

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
DIRS_OLD = f"{ROOT}/experiments/steering_v3/layer_sweep/directions"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
N_NULL = 1000
SEED = 42

POS_SIZE = 22  # Llama-only-pos
NEG_SIZE = 19  # Qwen-only-pos
TOTAL = POS_SIZE + NEG_SIZE  # 41 items drawn each iteration


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_win_residuals_with_labels(layer):
    cand_by_id = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cid = f"{r['source']}/r{r['round']}/i{r['idx']}"
        cand_by_id[cid] = r
    qwen = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if r.get("parse_ok"):
            qwen[r["conv_id"]] = r
    llama = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_scores_llama.jsonl"):
        r = json.loads(line)
        if r.get("parse_ok"):
            llama[r["conv_id"]] = r

    wins = []
    for cid, qr in qwen.items():
        if not qr.get("unsafe"):
            continue
        if cid not in llama or cid not in cand_by_id:
            continue
        cand = cand_by_id[cid]
        conv = cand["conversation"]
        asst = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        if not asst: continue
        tob = qr.get("turn_of_breach")
        rated = int(tob) if (tob is not None and int(tob) < len(asst)) else len(asst) - 1
        wins.append({
            "conv_id": cid, "source": cand["source"],
            "round": cand["round"], "idx": cand["idx"],
            "rated_turn": rated,
            "qwen_rating": qr["rating"],
            "llama_rating": llama[cid]["rating"],
        })

    win_keys = {(w["source"], w["round"], w["idx"]): w for w in wins}
    hs_by_id = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False, map_location="cpu")
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i in range(len(data["labels"])):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_","").replace(".pt",""))
                k = (src, rnum, i)
                if k in win_keys:
                    hs_by_id[win_keys[k]["conv_id"]] = arr[i]
            del data, arr; gc.collect()
    wins = [w for w in wins if w["conv_id"] in hs_by_id]
    X = np.stack([hs_by_id[w["conv_id"]][w["rated_turn"]] for w in wins])
    return X, wins


def main():
    print(f"Null sample size per layer: {N_NULL} draws of {TOTAL} wins, "
          f"split {POS_SIZE}/{NEG_SIZE}")
    print(f"{'L':>3}  {'observed':>10}  "
          f"{'null mean cos':>15} {'null sd':>9}  "
          f"{'null |cos| q95':>16} {'null |cos| q99':>16}  "
          f"{'percentile':>12} {'verdict':>8}")
    print("-" * 110)

    rng = np.random.default_rng(SEED)
    per_layer = []

    for L in LAYERS:
        X, wins = load_win_residuals_with_labels(L)
        N = len(wins)
        q = np.array([w["qwen_rating"] for w in wins])
        l = np.array([w["llama_rating"] for w in wins])
        q_pos = q >= 4
        l_pos = l >= 4
        # observed v_d
        C = q_pos & (~l_pos)   # 19 Qwen-only-pos
        D = (~q_pos) & l_pos   # 22 Llama-only-pos
        v_d_obs = normalize((X[D].mean(0) - X[C].mean(0)).astype(np.float32))

        v_md_comp = torch.load(f"{DIRS_OLD}/v_md_comp_L{L}.pt",
                               weights_only=False).float().numpy()
        obs_cos = float(np.dot(v_d_obs, v_md_comp))

        # Null distribution
        null_cos = np.empty(N_NULL)
        for k in range(N_NULL):
            picks = rng.choice(N, size=TOTAL, replace=False)
            np.random.default_rng(rng.integers(2**32)).shuffle(picks)
            pos_idx = picks[:POS_SIZE]
            neg_idx = picks[POS_SIZE:]
            v_null = normalize(
                (X[pos_idx].mean(0) - X[neg_idx].mean(0)).astype(np.float32))
            null_cos[k] = float(np.dot(v_null, v_md_comp))

        # Stats
        null_mean = float(null_cos.mean())
        null_sd = float(null_cos.std())
        q95 = float(np.percentile(np.abs(null_cos), 95))
        q99 = float(np.percentile(np.abs(null_cos), 99))
        # Percentile of |obs_cos| in null distribution of |cos|
        pct = float(100 * (np.abs(null_cos) < abs(obs_cos)).mean())
        verdict = "real" if pct > 95 else ("borderline" if pct > 90 else "null")

        print(f"L{L:<2}  {obs_cos:+10.4f}  {null_mean:+15.4f} {null_sd:9.4f}  "
              f"{q95:+16.4f} {q99:+16.4f}  {pct:>11.1f}% {verdict:>8}")

        per_layer.append({
            "layer": L,
            "observed_cos_vd_vmdcomp": obs_cos,
            "null_n_draws": N_NULL,
            "null_mean_cos": null_mean,
            "null_sd": null_sd,
            "null_abs_cos_q95": q95,
            "null_abs_cos_q99": q99,
            "percentile_in_null": pct,
            "verdict": verdict,
        })
        del X, wins; gc.collect()

    out = {"per_layer": per_layer, "n_null": N_NULL,
           "pos_size": POS_SIZE, "neg_size": NEG_SIZE}
    out_path = (f"{ROOT}/experiments/steering_v3/layer_sweep/"
                "null_v_d.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
