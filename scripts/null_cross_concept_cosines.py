"""Partition-shuffle null for every cross-concept cosine that appears in any
figure or claim.

For each (layer, method ∈ {LR, MD}, harm-judge ∈ {Qwen, Llama}):

  observed = cos( fit_harm(X_h, y_h_real),
                  fit_comp(X_c, y_c_real) )

  null distribution (n_iters):
    y_h_shuf = permute(y_h_real)        # preserve partition size
    y_c_shuf = permute(y_c_real)        # preserve partition size
    cos_null = cos( fit_harm(X_h, y_h_shuf), fit_comp(X_c, y_c_shuf) )

Both directions are refit per iter — this is essential, otherwise the null
underestimates the small-sample variance of MD over 124–127 positives and the
LR-direction noise from balanced subsampling 1744 positives + 1744 negatives.

Comp labels (Llama-tagged per-turn breach) are FIXED across both harm-judge
runs; the only thing that differs between Qwen and Llama harm runs is the
y_h label vector for the 289 wins.

Output:
  per-layer table: observed vs null mean / sd / q95 / q99 / percentile / verdict
  JSON dump at experiments/steering_v3/layer_sweep/null_cross_concept_cosines.json
"""
import argparse
import gc
import glob
import json
import os
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
DIRS_OLD = f"{ROOT}/experiments/steering_v3/layer_sweep/directions"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4
SEED = 42


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_compliance(layer, sb_keys):
    X, y = [], []
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False, map_location="cpu")
            arr = data["hidden_states_by_layer"][layer].numpy()
            labels = data["labels"].tolist()
            tobs = data["turns_of_breach"]
            for i in range(len(labels)):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_", "").replace(".pt", ""))
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
            del data, arr
            gc.collect()
    return np.stack(X), np.array(y, dtype=np.int32)


def load_harm(layer, scores_path):
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    wins = []
    for line in open(scores_path):
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
                rnum = int(os.path.basename(path).replace("round_", "").replace(".pt", ""))
                k = (src, rnum, i)
                if k in win_keys:
                    hs_by_key[k] = arr[i]
            del data, arr
            gc.collect()
    wins = [w for w in wins if w["key"] in hs_by_key]
    X = np.stack([hs_by_key[w["key"]][w["rated_turn"]] for w in wins])
    y = np.array([w["harm"] >= HARM_THRESH for w in wins], dtype=np.int32)
    return X, y


def fit_harm_lr(X, y):
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    return normalize(clf.coef_.ravel().astype(np.float32))


def fit_harm_md(X, y):
    return normalize((X[y == 1].mean(0) - X[y == 0].mean(0)).astype(np.float32))


def fit_comp_lr(X, y, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    sel = np.concatenate([pos_idx, keep])
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(X[sel], y[sel])
    return normalize(clf.coef_.ravel().astype(np.float32))


def fit_comp_md(X, y, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    sel = np.concatenate([pos_idx, keep])
    Xb, yb = X[sel], y[sel]
    return normalize((Xb[yb == 1].mean(0) - Xb[yb == 0].mean(0)).astype(np.float32))


def run_null(X_h, y_h, X_c, y_c, method, n_iters, rng):
    fit_harm = fit_harm_lr if method == "lr" else fit_harm_md
    fit_comp = fit_comp_lr if method == "lr" else fit_comp_md
    # observed (with real labels; for comp_lr/md use a fresh rng draw for the
    # balanced subsample to match the null's mode of variation)
    v_h_obs = fit_harm(X_h, y_h)
    v_c_obs = fit_comp(X_c, y_c, np.random.default_rng(SEED + 7919))
    obs = float(np.dot(v_h_obs, v_c_obs))

    null = np.empty(n_iters, dtype=np.float64)
    t0 = time.time()
    for k in range(n_iters):
        y_h_shuf = rng.permutation(y_h)
        y_c_shuf = rng.permutation(y_c)
        v_h = fit_harm(X_h, y_h_shuf)
        v_c = fit_comp(X_c, y_c_shuf, rng)
        null[k] = float(np.dot(v_h, v_c))
        if (k + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"      iter {k+1}/{n_iters}  {elapsed:.0f}s  "
                  f"rate={(k+1)/elapsed:.2f}/s",
                  flush=True)
    return obs, null


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--layers", default="0,4,8,12,16,20,24,28,31")
    p.add_argument("--methods", default="lr,md")
    p.add_argument("--judges", default="qwen,llama")
    p.add_argument("--out", default=f"{ROOT}/experiments/steering_v3/layer_sweep/null_cross_concept_cosines.json")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    methods = args.methods.split(",")
    judges = args.judges.split(",")

    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))

    scores_paths = {
        "qwen": f"{ROOT}/working/uplift/stage_b_scores.jsonl",
        "llama": f"{ROOT}/working/uplift/stage_b_scores_llama.jsonl",
    }

    rand_sd_floor = 1.0 / np.sqrt(4096)
    print(f"\nN iters per cell: {args.n_iters}")
    print(f"4096-dim random-pair 1σ floor: ±{rand_sd_floor:.4f}\n")
    print(f"{'L':>3}  {'method':>6}  {'judge':>5}  "
          f"{'observed':>10}  {'null mean':>10} {'null sd':>9}  "
          f"{'null|cos| q95':>13}  {'pct':>6}  verdict")
    print("-" * 100)

    rng_master = np.random.default_rng(SEED)
    results = []

    # Cache per-layer data to avoid reloading 6.5 GB of pool files per layer
    for L in layers:
        print(f"\n=== L{L} ===", flush=True)
        t_layer = time.time()
        X_c, y_c = load_compliance(L, sb_keys)
        print(f"  X_comp shape: {X_c.shape}  y_pos: {y_c.sum()}  load: {time.time()-t_layer:.1f}s", flush=True)
        # Pre-load harm residuals once per layer (same residuals; only labels change with judge)
        X_h_q, y_h_q = load_harm(L, scores_paths["qwen"])
        X_h_l, y_h_l = load_harm(L, scores_paths["llama"])
        # X_h_q and X_h_l should be identical (same convs, same residual extraction)
        print(f"  X_harm_Q: {X_h_q.shape}  y_pos_Q: {y_h_q.sum()}", flush=True)
        print(f"  X_harm_L: {X_h_l.shape}  y_pos_L: {y_h_l.sum()}", flush=True)

        for method in methods:
            for judge in judges:
                X_h, y_h = (X_h_q, y_h_q) if judge == "qwen" else (X_h_l, y_h_l)
                t0 = time.time()
                seed = int(rng_master.integers(2**31))
                rng = np.random.default_rng(seed)
                obs, null = run_null(X_h, y_h, X_c, y_c, method,
                                     args.n_iters, rng)
                null_mean = float(null.mean())
                null_sd = float(null.std())
                q95 = float(np.percentile(np.abs(null), 95))
                q99 = float(np.percentile(np.abs(null), 99))
                pct = float(100 * (np.abs(null) < abs(obs)).mean())
                verdict = "real" if pct > 95 else ("borderline" if pct > 90 else "null")
                print(f"  L{L:<2}  {method:>6}  {judge:>5}  "
                      f"{obs:+10.4f}  {null_mean:+10.4f} {null_sd:9.4f}  "
                      f"{q95:13.4f}  {pct:>5.1f}%  {verdict}  "
                      f"({time.time()-t0:.1f}s)",
                      flush=True)
                results.append({
                    "layer": L, "method": method, "judge": judge,
                    "observed_cos": obs,
                    "null_mean": null_mean, "null_sd": null_sd,
                    "null_abs_q95": q95, "null_abs_q99": q99,
                    "percentile_in_null": pct, "verdict": verdict,
                    "n_iters": args.n_iters,
                })
        del X_c, y_c, X_h_q, y_h_q, X_h_l, y_h_l
        gc.collect()
        with open(args.out, "w") as f:
            json.dump({"rand_sd_floor": rand_sd_floor,
                       "per_cell": results,
                       "n_iters_per_cell": args.n_iters}, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
