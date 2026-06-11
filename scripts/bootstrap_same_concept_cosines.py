"""Part 1.2 — same-concept method-agreement bootstrap CI.

For each (concept ∈ {harm, comp}, judge, layer):
  Resample conversations with replacement
  Refit BOTH LR and MD on the bootstrap sample
  Record cos(v_LR, v_MD)
  ≥500 draws → bootstrap mean + 95% CI

Expected:
  harm: tight at ~0.99 — methods agree on the harm direction
  compliance: low at ~0.10–0.36 — methods disagree on what the compliance
              direction is. The claim worth supporting is that the CI
              excludes high agreement (e.g. CI upper bound < ~0.5).
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
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4
SEED = 42


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_compliance(layer, sb_keys):
    """Returns (X, y, conv_ids) so we can bootstrap by conversation."""
    X, y, conv_ids = [], [], []
    cid = 0
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
                conv_id = cid; cid += 1
                for t in range(t_max + 1):
                    X.append(arr[i, t])
                    y.append(1 if (breach and t == t_star) else 0)
                    conv_ids.append(conv_id)
            del data, arr; gc.collect()
    return np.stack(X), np.array(y, dtype=np.int32), np.array(conv_ids)


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
            del data, arr; gc.collect()
    wins = [w for w in wins if w["key"] in hs_by_key]
    X = np.stack([hs_by_key[w["key"]][w["rated_turn"]] for w in wins])
    y = np.array([w["harm"] >= HARM_THRESH for w in wins], dtype=np.int32)
    return X, y


def fit_harm_lr_md(X, y):
    """Return (v_lr, v_md), both unit-normalized."""
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    v_lr = normalize(clf.coef_.ravel().astype(np.float32))
    v_md = normalize((X[y == 1].mean(0) - X[y == 0].mean(0)).astype(np.float32))
    return v_lr, v_md


def fit_comp_lr_md(X, y, rng):
    """Fit on balanced subsample. Return (v_lr, v_md)."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    sel = np.concatenate([pos_idx, keep])
    Xb, yb = X[sel], y[sel]
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(Xb, yb)
    v_lr = normalize(clf.coef_.ravel().astype(np.float32))
    v_md = normalize((Xb[yb == 1].mean(0) - Xb[yb == 0].mean(0)).astype(np.float32))
    return v_lr, v_md


def bootstrap_harm_cos(X, y, n_iters, rng):
    """Resample wins with replacement, refit LR + MD, record cos(LR, MD)."""
    n = len(y)
    cos_list = np.empty(n_iters)
    # Observed (no resample)
    v_lr_obs, v_md_obs = fit_harm_lr_md(X, y)
    obs = float(np.dot(v_lr_obs, v_md_obs))
    for k in range(n_iters):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        # Need both classes present
        if yb.sum() < 2 or (1 - yb).sum() < 2:
            cos_list[k] = np.nan
            continue
        v_lr, v_md = fit_harm_lr_md(Xb, yb)
        cos_list[k] = float(np.dot(v_lr, v_md))
    return obs, cos_list[~np.isnan(cos_list)]


def bootstrap_comp_cos(X, y, conv_ids, n_iters, rng):
    """Resample CONVERSATIONS with replacement, refit LR+MD on balanced subsample,
    record cos(LR, MD). Resampling at the conversation level prevents
    information leakage."""
    unique_convs = np.unique(conv_ids)
    n_convs = len(unique_convs)
    # observed
    v_lr_obs, v_md_obs = fit_comp_lr_md(X, y, np.random.default_rng(SEED + 7919))
    obs = float(np.dot(v_lr_obs, v_md_obs))
    cos_list = np.empty(n_iters)
    for k in range(n_iters):
        # bootstrap conv_ids
        boot_convs = rng.choice(unique_convs, size=n_convs, replace=True)
        # need to gather all rows whose conv_id is in boot_convs (with
        # multiplicity) — use a single pass via boolean masking with
        # counts, but np doesn't directly support that. Build a per-conv index
        # then take.
        # Faster: for each unique conv resampled, look up its rows.
        # Pre-build conv→indices map.
        if k == 0:
            from collections import defaultdict
            conv_to_rows = defaultdict(list)
            for r, c in enumerate(conv_ids):
                conv_to_rows[int(c)].append(r)
            for c in conv_to_rows:
                conv_to_rows[c] = np.array(conv_to_rows[c])
        rows = np.concatenate([conv_to_rows[int(c)] for c in boot_convs])
        Xb, yb = X[rows], y[rows]
        if yb.sum() < 2 or (1 - yb).sum() < 2:
            cos_list[k] = np.nan
            continue
        v_lr, v_md = fit_comp_lr_md(Xb, yb, rng)
        cos_list[k] = float(np.dot(v_lr, v_md))
    return obs, cos_list[~np.isnan(cos_list)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-iters", type=int, default=500)
    p.add_argument("--layers", default="0,4,8,12,16,20,24,28,31")
    p.add_argument("--skip-comp", action="store_true",
                   help="skip the expensive compliance bootstrap")
    p.add_argument("--out", default=f"{ROOT}/experiments/steering_v3/layer_sweep/bootstrap_same_concept.json")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))

    scores_paths = {
        "qwen": f"{ROOT}/working/uplift/stage_b_scores.jsonl",
        "llama": f"{ROOT}/working/uplift/stage_b_scores_llama.jsonl",
    }

    out = {"n_iters": args.n_iters, "per_cell": []}
    print(f"N iters per cell: {args.n_iters}\n")
    print(f"  {'concept':>7}  {'judge':>5}  {'L':>3}  "
          f"{'observed':>10}  {'boot mean':>10}  {'95% CI (low, high)':>22}  {'CI excludes 0.5':>16}")
    print("  " + "─" * 100)

    for L in layers:
        t_layer = time.time()
        # Harm both judges
        for judge in ["qwen", "llama"]:
            X_h, y_h = load_harm(L, scores_paths[judge])
            rng = np.random.default_rng(SEED + L*100 + (0 if judge=="qwen" else 1))
            t0 = time.time()
            obs, dist = bootstrap_harm_cos(X_h, y_h, args.n_iters, rng)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            excludes_half = (lo > 0.5)
            cell = {"concept": "harm", "judge": judge, "layer": L,
                    "observed": obs, "boot_mean": float(dist.mean()),
                    "ci_low": float(lo), "ci_high": float(hi),
                    "n_eff": len(dist),
                    "ci_excludes_0.5": bool(excludes_half)}
            out["per_cell"].append(cell)
            print(f"  {'harm':>7}  {judge:>5}  L{L:<2}  "
                  f"{obs:+10.4f}  {dist.mean():+10.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]      {str(excludes_half):>16}  "
                  f"({time.time()-t0:.0f}s)",
                  flush=True)
            del X_h, y_h
            gc.collect()

        if not args.skip_comp:
            X_c, y_c, conv_ids = load_compliance(L, sb_keys)
            rng = np.random.default_rng(SEED + L*100 + 2)
            t0 = time.time()
            obs, dist = bootstrap_comp_cos(X_c, y_c, conv_ids, args.n_iters, rng)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            excludes_half = (lo > 0.5)
            cell = {"concept": "comp", "judge": "llama", "layer": L,
                    "observed": obs, "boot_mean": float(dist.mean()),
                    "ci_low": float(lo), "ci_high": float(hi),
                    "n_eff": len(dist),
                    "ci_excludes_0.5": bool(excludes_half)}
            out["per_cell"].append(cell)
            print(f"  {'comp':>7}  {'llama':>5}  L{L:<2}  "
                  f"{obs:+10.4f}  {dist.mean():+10.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]      {str(excludes_half):>16}  "
                  f"({time.time()-t0:.0f}s)",
                  flush=True)
            del X_c, y_c, conv_ids
            gc.collect()

        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  (L{L} done in {time.time()-t_layer:.0f}s; saved partial)\n",
              flush=True)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
