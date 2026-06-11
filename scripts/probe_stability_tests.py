"""Phase 1 local tests for the probe-direction reliability question.

1. Seed stability: refit LR-compliance with 5 random downsamples; report
   pairwise cosines + cosine vs mean-diff.

2. Third extraction methods on both compliance and harm probes:
   - LR-L2 (current default)
   - Mean-difference (current alternative)
   - PCA-within-positive (top PC of mean-centered positive-class HS)
   - LR-L1 (sparse linear classifier; different geometric prior)

3. Per-layer cosine sweep: cos(LR, mean-diff) at L4, L8, L12, L16, L20,
   L24, L28, L31 — for both concepts.

4. ASR-vs-Likert decoupling at negative α_h: distinguish "suppresses harm
   content" from "amplifies refusal" using existing sweep_judged data.

Output: experiments/steering_v3/probe_stability_tests.json
        plus saves v_comp_downsampledLR_L16.pt and v_random_L16.pt
        for Phase 2 GPU experiments.
"""
import gc, glob, json, math, os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/steering_v3"
LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]
HARM_THRESH = 4
SEED = 42


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_compliance_data(layer, sb_keys):
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
                if (src, rnum, i) in sb_keys: continue
                breach = bool(labels[i])
                t_star = tobs[i] if breach else None
                if breach and t_star is None: continue
                t_max = 4 if not breach else int(t_star)
                for t in range(t_max + 1):
                    X.append(arr[i, t])
                    y.append(1 if (breach and t == t_star) else 0)
            del data, arr; gc.collect()
    return np.stack(X), np.array(y)


def load_harm_data(layer):
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
                key = (src, rnum, i)
                if key in win_keys:
                    hs_by_key[key] = arr[i]
            del data, arr; gc.collect()
    X = np.stack([hs_by_key[r["key"]][r["rated_turn"]] for r in wins])
    y = np.array([r["harm"] >= HARM_THRESH for r in wins], dtype=int)
    return X, y


def extract_lr_l2(X, y, C=1.0, balanced=True, seed=SEED):
    if balanced:
        rng = np.random.default_rng(seed)
        n_pos = int(y.sum()); n_neg = int(len(y)-n_pos)
        if n_neg > n_pos:
            neg_keep = rng.choice(np.where(y==0)[0], size=n_pos, replace=False)
            keep = np.concatenate([np.where(y==1)[0], neg_keep])
            X, y = X[keep], y[keep]
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    return normalize(clf.coef_.ravel().astype(np.float32))


def extract_lr_l1(X, y, C=1.0, balanced=True, seed=SEED):
    if balanced:
        rng = np.random.default_rng(seed)
        n_pos = int(y.sum()); n_neg = int(len(y)-n_pos)
        if n_neg > n_pos:
            neg_keep = rng.choice(np.where(y==0)[0], size=n_pos, replace=False)
            keep = np.concatenate([np.where(y==1)[0], neg_keep])
            X, y = X[keep], y[keep]
    clf = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=2000)
    clf.fit(X, y)
    return normalize(clf.coef_.ravel().astype(np.float32))


def extract_meandiff(X, y):
    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    return normalize((mu_pos - mu_neg).astype(np.float32))


def extract_pca_within_pos(X, y):
    """Top PC of mean-centered positive-class features."""
    Xp = X[y == 1]
    mu = Xp.mean(axis=0)
    Xc = Xp - mu
    pca = PCA(n_components=1)
    pca.fit(Xc)
    return normalize(pca.components_[0].astype(np.float32))


def main():
    # Stage-B exclusion (same as everywhere)
    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))

    results = {}

    # ------------------------------------------------------------------
    # Test 1: Seed stability of LR-compliance (downsampled negatives)
    # ------------------------------------------------------------------
    print("=== Test 1: Seed stability of downsampled LR-compliance ===")
    X_c, y_c = load_compliance_data(16, sb_keys)
    n_pos = int(y_c.sum())
    print(f"  L16 compliance: n_total={len(y_c)}, n_pos={n_pos}")

    seed_vecs = []
    for seed in [11, 22, 33, 44, 55]:
        rng = np.random.default_rng(seed)
        neg_keep = rng.choice(np.where(y_c == 0)[0], size=n_pos, replace=False)
        keep = np.concatenate([np.where(y_c == 1)[0], neg_keep])
        Xb, yb = X_c[keep], y_c[keep]
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Xb, yb)
        seed_vecs.append(normalize(clf.coef_.ravel().astype(np.float32)))
    # pairwise cosines
    pw = []
    for i in range(len(seed_vecs)):
        for j in range(i+1, len(seed_vecs)):
            pw.append(float(seed_vecs[i] @ seed_vecs[j]))
    print(f"  pairwise cos (5 seeds, 10 pairs): "
          f"mean={np.mean(pw):.3f}, min={min(pw):.3f}, max={max(pw):.3f}")
    # mean direction across seeds, vs mean-diff
    v_md_c = extract_meandiff(X_c, y_c)
    cos_seeds_md = [float(v @ v_md_c) for v in seed_vecs]
    print(f"  cos(each seed, mean-diff): "
          f"mean={np.mean(cos_seeds_md):.3f}, range=[{min(cos_seeds_md):.3f}, "
          f"{max(cos_seeds_md):.3f}]")
    results["seed_stability"] = {
        "pairwise_cos_mean": float(np.mean(pw)),
        "pairwise_cos_min": float(min(pw)),
        "pairwise_cos_max": float(max(pw)),
        "cos_vs_meandiff_mean": float(np.mean(cos_seeds_md)),
        "cos_vs_meandiff_min": float(min(cos_seeds_md)),
        "cos_vs_meandiff_max": float(max(cos_seeds_md)),
    }

    # Save downsampled-LR direction (seed 42) for Phase 2 GPU work
    v_c_lr_down = extract_lr_l2(X_c, y_c, balanced=True, seed=42)
    torch.save(torch.tensor(v_c_lr_down), f"{OUT}/v_comp_lr_downsampled_L16.pt")
    print(f"  saved v_comp_lr_downsampled_L16.pt")

    # ------------------------------------------------------------------
    # Test 2: Third / fourth extraction methods, both concepts
    # ------------------------------------------------------------------
    print("\n=== Test 2: Multi-method extraction at L16 ===")
    # Compliance: use balanced data (already loaded; downsample per method)
    methods_c = {
        "lr_l2_balanced": extract_lr_l2(X_c, y_c, balanced=True, seed=42),
        "meandiff": v_md_c,
        "pca_within_pos": extract_pca_within_pos(X_c, y_c),
        "lr_l1_balanced": extract_lr_l1(X_c, y_c, balanced=True, seed=42),
    }
    print("\n  Compliance (L16):")
    keys = list(methods_c.keys())
    print(f"    {'method1':<22} {'method2':<22}  cosine")
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            c = float(methods_c[keys[i]] @ methods_c[keys[j]])
            print(f"    {keys[i]:<22} {keys[j]:<22}  {c:+.3f}")

    # Harm
    print("\n  Loading harm data L16...")
    X_h, y_h = load_harm_data(16)
    print(f"  L16 harm: n_total={len(y_h)}, n_high={int(y_h.sum())}, n_low={int(len(y_h)-y_h.sum())}")
    v_md_h = extract_meandiff(X_h, y_h)
    methods_h = {
        "lr_l2_balanced": extract_lr_l2(X_h, y_h, balanced=True, seed=42),
        "meandiff": v_md_h,
        "pca_within_pos": extract_pca_within_pos(X_h, y_h),
        "lr_l1_balanced": extract_lr_l1(X_h, y_h, balanced=True, seed=42),
    }
    print("\n  Harm (L16):")
    print(f"    {'method1':<22} {'method2':<22}  cosine")
    cosines_harm = {}
    cosines_comp = {}
    keys = list(methods_h.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            c_h = float(methods_h[keys[i]] @ methods_h[keys[j]])
            c_c = float(methods_c[keys[i]] @ methods_c[keys[j]])
            print(f"    {keys[i]:<22} {keys[j]:<22}  harm={c_h:+.3f}   comp={c_c:+.3f}")
            cosines_harm[f"{keys[i]}_vs_{keys[j]}"] = c_h
            cosines_comp[f"{keys[i]}_vs_{keys[j]}"] = c_c
    results["multi_method"] = {"harm": cosines_harm, "compliance": cosines_comp}

    del X_h, y_h, X_c, y_c; gc.collect()

    # ------------------------------------------------------------------
    # Test 3: Per-layer cos(LR-L2 balanced, mean-diff) for both concepts
    # ------------------------------------------------------------------
    print("\n=== Test 3: Per-layer LR-L2 vs mean-diff cosine ===")
    per_layer = {}
    for L in LAYERS:
        Xc, yc = load_compliance_data(L, sb_keys)
        v_md_c = extract_meandiff(Xc, yc)
        v_lr_c = extract_lr_l2(Xc, yc, balanced=True, seed=42)
        cos_c = float(v_md_c @ v_lr_c)
        del Xc, yc; gc.collect()

        Xh, yh = load_harm_data(L)
        v_md_h = extract_meandiff(Xh, yh)
        v_lr_h = extract_lr_l2(Xh, yh, balanced=True, seed=42)
        cos_h = float(v_md_h @ v_lr_h)
        del Xh, yh; gc.collect()

        per_layer[f"L{L}"] = {"compliance": cos_c, "harm": cos_h}
        print(f"  L{L:>2}: compliance LR↔meandiff = {cos_c:+.3f}    "
              f"harm LR↔meandiff = {cos_h:+.3f}")
    results["per_layer_cos_lr_vs_meandiff"] = per_layer

    # ------------------------------------------------------------------
    # Test 6: ASR-vs-Likert decoupling at negative α_h (existing sweep)
    # ------------------------------------------------------------------
    print("\n=== Test 6: ASR vs Likert at negative α_h (existing sweep_judged data) ===")
    judged = []
    for path in [f"{ROOT}/experiments/steering_v3/sweep_judged.jsonl",
                 f"{ROOT}/experiments/steering_v3/sweep_new_judged.jsonl"]:
        for line in open(path):
            judged.append(json.loads(line))
    wins = [r for r in judged if r.get("prompt_type")=="win" and r.get("layer",16)==16
            and abs(r.get("alpha_c", 0)) < 1e-9]
    print(f"  wins with α_c=0 at L16: {len(wins)}")
    print(f"  {'α_h':>8}  {'n':>3}  {'ASR':>7}  {'meanHarm':>10}")
    decoupling = {}
    for ah in sorted({round(r["alpha_h"], 2) for r in wins}):
        rs = [r for r in wins if abs(r["alpha_h"] - ah) < 1e-9]
        comps = [bool(r["judge_compliance_unsafe"]) for r in rs
                 if r["judge_compliance_unsafe"] is not None]
        harms = [r["judge_harm_likert"] for r in rs
                 if r["judge_harm_likert"] is not None]
        asr = 100*sum(comps)/max(1,len(comps))
        mh = float(np.mean(harms)) if harms else float("nan")
        print(f"   {ah:+5.2f}  {len(rs):>3}  {asr:>5.1f}%   {mh:>5.2f}")
        decoupling[f"alpha_h_{ah:+.2f}"] = {"n": len(rs), "asr_pct": asr, "mean_harm": mh}
    results["asr_vs_likert_decoupling"] = decoupling

    # ------------------------------------------------------------------
    # Save random direction for Phase 2 control
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    v_rand = rng.standard_normal(4096).astype(np.float32)
    v_rand /= np.linalg.norm(v_rand)
    torch.save(torch.tensor(v_rand), f"{OUT}/v_random_L16.pt")
    print(f"\n  saved v_random_L16.pt for control sweep")

    with open(f"{OUT}/probe_stability_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}/probe_stability_tests.json")


if __name__ == "__main__":
    main()
