"""Phase B (gate-failure follow-up): decompose the MD-harm direction rotation
under the Qwen→Llama label swap.

The 289 wins partition into 4 disjoint groups by (Qwen-≥4, Llama-≥4):
  A = kept-pos     (105 items)  ≥4 under both
  B = kept-neg     (143 items)  <4 under both
  C = Q_only_pos   ( 19 items)  Qwen ≥4, Llama <4   (exits)
  D = L_only_pos   ( 22 items)  Qwen <4, Llama ≥4   (entries)

So:
  v_q (Qwen MD-harm)   = normalize( μ(A∪C) − μ(B∪D) )
  v_l (Llama MD-harm)  = normalize( μ(A∪D) − μ(B∪C) )
  v_intersect          = normalize( μ(A) − μ(B) )           # judges agree
  v_disagree           = normalize( μ(D) − μ(C) )           # judges disagree
  v_perp               = normalize( v_l − (v_l·v_q) v_q )    # rotation axis

We then compute, per layer:
  - cos(v_l, v_comp_md)   the failure observed at L8-L24
  - cos(v_intersect, v_comp_md)   the agreed-on signal
  - cos(v_disagree, v_comp_md)   the disagreement signal (key question:
        does the disagreement axis align with compliance?)
  - cos(v_perp, v_comp_md)   does the new direction Llama adds to v_q
        point toward compliance? Or is it orthogonal to v_comp?
  - cos(v_l, v_q)   total rotation magnitude (already known ~0.67 at mid)
  - cos(v_intersect, v_q)   how much of v_q is reproducible without
        the disagreed-on 41 items

Plus a balanced-subsample MD variant (subsample neg class to pos size) to
test whether the failure is recipe-driven (MD without LR's balancing).

Plus an L=5 mass-shift diagnostic: does v_md fit on "Llama L=4 items that
were Qwen L=5" differ from "Llama L=4 items that were also Qwen L=4"?

Outputs per-layer cosine table + writes JSON summary.
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
SEED = 42


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_win_residuals(layer):
    """Return:
      hs[conv_id] -> per-turn HS at this layer (shape (n_turns, dim))
      wins: list of {conv_id, source, round, idx, rated_turn,
                     qwen_rating, llama_rating}
    """
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
        if not asst:
            continue
        tob = qr.get("turn_of_breach")
        if tob is not None and int(tob) < len(asst):
            rated = int(tob)
        else:
            rated = len(asst) - 1
        wins.append({
            "conv_id": cid,
            "source": cand["source"],
            "round": cand["round"],
            "idx": cand["idx"],
            "rated_turn": rated,
            "qwen_rating": qr["rating"],
            "llama_rating": llama[cid]["rating"],
        })

    # Pull residuals
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


def md_dir(X, mask_pos, mask_neg):
    """Mean-diff direction: normalize(μ_pos − μ_neg)."""
    mu_pos = X[mask_pos].mean(0)
    mu_neg = X[mask_neg].mean(0)
    return normalize((mu_pos - mu_neg).astype(np.float32))


def md_balanced(X, mask_pos, mask_neg, seed=SEED):
    """MD with neg-class subsampled to pos-class size."""
    pos_idx = np.where(mask_pos)[0]
    neg_idx = np.where(mask_neg)[0]
    rng = np.random.default_rng(seed)
    k = min(len(pos_idx), len(neg_idx))
    neg_keep = rng.choice(neg_idx, size=k, replace=False)
    return normalize((X[pos_idx].mean(0) - X[neg_keep].mean(0)).astype(np.float32))


def main():
    out_per_layer = []
    print(f"{'L':>3}  {'A':>4} {'B':>4} {'C':>4} {'D':>4}  "
          f"{'cos(v_l,vc)':>11} {'cos(v_i,vc)':>11} {'cos(v_d,vc)':>11} {'cos(v_p,vc)':>11}  "
          f"{'cos(v_l,v_q)':>12} {'cos(v_i,v_q)':>12} {'cos(v_bal,vc)':>13}")
    print("-" * 138)

    for L in LAYERS:
        X, wins = load_win_residuals(L)
        n = len(wins)
        q = np.array([w["qwen_rating"] for w in wins])
        l = np.array([w["llama_rating"] for w in wins])
        q_pos = q >= 4
        l_pos = l >= 4
        A = q_pos & l_pos
        B = (~q_pos) & (~l_pos)
        C = q_pos & (~l_pos)
        D = (~q_pos) & l_pos

        # Sanity: A+B+C+D should partition all wins
        assert (A.sum() + B.sum() + C.sum() + D.sum()) == n

        # Five MD directions
        v_q  = md_dir(X, A | C, B | D)  # Qwen MD-harm
        v_l  = md_dir(X, A | D, B | C)  # Llama MD-harm
        v_i  = md_dir(X, A, B)          # intersection: judges agree
        # disagree axis: D (Llama-pos, Qwen-neg) − C (Qwen-pos, Llama-neg)
        v_d  = md_dir(X, D, C)
        # rotation direction: component of v_l perpendicular to v_q
        v_perp_raw = v_l - float(np.dot(v_l, v_q)) * v_q
        v_perp = normalize(v_perp_raw)
        # balanced-subsample Llama MD
        v_bal = md_balanced(X, A | D, B | C)

        # Load v_comp directions (MD-comp; LR-comp also for completeness)
        v_md_comp = torch.load(f"{DIRS_OLD}/v_md_comp_L{L}.pt", weights_only=False).float().numpy()
        v_lr_comp = torch.load(f"{DIRS_OLD}/v_lr_comp_L{L}.pt", weights_only=False).float().numpy()

        cos = lambda u, w: float(np.dot(u, w))
        c_l_md  = cos(v_l, v_md_comp)
        c_i_md  = cos(v_i, v_md_comp)
        c_d_md  = cos(v_d, v_md_comp)
        c_p_md  = cos(v_perp, v_md_comp)
        c_lq    = cos(v_l, v_q)
        c_iq    = cos(v_i, v_q)
        c_bal_md = cos(v_bal, v_md_comp)

        # L=5 mass-shift diagnostic: within Llama-L=4 (which contains 99 items
        # that were Qwen-L=4 and 26 items that were Qwen-L=5), do the two
        # sub-groups produce different MD directions?
        was_5 = (q == 5) & l_pos   # were Qwen-L=5, are Llama-≥4
        was_4 = (q == 4) & l_pos   # were Qwen-L=4, are Llama-≥4
        n_was5 = int(was_5.sum()); n_was4 = int(was_4.sum())
        v_pos_was5 = X[was_5].mean(0) if n_was5 > 0 else None
        v_pos_was4 = X[was_4].mean(0) if n_was4 > 0 else None
        v_md_from_was5 = (normalize(v_pos_was5 - X[B].mean(0))
                          if v_pos_was5 is not None else None)
        v_md_from_was4 = (normalize(v_pos_was4 - X[B].mean(0))
                          if v_pos_was4 is not None else None)
        c_was4_was5 = (cos(v_md_from_was4, v_md_from_was5)
                       if v_md_from_was5 is not None and v_md_from_was4 is not None
                       else None)

        print(f"L{L:<2}  {A.sum():>4} {B.sum():>4} {C.sum():>4} {D.sum():>4}  "
              f"{c_l_md:+11.4f} {c_i_md:+11.4f} {c_d_md:+11.4f} {c_p_md:+11.4f}  "
              f"{c_lq:+12.4f} {c_iq:+12.4f} {c_bal_md:+13.4f}")

        out_per_layer.append({
            "layer": L,
            "n_A": int(A.sum()), "n_B": int(B.sum()),
            "n_C": int(C.sum()), "n_D": int(D.sum()),
            "cos_vl_vmd_comp": c_l_md,
            "cos_vintersect_vmd_comp": c_i_md,
            "cos_vdisagree_vmd_comp": c_d_md,
            "cos_vperp_vmd_comp": c_p_md,
            "cos_vl_vq": c_lq,
            "cos_vintersect_vq": c_iq,
            "cos_vbal_vmd_comp": c_bal_md,
            "cos_vl_vlr_comp": cos(v_l, v_lr_comp),
            "cos_vintersect_vlr_comp": cos(v_i, v_lr_comp),
            "cos_vdisagree_vlr_comp": cos(v_d, v_lr_comp),
            "cos_vperp_vlr_comp": cos(v_perp, v_lr_comp),
            "n_was5_to_geq4": n_was5,
            "n_was4_to_geq4": n_was4,
            "cos_was4md_vs_was5md": c_was4_was5,
        })
        del X, wins; gc.collect()

    out_path = (f"{ROOT}/experiments/steering_v3/layer_sweep/"
                "md_harm_rotation_decomp.json")
    with open(out_path, "w") as f:
        json.dump({
            "groups": {"A_kept_pos": "Qwen≥4 ∩ Llama≥4",
                       "B_kept_neg": "Qwen<4 ∩ Llama<4",
                       "C_Qwen_only_pos": "Qwen≥4 only (exits ≥4 under Llama)",
                       "D_Llama_only_pos": "Llama≥4 only (enters ≥4 under Llama)"},
            "per_layer": out_per_layer,
            "rand_sd_floor": 0.0156,
            "pass_threshold_3sigma": 0.0469,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
