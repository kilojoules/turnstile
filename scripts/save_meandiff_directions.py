"""Compute Arditi-style mean-difference directions at L16.

v_comp_meandiff_L16 = mean(HS at breach turns) - mean(HS at safe turns)
v_harm_meandiff_L16 = mean(HS at high-harm responses) - mean(HS at low-harm)

Saves to experiments/steering_v3/ and prints cosines with the existing
LR-direction vectors for comparison.
"""
import gc, glob, json, os
import numpy as np
import torch

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
OUT = f"{ROOT}/experiments/steering_v3"
POOL = f"{ROOT}/experiments/pooled_hs"
HARM_THRESH = 4
LAYER = 16


def main():
    # ----- 1. Compliance mean-diff -----
    # For each conversation, gather HS at breach turn (pos) vs HS at safe turns (neg)
    # using per-turn label semantics, excluding Stage B convs
    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))

    pos_acc = np.zeros(4096, dtype=np.float64); n_pos = 0
    neg_acc = np.zeros(4096, dtype=np.float64); n_neg = 0
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][LAYER].numpy()
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
                    if breach and t == t_star:
                        pos_acc += arr[i, t]; n_pos += 1
                    else:
                        neg_acc += arr[i, t]; n_neg += 1
            del data, arr; gc.collect()
    mu_pos = pos_acc / n_pos
    mu_neg = neg_acc / n_neg
    md_comp = (mu_pos - mu_neg).astype(np.float32)
    nrm = np.linalg.norm(md_comp)
    v_comp_md = md_comp / nrm
    print(f"Compliance mean-diff: n_pos={n_pos}, n_neg={n_neg}, "
          f"||mean-diff||={nrm:.2f}")

    # ----- 2. Harm mean-diff (from SB wins) -----
    cand = {}
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        cand[(r["source"], r["round"], r["idx"])] = r
    wins = []
    for line in open(f"{ROOT}/working/uplift/stage_b_scores.jsonl"):
        r = json.loads(line)
        if not r.get("parse_ok"): continue
        if not r.get("unsafe"): continue
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
            rated = len(asst) - 1
        wins.append({"key": key, "rated_turn": rated, "harm": r["rating"]})

    # build (key -> hs) map for L16, restricted to win keys
    win_keys = {r["key"] for r in wins}
    hs_by_key = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][LAYER].numpy()
            for i in range(len(data["labels"])):
                src = os.path.basename(os.path.dirname(path))
                rnum = int(os.path.basename(path).replace("round_","").replace(".pt",""))
                key = (src, rnum, i)
                if key in win_keys:
                    hs_by_key[key] = arr[i]
            del data, arr; gc.collect()

    high = np.zeros(4096, dtype=np.float64); n_h = 0
    low = np.zeros(4096, dtype=np.float64); n_l = 0
    for r in wins:
        hs = hs_by_key[r["key"]][r["rated_turn"]]
        if r["harm"] >= HARM_THRESH:
            high += hs; n_h += 1
        else:
            low += hs; n_l += 1
    mu_high = high / n_h
    mu_low = low / n_l
    md_harm = (mu_high - mu_low).astype(np.float32)
    nrm_h = np.linalg.norm(md_harm)
    v_harm_md = md_harm / nrm_h
    print(f"Harm mean-diff: n_high={n_h}, n_low={n_l}, "
          f"||mean-diff||={nrm_h:.2f}")

    # ----- 3. Compare with existing LR directions -----
    v_comp_lr = torch.load(f"{OUT}/v_comp_L16.pt", weights_only=False).numpy()
    v_harm_lr = torch.load(f"{OUT}/v_harm_L16.pt", weights_only=False).numpy()
    cos_cc = float(v_comp_md @ v_comp_lr)
    cos_hh = float(v_harm_md @ v_harm_lr)
    cos_ch = float(v_comp_md @ v_harm_md)
    print(f"\ncos(v_comp_meandiff, v_comp_LR) = {cos_cc:+.4f}")
    print(f"cos(v_harm_meandiff, v_harm_LR) = {cos_hh:+.4f}")
    print(f"cos(v_comp_meandiff, v_harm_meandiff) = {cos_ch:+.4f}")
    print(f"floor: ±{1/np.sqrt(4096):.4f}")

    # ----- 4. Save -----
    torch.save(torch.tensor(v_comp_md), f"{OUT}/v_comp_meandiff_L16.pt")
    torch.save(torch.tensor(v_harm_md), f"{OUT}/v_harm_meandiff_L16.pt")
    print(f"\nSaved v_comp_meandiff_L16.pt, v_harm_meandiff_L16.pt")


if __name__ == "__main__":
    main()
