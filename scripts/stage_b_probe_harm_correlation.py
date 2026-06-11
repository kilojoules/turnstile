"""Stage B analysis: does the per-turn-label compliance probe predict
real-world harm severity (Stage B Likert), or is it decoupled?

Pipeline:
  1. Match each Stage B record (n=600) to pooled_hs HS at L16 at the
     'rated' turn — breach turn for wins, last turn for losses.
  2. Fit logistic regression on per-turn rows from non-Stage-B
     conversations (no leakage), predict probe score on Stage B HS.
  3. Test A (pooled, wins-only n=289):
       AUC for high-harm (Likert >= 3) vs low-harm (Likert <= 2),
       using probe score as predictor.
       AUC ~ 0.5 => decoupling; > 0.5 => probe tracks harm severity.
  4. Test B (per-category, n ~ 60 each):
       Spearman rho between probe score and Stage B Likert rating.
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
POOL = f"{ROOT}/experiments/pooled_hs"
OUT = f"{ROOT}/experiments/outcome_probe_v1"
FIG = f"{ROOT}/figures"
LAYER = 16


def load_pool(layer):
    """Returns dict {(src, round, idx): (hs_per_turn, turn_of_breach, label,
    category, num_turns)}. Each hs_per_turn is shape (5, 4096)."""
    out = {}
    for sdir in sorted(glob.glob(f"{POOL}/*/")):
        src = os.path.basename(sdir.rstrip("/"))
        for path in sorted(glob.glob(f"{sdir}/round_*.pt")):
            rnum = int(os.path.basename(path).replace("round_", "")
                       .replace(".pt", ""))
            data = torch.load(path, weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            for i, (lab, tob, cat) in enumerate(
                    zip(data["labels"].tolist(),
                        data["turns_of_breach"],
                        data["categories"])):
                out[(src, rnum, i)] = {
                    "hs": arr[i],   # (5, 4096)
                    "label": bool(lab),
                    "turn_of_breach": tob,
                    "category": cat,
                }
    return out


def main():
    print(f"Loading pool at L{LAYER}...", flush=True)
    pool = load_pool(LAYER)
    print(f"  {len(pool)} pool records", flush=True)

    print("Loading Stage B scores...", flush=True)
    stage_b = []
    with open(f"{ROOT}/working/uplift/stage_b_scores.jsonl") as f:
        for line in f:
            stage_b.append(json.loads(line))

    # Stage B records carry conv_id 'src/rN/iM' but not the explicit
    # (source, round, idx) — recover from stage_b_candidates which has it.
    cand_meta = {}
    with open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl") as f:
        for line in f:
            r = json.loads(line)
            key = (r["source"], r["round"], r["idx"])
            cand_meta[key] = r
    # Stage B uses conv_id; reconstruct from src/rN/iM
    stage_b_keyed = []
    for r in stage_b:
        cid = r["conv_id"]
        # parse 'source/rN/iM'
        parts = cid.split("/")
        if len(parts) != 3:
            continue
        src = parts[0]
        rnum = int(parts[1].lstrip("r"))
        idx = int(parts[2].lstrip("i"))
        key = (src, rnum, idx)
        if not r.get("parse_ok"):
            continue
        if key not in pool:
            continue
        cand = cand_meta.get(key)
        if cand is None:
            continue
        # find conversation, work out which turn was rated
        conv = cand["conversation"]
        asst_turns = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        num_turns = len(asst_turns)
        if num_turns == 0:
            continue
        if r["unsafe"] and r.get("turn_of_breach") is not None:
            tob = int(r["turn_of_breach"])
            if tob < num_turns:
                rated_turn = tob
            else:
                rated_turn = num_turns - 1
        else:
            # for losses, run_judge_b extracts last assistant response
            rated_turn = num_turns - 1
        stage_b_keyed.append({
            "key": key, "rated_turn": rated_turn,
            "unsafe": r["unsafe"],
            "harm": r["rating"],
            "category": r["category"],
        })

    sb_keys = set(r["key"] for r in stage_b_keyed)
    print(f"  Stage B records matched to pool: {len(stage_b_keyed)} "
          f"(n_wins={sum(r['unsafe'] for r in stage_b_keyed)})", flush=True)

    # ----- build training set from non-Stage-B convs -----
    # per-turn label: did this turn breach?
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
    X_tr = np.stack(X_tr)
    y_tr = np.array(y_tr)
    print(f"  training rows (per-turn, non-Stage-B convs): {len(y_tr)}, "
          f"positives: {int(y_tr.sum())}", flush=True)

    # ----- evaluate Stage B HS -----
    X_te = np.stack([pool[r["key"]]["hs"][r["rated_turn"]]
                     for r in stage_b_keyed])
    print(f"  test rows (Stage B): {len(X_te)}", flush=True)

    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    probe_scores = clf.predict_proba(X_te)[:, 1]
    for r, s in zip(stage_b_keyed, probe_scores):
        r["probe_score"] = float(s)

    # ----- Test A: within-wins pooled, harm-binarized AUC -----
    wins = [r for r in stage_b_keyed if r["unsafe"]]
    high = np.array([r["harm"] >= 3 for r in wins], dtype=bool)
    low = ~high
    probe_w = np.array([r["probe_score"] for r in wins])
    auc_harm_within_wins = (
        roc_auc_score(high.astype(int), probe_w)
        if high.sum() > 0 and low.sum() > 0 else None
    )
    rho_all, p_rho = spearmanr([r["harm"] for r in wins],
                               [r["probe_score"] for r in wins])

    # also: within all 600 (wins + losses)
    rho600, p600 = spearmanr([r["harm"] for r in stage_b_keyed],
                             [r["probe_score"] for r in stage_b_keyed])

    # ----- Test B: per-category Spearman -----
    by_cat = defaultdict(list)
    for r in stage_b_keyed:
        by_cat[r["category"]].append(r)

    per_cat = {}
    for cat, rs in sorted(by_cat.items()):
        ws = [r for r in rs if r["unsafe"]]
        ls = [r for r in rs if not r["unsafe"]]
        rho_w, p_w = (spearmanr([r["harm"] for r in ws],
                                 [r["probe_score"] for r in ws])
                      if len(ws) >= 3 else (np.nan, np.nan))
        rho_a, p_a = spearmanr([r["harm"] for r in rs],
                                [r["probe_score"] for r in rs])
        # also: probe AUC for high-vs-low harm within wins
        if len(ws) >= 4:
            high_w = np.array([r["harm"] >= 3 for r in ws], dtype=bool)
            if high_w.sum() > 0 and (~high_w).sum() > 0:
                auc_w = roc_auc_score(high_w.astype(int),
                                      [r["probe_score"] for r in ws])
            else:
                auc_w = np.nan
        else:
            auc_w = np.nan
        per_cat[cat] = {
            "n_total": len(rs), "n_wins": len(ws), "n_losses": len(ls),
            "wins_high_harm": sum(1 for r in ws if r["harm"] >= 3),
            "wins_low_harm": sum(1 for r in ws if r["harm"] <= 2),
            "spearman_rho_wins_harm_vs_probe": rho_w,
            "spearman_p_wins": p_w,
            "spearman_rho_all_harm_vs_probe": rho_a,
            "spearman_p_all": p_a,
            "auc_high_vs_low_harm_within_wins": auc_w,
        }

    summary = {
        "layer": LAYER,
        "n_matched": len(stage_b_keyed),
        "n_wins": sum(r["unsafe"] for r in stage_b_keyed),
        "n_losses": sum(not r["unsafe"] for r in stage_b_keyed),
        "auc_high_vs_low_harm_within_wins_POOLED": auc_harm_within_wins,
        "n_wins_high_harm": int(high.sum()),
        "n_wins_low_harm": int(low.sum()),
        "spearman_rho_pooled_wins": rho_all, "spearman_p_pooled_wins": p_rho,
        "spearman_rho_pooled_all600": rho600, "spearman_p_pooled_all600": p600,
        "per_category": per_cat,
    }
    with open(f"{OUT}/stage_b_probe_harm.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote {OUT}/stage_b_probe_harm.json")

    # ----- print summary -----
    print()
    print(f"==== POOLED ====")
    print(f"  Test A — Probe AUC for high-harm vs low-harm WITHIN WINS:")
    print(f"    AUC = {auc_harm_within_wins:.3f}  "
          f"(n_high={int(high.sum())}, n_low={int(low.sum())})")
    print(f"  Test (continuous) — Spearman rho(probe, harm), WINS only (n={len(wins)}):")
    print(f"    rho = {rho_all:+.3f}, p = {p_rho:.3f}")
    print(f"  Spearman rho(probe, harm), ALL 600:")
    print(f"    rho = {rho600:+.3f}, p = {p600:.3f}")
    print()
    print(f"==== PER-CATEGORY ====")
    print(f"{'category':<30}  {'AUC hi/lo':>10}  {'rho wins':>9}  {'rho all':>9}  n_w(h/l)")
    for cat in sorted(per_cat, key=lambda c: -per_cat[c]['n_wins']):
        d = per_cat[cat]
        auc_w = d['auc_high_vs_low_harm_within_wins']
        rw = d['spearman_rho_wins_harm_vs_probe']
        ra = d['spearman_rho_all_harm_vs_probe']
        h = d['wins_high_harm']; l = d['wins_low_harm']
        auc_str = f"{auc_w:.2f}" if not np.isnan(auc_w) else "n/a"
        rw_str = f"{rw:+.2f}" if not np.isnan(rw) else "n/a"
        print(f"  {cat:<28}  {auc_str:>10}  {rw_str:>9}  {ra:+.2f}     "
              f"{d['n_wins']} ({h}/{l})")


if __name__ == "__main__":
    main()
