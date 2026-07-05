#!/usr/bin/env python3
"""
harm_directions_postresponse.py

Fit LR and mean-diff harm directions using POST-RESPONSE activations from
Stage-B wins. Mean-pools the victim's response tokens at the target layer.

Motivation: post-response activations encode harm more strongly (AUC ~0.95 vs
~0.75 pre-response). Do directions derived from this richer signal transfer
back to pre-response steering?

Outputs:
  experiments/steering_v3/layer_sweep/directions_postresponse/
    v_lr_harm_L{L}.pt          (unit-norm, shape [D])
    v_md_harm_L{L}.pt          (unit-norm, shape [D])
    metadata_postresponse.json  (AUCs, sample sizes)
  experiments/postresponse_acts/
    activations_L{L}.npy        (shape [N, D], cached)
    labels.npy                  (shape [N], harm >= HARM_THRESH)

Usage (remote GPU):
  python scripts/harm_directions_postresponse.py --root /root/turnstile
  python scripts/harm_directions_postresponse.py --root /root/turnstile --skip-extract
"""
import argparse, gc, glob, json, os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

VICTIM_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HARM_THRESH = 4
SEED = 42


def normalize(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_wins(scores_path, cands_path):
    """Load Stage-B wins with Likert harm scores and full conversations."""
    cand = {}
    for line in open(cands_path):
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
        wins.append({
            "key": key,
            "rated_turn": rated,   # index into assistant turns (0-based)
            "harm": r["rating"],
            "conversation": conv,
        })
    return wins


@torch.inference_mode()
def extract_activations(wins, model, tokenizer, layer, device, batch_size=2):
    """
    For each win, run a forward pass over the conversation up to and including
    the breach-turn response, then mean-pool the response token hidden states.

    Left-padding accounting:
      enc has shape [B, T_max]. For item j with full_len actual tokens:
        - actual content starts at pad_offset = T_max - full_len
        - prefix occupies the first prefix_len of those actual tokens
        - response tokens are [pad_offset + prefix_len, T_max)
    """
    acts = []
    for start in range(0, len(wins), batch_size):
        batch = wins[start:start + batch_size]

        texts_full = []
        texts_prefix = []
        for rec in batch:
            conv = rec["conversation"]
            asst_indices = [j for j, t in enumerate(conv) if t["role"] == "assistant"]
            asst_conv_idx = asst_indices[rec["rated_turn"]]

            # Full: conversation up to and including the rated assistant turn
            msgs_full = conv[:asst_conv_idx + 1]
            # Prefix: same conversation but stopping before the response
            # (add_generation_prompt=True appends the assistant header tokens)
            msgs_prefix = conv[:asst_conv_idx]

            texts_full.append(tokenizer.apply_chat_template(
                msgs_full, tokenize=False, add_generation_prompt=False))
            texts_prefix.append(tokenizer.apply_chat_template(
                msgs_prefix, tokenize=False, add_generation_prompt=True))

        enc = tokenizer(texts_full, return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to(device)
        enc_prefix = tokenizer(texts_prefix, return_tensors="pt", padding=True,
                               truncation=True, max_length=2048)

        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer + 1]   # [B, T_max, D]

        T_max = enc["attention_mask"].shape[1]
        for j in range(len(batch)):
            full_len    = int(enc["attention_mask"][j].sum().item())
            prefix_len  = int(enc_prefix["attention_mask"][j].sum())
            pad_offset  = T_max - full_len
            resp_start  = pad_offset + prefix_len
            resp_end    = T_max

            if resp_start >= resp_end:
                resp_start = T_max - 1   # degenerate fallback: last token only

            rep = hs[j, resp_start:resp_end, :].float().mean(0)
            acts.append(rep.cpu().numpy())

        del out, hs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        done = min(start + batch_size, len(wins))
        print(f"  [{done:3d}/{len(wins)}] extracted", flush=True)

    return np.stack(acts)   # [N, D]


def cv_auc(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs_lr, aucs_md = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aucs_lr.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        v_md = normalize(X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0))
        scores = (X[te] - X[tr].mean(0)) @ v_md
        aucs_md.append(roc_auc_score(y[te], scores))
    return float(np.mean(aucs_lr)), float(np.mean(aucs_md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/julianquick/portfolio_copy/turnstile")
    ap.add_argument("--layers", default="16",
                    help="Comma-separated layer indices (e.g. 16 or 0,4,8,12,16)")
    ap.add_argument("--scores", default=None,
                    help="Path to stage_b_scores.jsonl (default: <root>/working/uplift/)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Reload cached activations instead of re-running model")
    args = ap.parse_args()

    root = args.root
    layers = [int(x) for x in args.layers.split(",")]
    scores_path = args.scores or f"{root}/working/uplift/stage_b_scores.jsonl"
    cands_path  = f"{root}/working/uplift/stage_b_candidates.jsonl"
    out_dir     = f"{root}/experiments/steering_v3/layer_sweep/directions_postresponse"
    acts_dir    = f"{root}/experiments/postresponse_acts"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(acts_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    print("\nLoading Stage-B wins...")
    wins = load_wins(scores_path, cands_path)
    y = np.array([w["harm"] >= HARM_THRESH for w in wins], dtype=int)
    print(f"  n={len(wins)}  n_pos={y.sum()}  n_neg={(1-y).sum()}")

    # ── activation extraction ─────────────────────────────────────────────────
    all_acts = {}
    need_extract = [L for L in layers
                    if not (args.skip_extract and
                            os.path.exists(f"{acts_dir}/activations_L{L}.npy"))]

    if need_extract:
        print(f"\nLoading {VICTIM_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(VICTIM_ID)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            VICTIM_ID, torch_dtype=dtype).to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  {n_params:.1f}B params  dtype={dtype}")

        for L in need_extract:
            print(f"\nExtracting L{L} post-response activations...")
            X = extract_activations(wins, model, tokenizer, L, device, args.batch_size)
            np.save(f"{acts_dir}/activations_L{L}.npy", X)
            all_acts[L] = X
            print(f"  Saved activations {X.shape} → {acts_dir}/activations_L{L}.npy")

        del model; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    np.save(f"{acts_dir}/labels.npy", y)

    for L in layers:
        if L not in all_acts:
            all_acts[L] = np.load(f"{acts_dir}/activations_L{L}.npy")
            print(f"Loaded cached L{L} activations {all_acts[L].shape}")

    # ── fit directions ────────────────────────────────────────────────────────
    meta = {
        "harm_thresh": HARM_THRESH, "layers": layers,
        "n_wins": len(wins), "n_pos": int(y.sum()), "n_neg": int((1-y).sum()),
        "scores_path": scores_path, "per_layer": {},
    }

    print(f"\n{'L':>3}  {'LR_AUC':>8}  {'MD_AUC':>8}")
    print("-" * 28)
    for L in layers:
        X = all_acts[L]
        auc_lr, auc_md = cv_auc(X, y)

        # Fit on full dataset for the saved direction vectors
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X, y)
        v_lr = normalize(clf.coef_.ravel().astype(np.float32))
        v_md = normalize((X[y == 1].mean(0) - X[y == 0].mean(0)).astype(np.float32))

        torch.save(torch.tensor(v_lr), f"{out_dir}/v_lr_harm_L{L}.pt")
        torch.save(torch.tensor(v_md), f"{out_dir}/v_md_harm_L{L}.pt")

        meta["per_layer"][f"L{L}"] = {"lr_auc": auc_lr, "md_auc": auc_md}
        print(f"L{L:<2}  {auc_lr:>8.4f}  {auc_md:>8.4f}")

    with open(f"{out_dir}/metadata_postresponse.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDirections → {out_dir}/")
    print(f"Metadata   → {out_dir}/metadata_postresponse.json")
    print("\nNext: run layer_sweep_steering.py with --directions-dir pointing here,")
    print("      using --methods lr_harm,md_harm (file names match).")


if __name__ == "__main__":
    main()
