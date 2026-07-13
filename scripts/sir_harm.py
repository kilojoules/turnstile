#!/usr/bin/env python3
"""
sir_harm.py — Sliced Inverse Regression of harm Likert score
              w.r.t. victim hidden states at a target layer.

Pipeline:
  1. Load judged sweep data; filter by coherence
  2. Stratify by judge_harm_likert (1–5); sample n_per_slice per level
  3. Run victim forward pass; extract mean-pooled response activations at layer L
  4. PCA (reduce to pca_dim) → SIR (solve generalized eigenproblem) → EDR dirs
  5. Save directions, projections, eigenvalues, scatter plot
"""
import argparse, json, random, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

VICTIM_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# ── helpers ───────────────────────────────────────────────────────────────────

def coh_ok(r):
    c = r.get("coherence") or {}
    return (c.get("n_tokens", 0) >= 5
            and c.get("token_unique_ratio", 0) >= 0.15
            and c.get("max_repeat", 999) <= 20)


def load_data(paths, score_key):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line.strip())
                if r.get(score_key) is not None and coh_ok(r):
                    rows.append(r)
    return rows


def stratified_sample(rows, score_key, n_per_slice, seed=42):
    rng = random.Random(seed)
    by_score = defaultdict(list)
    for r in rows:
        by_score[int(r[score_key])].append(r)
    sampled = []
    for k in sorted(by_score):
        pool = by_score[k]
        n = min(n_per_slice, len(pool))
        chosen = rng.sample(pool, n)
        sampled.extend(chosen)
        print(f"  score {k}: {len(pool):5d} available  →  {n} sampled")
    return sampled


# ── activation extraction ─────────────────────────────────────────────────────

@torch.inference_mode()
def extract_activations(rows, model, tokenizer, layer, device, batch_size=4):
    model.eval()
    acts = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]

        # Format: goal only as user turn; add_generation_prompt=True appends
        # the assistant-turn marker so the last token is where the model
        # is about to generate — the "pre-response" hidden state.
        texts = []
        for r in batch:
            msgs = [{"role": "user", "content": r["goal"]}]
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)

        out = model(**enc, output_hidden_states=True)
        # hidden_states[0] = embedding; hidden_states[L+1] = after layer L
        hs = out.hidden_states[layer + 1]   # [B, T, D]

        for j in range(len(batch)):
            # Left-padded: last attended token is the final prompt token,
            # i.e., the position from which the model would generate.
            mask = enc["attention_mask"][j].bool()
            last_tok_pos = mask.nonzero()[-1].item()
            rep = hs[j, last_tok_pos, :].float()
            acts.append(rep.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(rows))
            print(f"  [{done:3d}/{len(rows)}] extracted", flush=True)

    return np.stack(acts)   # (N, D)


# ── PCA-SIR ───────────────────────────────────────────────────────────────────

def pca_sir(X, y, n_edr=4, pca_dim=256, ridge=1e-3):
    """
    1. PCA: X → Z  (N, pca_dim)
    2. SIR: solve  B v = λ Σ v  (generalised eigenproblem in PCA space)
    3. Project EDR directions back to original D-dim space

    Returns: (dirs [D, n_edr], eigenvalues [n_edr], projections [N, n_edr], pca)
    """
    from sklearn.decomposition import PCA
    from scipy.linalg import eigh

    K = min(pca_dim, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=K, random_state=0)
    Z = pca.fit_transform(X)           # (N, K)

    levels = np.unique(y)
    N = len(y)

    # Slice conditional means → between-slice scatter B
    mu_all = Z.mean(axis=0)
    B = np.zeros((K, K))
    for lv in levels:
        mask = y == lv
        n_lv  = mask.sum()
        mu_lv = Z[mask].mean(axis=0)
        d = mu_lv - mu_all
        B += (n_lv / N) * np.outer(d, d)

    Sigma = np.cov(Z.T) + ridge * np.eye(K)

    # eigh returns ascending eigenvalues for A v = λ B v
    vals, vecs = eigh(B, Sigma)
    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]                # (K, K), columns are eigenvectors

    n_edr = min(n_edr, K)
    dirs_pca = vecs[:, :n_edr]         # (K, n_edr)

    # Back to original activation space
    dirs = pca.components_.T @ dirs_pca  # (D, n_edr)
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)

    projections = Z @ dirs_pca          # (N, n_edr)

    return dirs, vals[:n_edr], projections, pca


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plots(proj, ys, vals, out_dir):
    import matplotlib.pyplot as plt

    SCORE_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800",
                    4: "#F44336", 5: "#9C27B0"}
    n_edr = proj.shape[1]

    # ── scatter: each EDR component vs example index (sorted by harm score)
    fig, axes = plt.subplots(1, n_edr, figsize=(4.5 * n_edr, 4.2))
    if n_edr == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        for k in sorted(np.unique(ys)):
            mask = ys == k
            idxs = np.where(mask)[0]
            ax.scatter(idxs, proj[mask, i],
                       c=SCORE_COLORS.get(k, "gray"), label=f"Likert {k}",
                       alpha=0.55, s=22, edgecolors="none")
        ax.set_title(f"EDR {i+1}  λ={vals[i]:.5f}", fontsize=10)
        ax.set_xlabel("example (sorted by harm score)")
        ax.set_ylabel(f"EDR {i+1} projection")
        if i == 0:
            ax.legend(fontsize=8, markerscale=1.4)
        ax.grid(alpha=0.2, linewidth=0.4)
    fig.suptitle("PCA-SIR: harm Likert vs victim L16 hidden states",
                 fontsize=11)
    fig.tight_layout()
    p1 = out_dir / "sir_scatter.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── pairplot of EDR 1 vs EDR 2
    if n_edr >= 2:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        for k in sorted(np.unique(ys)):
            mask = ys == k
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=SCORE_COLORS.get(k, "gray"), label=f"Likert {k}",
                       alpha=0.6, s=28, edgecolors="none")
        ax.set_xlabel("EDR 1"); ax.set_ylabel("EDR 2")
        ax.set_title("SIR subspace (EDR 1 × EDR 2)"); ax.legend(fontsize=9)
        ax.grid(alpha=0.2, linewidth=0.4)
        p2 = out_dir / "sir_edr1_edr2.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── eigenvalue bar chart
    fig, ax = plt.subplots(figsize=(max(4, n_edr * 1.2), 3.5))
    ax.bar(range(1, n_edr + 1), vals, color="#3b6fb0", alpha=0.85)
    ax.set_xticks(range(1, n_edr + 1))
    ax.set_xlabel("EDR direction"); ax.set_ylabel("SIR eigenvalue")
    ax.set_title("Variance explained by each EDR direction")
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    p3 = out_dir / "sir_eigenvalues.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"Plots → {p1.name}, {p2.name if n_edr >= 2 else 'n/a'}, {p3.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["experiments/steering_v3/layer_sweep/sweep_judged.jsonl"])
    ap.add_argument("--score-key",    default="judge_harm_likert")
    ap.add_argument("--n-per-slice",  type=int,   default=80)
    ap.add_argument("--layer",        type=int,   default=16,
                    help="0-indexed transformer layer to extract from")
    ap.add_argument("--pca-dim",      type=int,   default=256)
    ap.add_argument("--n-edr",        type=int,   default=4)
    ap.add_argument("--ridge",        type=float, default=1e-3)
    ap.add_argument("--batch-size",   type=int,   default=2)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--out-dir",      default="experiments/sir_harm_v1")
    ap.add_argument("--device",       default="auto")
    ap.add_argument("--skip-extract", action="store_true",
                    help="load saved activations.npy instead of re-running model")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ── load + stratify ───────────────────────────────────────────────────────
    print("\nLoading judged data ...")
    rows = load_data(args.data, args.score_key)
    print(f"  {len(rows)} coherent rows with '{args.score_key}'")

    print("Stratified sample:")
    sampled = stratified_sample(rows, args.score_key, args.n_per_slice, args.seed)
    sampled.sort(key=lambda r: r[args.score_key])   # sort for cleaner scatter
    ys = np.array([int(r[args.score_key]) for r in sampled])
    print(f"  Total: {len(sampled)} examples across {len(np.unique(ys))} levels")

    # ── activations ───────────────────────────────────────────────────────────
    acts_path = out / "activations.npy"
    labs_path  = out / "labels.npy"

    if args.skip_extract and acts_path.exists():
        print(f"\nLoading saved activations from {acts_path}")
        X  = np.load(acts_path)
        ys = np.load(labs_path)
    else:
        print(f"\nLoading {VICTIM_ID} ({device}) ...")
        tokenizer = AutoTokenizer.from_pretrained(VICTIM_ID)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            VICTIM_ID, torch_dtype=dtype).to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  Loaded {n_params:.1f}B params  dtype={dtype}")

        print(f"\nExtracting activations at layer {args.layer} ...")
        X = extract_activations(sampled, model, tokenizer,
                                args.layer, device, args.batch_size)

        del model
        if device == "mps":   torch.mps.empty_cache()
        elif device == "cuda": torch.cuda.empty_cache()

        np.save(acts_path, X)
        np.save(labs_path, ys)
        print(f"  Saved activations {X.shape} → {acts_path}")

    # ── PCA-SIR ───────────────────────────────────────────────────────────────
    print(f"\nRunning PCA-SIR  (pca_dim={args.pca_dim}, n_edr={args.n_edr}, "
          f"ridge={args.ridge}) ...")
    dirs, vals, proj, pca = pca_sir(X, ys, args.n_edr, args.pca_dim, args.ridge)

    np.save(out / "edr_directions.npy",  dirs)
    np.save(out / "edr_eigenvalues.npy", vals)
    np.save(out / "edr_projections.npy", proj)

    print("\nEigenvalues:")
    total = vals.sum() + 1e-12
    for i, v in enumerate(vals):
        print(f"  EDR {i+1}: {v:.6f}  ({100*v/total:.1f}% of top-{args.n_edr} sum)")

    # ── plot ──────────────────────────────────────────────────────────────────
    make_plots(proj, ys, vals, out)

    # ── metadata ──────────────────────────────────────────────────────────────
    meta = {
        "data": args.data, "score_key": args.score_key,
        "n_per_slice": args.n_per_slice, "n_sampled": int(len(sampled)),
        "layer": args.layer, "pca_dim": args.pca_dim,
        "n_edr": args.n_edr, "ridge": args.ridge,
        "slice_counts": {int(k): int((ys == k).sum()) for k in np.unique(ys)},
        "eigenvalues": vals.tolist(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nAll results → {out}/")


if __name__ == "__main__":
    main()
