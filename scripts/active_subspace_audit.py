#!/usr/bin/env python
"""Active subspace analysis of the audit MLP probe from compliance_obliteration_v1.

Computes the gradient covariance C = E[ ∇_h φ(h) ∇_h φ(h)^T ] over a set of
L16 residuals (read from experiments/pooled_hs/), eigendecomposes it (dual
form: SVD of the centered gradient matrix), then projects each residual onto
the top-2 eigenvectors and scatters colored by per-turn compliance label.

The residuals are PRE-obliteration (the audit probe is applied to base-model
L16 residuals). The audit probe was trained on post-obliteration residuals,
so this shows where the audit probe's sensitivity axes land in the original
L16 space. To do this cleanly on POST-obliteration residuals, swap the
loader to do forward passes through base + LoRA.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


POOL_SOURCES = [
    "selfplay_jbb_v1", "stealth_s42", "stealth_hard_s456",
    "control_s42", "control_hard_s456",
    "authority_dpo", "reward_dpo",
    "incrementalism_dpo", "urgency_dpo",
    "frozen_v1", "urgency_v1", "stealth_jbb_v1",
]


class AuditProbe(nn.Module):
    """Mirrors scripts/train_compliance_obliteration.py:AuditProbe (256-256-1 MLP)."""
    def __init__(self, dim=4096, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_residuals_and_labels(root, sources, labels_path, layer=16):
    """Join (per-turn L16 residual, per-turn JBB compliance label).

    Skips zero-padded turns (conversations shorter than 5 turns).
    """
    labels = {}
    with open(labels_path) as f:
        for line in f:
            r = json.loads(line)
            labels[(r["source"], r["round"], r["idx"])] = r.get("per_turn_unsafe") or []

    Xs, ys, srcs, turns = [], [], [], []
    for src in sources:
        srcd = Path(root) / "experiments" / "pooled_hs" / src
        if not srcd.is_dir():
            print(f"  [skip] {src}: no pooled_hs dir")
            continue
        for rf in sorted(srcd.glob("round_*.pt")):
            rnum = int(rf.stem.replace("round_", ""))
            t = torch.load(rf, map_location="cpu", weights_only=False)
            hs = t["hidden_states_by_layer"][layer]  # (n_convs, 5, 4096)
            for ci in range(hs.shape[0]):
                per_turn = labels.get((src, rnum, ci))
                if per_turn is None:
                    continue
                for ti, lab in enumerate(per_turn[:5]):
                    h = hs[ci, ti]
                    if h.norm().item() < 1e-3:
                        continue  # zero-padded turn
                    Xs.append(h.float())
                    ys.append(int(bool(lab)))
                    srcs.append(src)
                    turns.append(ti)
    return torch.stack(Xs), torch.tensor(ys), srcs, turns


def gradient_matrix(probe, X, batch_size=256, device="cpu"):
    """Compute ∇_h φ(h) per row. Returns G with shape (N, D)."""
    probe = probe.to(device).eval()
    out_grads = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i: i + batch_size].to(device).clone().requires_grad_(True)
        probe.zero_grad(set_to_none=True)
        probe(xb).sum().backward()
        out_grads.append(xb.grad.detach().cpu())
    return torch.cat(out_grads, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit",
                    default="experiments/compliance_obliteration_v1/audit_probe.pt")
    ap.add_argument("--output",
                    default="experiments/compliance_obliteration_v1/active_subspace.png")
    ap.add_argument("--labels",
                    default="experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl")
    ap.add_argument("--sources", nargs="+", default=POOL_SOURCES)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max-points", type=int, default=None,
                    help="Random subsample for plotting (still uses all for SVD)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("[load] residuals + labels")
    X, y, srcs, turns = load_residuals_and_labels(
        ".", args.sources, args.labels, layer=args.layer,
    )
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"  X={tuple(X.shape)}  y={tuple(y.shape)}  "
          f"{n_pos} comply / {n_neg} refuse")

    print("[load] audit probe")
    probe = AuditProbe(dim=X.shape[1], hidden=256)
    sd = torch.load(args.audit, map_location="cpu", weights_only=False)
    probe.load_state_dict(sd)

    print("[grad] computing ∇_h φ(h) for each h")
    G = gradient_matrix(probe, X)
    print(f"  G={tuple(G.shape)}")

    print("[svd] eigendecomposing gradient covariance")
    # Dual: top right singular vectors of centered G = top eigenvectors of G^T G
    Gc = G - G.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Gc, full_matrices=False)
    eigvals = (S ** 2) / max(1, Gc.shape[0] - 1)
    total = eigvals.sum().item()
    print(f"  top-10 eigvals (raw):   {[round(float(v), 3) for v in eigvals[:10]]}")
    print(f"  top-10 eigvals (frac):  "
          f"{[round(float(v) / total * 100, 2) for v in eigvals[:10]]}%")
    print(f"  cum var top-2 = {eigvals[:2].sum()/total*100:.1f}%, "
          f"top-5 = {eigvals[:5].sum()/total*100:.1f}%, "
          f"top-10 = {eigvals[:10].sum()/total*100:.1f}%")

    v1, v2 = Vh[0], Vh[1]
    phi1 = (X @ v1).numpy()
    phi2 = (X @ v2).numpy()
    y_np = y.numpy()

    # Subsample for plotting if requested
    if args.max_points and X.shape[0] > args.max_points:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(X.shape[0], args.max_points, replace=False)
        phi1, phi2, y_np = phi1[idx], phi2[idx], y_np[idx]

    refuse = y_np == 0
    comply = y_np == 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: φ_1 vs φ_2 scatter colored by refusal
    ax = axes[0]
    ax.scatter(phi1[refuse], phi2[refuse], s=6, alpha=0.4, c="#1f77b4",
               label=f"refuse (n={int(refuse.sum())})", edgecolors="none")
    ax.scatter(phi1[comply], phi2[comply], s=6, alpha=0.4, c="#d62728",
               label=f"comply (n={int(comply.sum())})", edgecolors="none")
    ax.set_xlabel(f"$\\varphi_1$  ({eigvals[0]/total*100:.1f}% grad var)")
    ax.set_ylabel(f"$\\varphi_2$  ({eigvals[1]/total*100:.1f}% grad var)")
    ax.set_title("Active subspace of audit MLP\n"
                 "(audit AUC≈0.58 on post-obliteration L16; "
                 "projection of PRE-obliteration L16 residuals)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Right: eigenvalue scree
    ax = axes[1]
    k = min(20, len(eigvals))
    ax.bar(range(1, k + 1), (eigvals[:k] / total * 100).numpy(),
           color="#444", edgecolor="black")
    ax.set_xlabel("eigenvector index")
    ax.set_ylabel("% gradient variance")
    ax.set_title(f"Scree (top {k} of {len(eigvals)})")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_xticks(range(1, k + 1, 2))

    fig.suptitle("Active subspace analysis of compliance_obliteration_v1 audit probe",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[done] -> {args.output}")

    # Also dump the numpy arrays so you can re-plot/zoom without re-running
    out_npz = args.output.replace(".png", ".npz")
    np.savez(out_npz, phi1=phi1, phi2=phi2, y=y_np,
             eigvals=eigvals.numpy(), v1=v1.numpy(), v2=v2.numpy())
    print(f"[done] arrays -> {out_npz}")


if __name__ == "__main__":
    main()
