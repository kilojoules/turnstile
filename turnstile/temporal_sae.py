"""Temporal Sparse Autoencoder (T-SAE) for multi-turn safety analysis.

Adapts Bhalla et al. (ICLR 2026) "Temporal Sparse Autoencoders" from
token-level to turn-level temporal consistency. A Matryoshka feature
partition separates high-level (conversation trajectory) features from
low-level (per-turn content) features. A contrastive loss encourages
high-level features to capture smooth temporal dynamics.

Key adaptation: the original T-SAE enforces consistency between adjacent
TOKENS within a sequence. We enforce consistency between adjacent TURNS
within a multi-turn conversation. The hypothesis is that safety erosion
in multi-turn attacks is a gradual process — high-level features should
capture this trajectory while low-level features capture individual
turn semantics.

Architecture follows Bhalla et al. exactly:
  - BatchTopK activation (k=20) instead of ReLU + L1
  - Matryoshka reconstruction: L_H (high-level) + L_L (all features)
  - Bidirectional InfoNCE contrastive loss on high-level features
  - 20/80 feature split (high/low)

Reference: https://arxiv.org/abs/2511.05541

Usage:
  python -m turnstile.temporal_sae \
      --hidden-states-dir experiments/frozen_v1/hidden_states \
      --output-dir results/tsae/frozen_v1
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BatchTopK activation (Gao et al. 2024, used by Bhalla et al.)
# ---------------------------------------------------------------------------

def _batch_topk(x, k):
    """Keep top-k activations per sample, zero out the rest.

    Gradients flow only through the selected (top-k) positions.
    """
    topk_vals, topk_idx = x.topk(k, dim=-1)
    mask = torch.zeros_like(x)
    mask.scatter_(-1, topk_idx, 1.0)
    return x * mask


# ---------------------------------------------------------------------------
# Temporal SAE
# ---------------------------------------------------------------------------

class TemporalSAE(nn.Module):
    """Matryoshka Temporal SAE with contrastive turn-level consistency.

    Feature partition:
        [0 : n_high)     — high-level: conversation dynamics, safety trajectory
        [n_high : n_feat) — low-level: per-turn content, topic

    Losses:
        L_matr (Matryoshka reconstruction):
            L_H = MSE using only high-level decoder columns
            L_L = MSE using all features
        L_contr (bidirectional InfoNCE):
            Positive: adjacent turns from same conversation
            Negative: adjacent turns from different conversations
        Sparsity: enforced by BatchTopK (no L1 term needed)
    """

    def __init__(self, d_model, n_features, n_high_level, k=20):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.n_high = n_high_level
        self.k = k

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        """Encode input → sparse features via ReLU + BatchTopK."""
        pre = self.encoder(x)
        post = F.relu(pre)
        return _batch_topk(post, self.k)

    def decode(self, f):
        """Decode features → reconstruction (all features)."""
        return self.decoder(f)

    def decode_high(self, f):
        """Decode using only high-level features (first n_high columns).

        Implements W_dec[0:h] · f[0:h](x) + b_dec from Bhalla et al. Eq. 2.
        """
        f_high = f[:, :self.n_high]
        return F.linear(
            f_high,
            self.decoder.weight[:, :self.n_high],
            self.decoder.bias,
        )

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def compute_loss(self, x_t, x_prev, alpha=1.0, temperature=0.07):
        """Compute combined Matryoshka + contrastive loss.

        Follows Bhalla et al. Equations 1-3:
          L = sum_i L_matr(x_i) + alpha * L_contr

        Args:
            x_t:     (B, d_model) hidden states at turn t
            x_prev:  (B, d_model) hidden states at turn t-1
            alpha:    weight for contrastive loss (default 1.0 per paper)
            temperature: InfoNCE temperature (lower = sharper)

        Returns:
            (total_loss, metrics_dict)
        """
        # Encode both turns
        f_t = self.encode(x_t)
        f_prev = self.encode(x_prev)

        # --- Matryoshka reconstruction (Eq. 1-2) ---
        # L_L: full reconstruction using all features
        x_hat_t = self.decode(f_t)
        x_hat_prev = self.decode(f_prev)
        l_full = (
            (x_t - x_hat_t).pow(2).sum(-1).mean()
            + (x_prev - x_hat_prev).pow(2).sum(-1).mean()
        ) / 2

        # L_H: high-level reconstruction using only first n_high features
        x_hat_t_h = self.decode_high(f_t)
        x_hat_prev_h = self.decode_high(f_prev)
        l_high = (
            (x_t - x_hat_t_h).pow(2).sum(-1).mean()
            + (x_prev - x_hat_prev_h).pow(2).sum(-1).mean()
        ) / 2

        # --- Bidirectional InfoNCE on high-level features (Eq. 3) ---
        z_t = F.normalize(f_t[:, :self.n_high], dim=-1)
        z_prev = F.normalize(f_prev[:, :self.n_high], dim=-1)

        # Similarity matrix: (B, B)
        sim = z_t @ z_prev.T / temperature

        # Bidirectional cross-entropy (CLIP-style)
        labels = torch.arange(sim.shape[0], device=sim.device)
        l_contrast = (
            F.cross_entropy(sim, labels)
            + F.cross_entropy(sim.T, labels)
        ) / 2

        # Total loss (no L1 — BatchTopK enforces sparsity)
        total = l_full + l_high + alpha * l_contrast

        metrics = {
            "total": total.item(),
            "mse_full": l_full.item(),
            "mse_high": l_high.item(),
            "contrastive": l_contrast.item(),
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_activations(x):
    """Scale so average squared L2 norm equals d_model."""
    d = x.shape[-1]
    avg_sq_norm = x.pow(2).sum(dim=-1).mean()
    scale = (d / avg_sq_norm).sqrt()
    return x * scale, scale.item()


# ---------------------------------------------------------------------------
# Data loading: build (turn_t, turn_{t-1}) pairs
# ---------------------------------------------------------------------------

def build_turn_pairs(hidden_states_dir):
    """Load per-turn hidden states and build adjacent-turn pairs.

    Args:
        hidden_states_dir: directory with round_*.pt files from the loop

    Returns:
        pairs_t:       (N, d_model) — hidden states at turn t
        pairs_prev:    (N, d_model) — hidden states at turn t-1
        pair_labels:   (N,) bool — True if conversation was unsafe
        pair_conv_ids: (N,) int — conversation index for each pair
    """
    pairs_t, pairs_prev = [], []
    pair_labels, pair_conv_ids = [], []
    conv_id = 0

    round_files = sorted(
        f for f in os.listdir(hidden_states_dir)
        if f.startswith("round_") and f.endswith(".pt")
    )

    for fname in round_files:
        data = torch.load(
            os.path.join(hidden_states_dir, fname), weights_only=False
        )
        for hs, label in zip(data["hidden_states"], data["labels"]):
            # hs: (n_turns, d_model)
            for t in range(1, hs.shape[0]):
                pairs_t.append(hs[t])
                pairs_prev.append(hs[t - 1])
                pair_labels.append(bool(label))
                pair_conv_ids.append(conv_id)
            conv_id += 1

    return (
        torch.stack(pairs_t),
        torch.stack(pairs_prev),
        torch.tensor(pair_labels, dtype=torch.bool),
        torch.tensor(pair_conv_ids, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_temporal_sae(
    hidden_states_dir,
    output_dir,
    expansion=4,
    high_level_frac=0.2,
    k=20,
    alpha=1.0,
    temperature=0.07,
    lr=3e-4,
    steps=10000,
    batch_size=256,
):
    """Train a Temporal SAE on per-turn hidden state pairs.

    Saves T-SAE weights, normalization scale, and training metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build turn pairs
    print("  Loading turn pairs...")
    pairs_t, pairs_prev, labels, conv_ids = build_turn_pairs(hidden_states_dir)
    n_convs = conv_ids.max().item() + 1
    print(f"  {len(pairs_t)} turn pairs from {n_convs} conversations")
    print(f"  Unsafe: {labels.sum().item()}, Safe: {(~labels).sum().item()}")

    d_model = pairs_t.shape[-1]
    n_features = d_model * expansion
    n_high = int(n_features * high_level_frac)

    # Normalize (scale so avg squared L2 norm = d_model)
    all_states = torch.cat([pairs_t, pairs_prev], dim=0)
    _, scale = normalize_activations(all_states)
    pairs_t_norm = pairs_t * scale
    pairs_prev_norm = pairs_prev * scale

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  T-SAE: {n_features} features "
          f"({n_high} high-level, k={k}), device={device}")

    tsae = TemporalSAE(d_model, n_features, n_high, k=k).to(device)
    optimizer = torch.optim.Adam(tsae.parameters(), lr=lr)

    losses = []
    for step in range(steps):
        idx = torch.randint(
            0, len(pairs_t_norm), (min(batch_size, len(pairs_t_norm)),)
        )
        loss, metrics = tsae.compute_loss(
            pairs_t_norm[idx].to(device),
            pairs_prev_norm[idx].to(device),
            alpha=alpha,
            temperature=temperature,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(tsae.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(metrics)
        if (step + 1) % 500 == 0:
            print(f"    Step {step+1}/{steps} | "
                  f"Total: {metrics['total']:.4f} "
                  f"(MSE_full: {metrics['mse_full']:.4f}, "
                  f"MSE_high: {metrics['mse_high']:.4f}, "
                  f"Contrast: {metrics['contrastive']:.4f})")

    tsae = tsae.cpu()

    # Save T-SAE
    save_data = {
        "tsae_state_dict": tsae.state_dict(),
        "d_model": d_model,
        "n_features": n_features,
        "n_high_level": n_high,
        "k": k,
        "alpha": alpha,
        "temperature": temperature,
        "normalize_scale": scale,
        "losses": losses,
    }
    torch.save(save_data, os.path.join(output_dir, "temporal_sae.pt"))

    # Save turn pairs + labels for probe fitting
    torch.save({
        "pairs_t": pairs_t,
        "pairs_prev": pairs_prev,
        "labels": labels,
        "conv_ids": conv_ids,
    }, os.path.join(output_dir, "turn_pairs.pt"))

    print(f"  T-SAE saved to {output_dir}")
    return tsae, scale


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Temporal SAE on multi-turn hidden states"
    )
    parser.add_argument("--hidden-states-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--high-level-frac", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=20,
                        help="BatchTopK k (features active per sample)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Contrastive loss weight (default: 1.0)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_temporal_sae(
        args.hidden_states_dir, args.output_dir,
        expansion=args.expansion, high_level_frac=args.high_level_frac,
        k=args.k, alpha=args.alpha, temperature=args.temperature,
        steps=args.steps, lr=args.lr,
    )
