"""Per-turn probe scoring for multi-turn conversations (Phase 3 baseline).

Provides a baseline for comparison with the temporal SAE (Phase 4).
Each turn's victim hidden state is scored independently through
a standard SAE + logistic regression probe. This probe CANNOT detect
context-accumulation attacks that only become harmful when turns
are viewed together — motivating the temporal SAE.

Usage:
  python -m turnstile.probe \
      --hidden-states-dir experiments/frozen_v1/hidden_states \
      --output-dir results/probe/frozen_v1
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Standard SAE (same architecture as REDKWEEN's sae_analysis.py)."""

    def __init__(self, d_model, n_features, l1_coeff=5.0):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def forward(self, x):
        f = self.encode(x)
        return self.decoder(f), f

    def compute_loss(self, x):
        x_hat, f = self.forward(x)
        mse = (x - x_hat).pow(2).sum(dim=-1).mean()
        dec_norms = self.decoder.weight.norm(dim=0)
        l1 = (f * dec_norms).sum(dim=-1).mean()
        return mse + self.l1_coeff * l1, mse.item(), l1.item()


def normalize_activations(x):
    """Scale so average squared L2 norm equals d_model."""
    d = x.shape[-1]
    avg_sq_norm = x.pow(2).sum(dim=-1).mean()
    scale = (d / avg_sq_norm).sqrt()
    return x * scale, scale.item()


class PerTurnProbe:
    """SAE + logistic regression probe applied independently per turn.

    Scores each turn's hidden state independently — cannot detect
    context-accumulation attacks that only become harmful when turns
    are viewed together.
    """

    def __init__(self, sae, scale, weights, bias):
        self.sae = sae
        self.sae.eval()
        self.scale = scale
        self.w = torch.tensor(weights, dtype=torch.float32)
        self.b = bias

    @classmethod
    def from_pretrained(cls, probe_dir):
        """Load SAE + fit probe from saved artifacts."""
        sae_data = torch.load(
            os.path.join(probe_dir, "sae.pt"), weights_only=False
        )
        act_data = torch.load(
            os.path.join(probe_dir, "activations.pt"), weights_only=False
        )

        sae = SparseAutoencoder(
            sae_data["d_model"], sae_data["n_features"], sae_data["l1_coeff"]
        )
        sae.load_state_dict(sae_data["sae_state_dict"])
        sae.eval()
        scale = sae_data["normalize_scale"]

        x_norm = act_data["hidden_states"] * scale
        labels = act_data["labels"].numpy().astype(int)

        with torch.no_grad():
            features = sae.encode(x_norm).numpy()

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(features, labels)
        auc = roc_auc_score(labels, clf.predict_proba(features)[:, 1])
        print(f"  [PerTurnProbe] Loaded from {probe_dir}, train AUC: {auc:.4f}")

        return cls(sae, scale, clf.coef_[0], clf.intercept_[0])

    def score_turn(self, hidden_state):
        """Score a single turn's hidden state. Returns float in [0, 1]."""
        x = hidden_state.unsqueeze(0) * self.scale
        with torch.no_grad():
            f = self.sae.encode(x)
        logit = f @ self.w.unsqueeze(1) + self.b
        return torch.sigmoid(logit).item()

    def score_conversation(self, per_turn_hidden_states):
        """Score each turn independently. Returns (n_turns,) array."""
        x = per_turn_hidden_states * self.scale
        with torch.no_grad():
            f = self.sae.encode(x)
        logits = f @ self.w.unsqueeze(1) + self.b
        return torch.sigmoid(logits).squeeze(1).numpy()


def train_sae(hidden_states_dir, output_dir, expansion=4, l1_coeff=5.0,
              lr=3e-4, steps=10000, batch_size=256):
    """Train a standard SAE on per-turn hidden states from the loop.

    Collects all per-turn hidden states (flattened across turns and
    conversations) and trains a standard SAE, then fits a logistic probe.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all hidden states + labels
    all_states = []
    all_labels = []
    round_files = sorted(
        f for f in os.listdir(hidden_states_dir)
        if f.startswith("round_") and f.endswith(".pt")
    )

    for fname in round_files:
        data = torch.load(
            os.path.join(hidden_states_dir, fname), weights_only=False
        )
        for hs, label in zip(data["hidden_states"], data["labels"]):
            # hs: (n_turns, d_model) — flatten to individual turn samples
            for t in range(hs.shape[0]):
                all_states.append(hs[t])
                all_labels.append(int(label))

    x_raw = torch.stack(all_states)
    labels = torch.tensor(all_labels, dtype=torch.bool)
    print(f"  Collected {len(x_raw)} turn-level samples "
          f"({labels.sum()} unsafe, {(~labels).sum()} safe)")

    d_model = x_raw.shape[-1]
    n_features = d_model * expansion

    # Normalize
    x_norm, scale = normalize_activations(x_raw)

    # Train SAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = SparseAutoencoder(d_model, n_features, l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    print(f"  Training SAE: {n_features} features, {steps} steps on {device}")

    for step in range(steps):
        idx = torch.randint(0, len(x_norm), (min(batch_size, len(x_norm)),))
        batch = x_norm[idx].to(device)
        total, mse, l1 = sae.compute_loss(batch)
        total.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1}/{steps} | "
                  f"Loss: {total.item():.4f} (MSE: {mse:.4f}, L1: {l1:.4f})")

    sae = sae.cpu()

    # Save SAE
    torch.save({
        "sae_state_dict": sae.state_dict(),
        "d_model": d_model,
        "n_features": n_features,
        "l1_coeff": l1_coeff,
        "normalize_scale": scale,
    }, os.path.join(output_dir, "sae.pt"))

    # Save activations for probe fitting
    torch.save({
        "hidden_states": x_raw,
        "labels": labels,
    }, os.path.join(output_dir, "activations.pt"))

    print(f"  SAE + activations saved to {output_dir}")

    # Fit probe and report AUC (80/20 train/test split)
    with torch.no_grad():
        x_encoded = sae.encode(x_norm).numpy()
    labels_np = labels.numpy().astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    split_idx = np.arange(len(labels_np))
    try:
        train_idx, test_idx = train_test_split(
            split_idx, test_size=0.2, random_state=42, stratify=labels_np
        )
    except ValueError:
        # Not enough samples per class for stratified split; fall back to
        # train-only AUC with a warning
        print("  [Warning] Not enough samples per class for train/test split")
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(x_encoded, labels_np)
        auc = roc_auc_score(labels_np, clf.predict_proba(x_encoded)[:, 1])
        print(f"  Per-turn probe AUC (train, no split): {auc:.4f}")
        return sae, scale

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(x_encoded[train_idx], labels_np[train_idx])
    probs_test = clf.predict_proba(x_encoded[test_idx])[:, 1]
    auc = roc_auc_score(labels_np[test_idx], probs_test)
    print(f"  Per-turn probe AUC (test): {auc:.4f}")

    return sae, scale


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train per-turn SAE probe on Turnstile hidden states"
    )
    parser.add_argument("--hidden-states-dir", required=True,
                        help="Directory with round_*.pt hidden state files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for SAE + probe artifacts")
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--l1-coeff", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    train_sae(
        args.hidden_states_dir, args.output_dir,
        expansion=args.expansion, l1_coeff=args.l1_coeff, steps=args.steps,
    )
