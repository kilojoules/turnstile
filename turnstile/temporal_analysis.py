"""Feature analysis and visualization for Temporal SAE.

Compares T-SAE temporal probe against per-turn standard probe:
  - Fit logistic probes on high-level vs all T-SAE features
  - Compute smoothness metric (Delta_s from Bhalla et al. Sec 4.3)
  - Visualize feature trajectories for successful vs failed attacks
  - Generate heatmaps of feature activations across turns

Delta_s (smoothness) is defined as:
  For each active feature i over a conversation of T turns:
    delta_i = max_{t in [1..T]} |f_i(x_t) - f_i(x_{t-1})| / ||x_t - x_{t-1}||_2
  Delta_s = mean over active features of delta_i
  Lower Delta_s = smoother = more temporally consistent.

Usage:
  python -m turnstile.temporal_analysis \
      --hidden-states-dir experiments/frozen_v1/hidden_states \
      --tsae-dir results/tsae/frozen_v1
"""

import json
import os

import numpy as np
import torch

from turnstile.temporal_sae import TemporalSAE, normalize_activations


def load_temporal_sae(tsae_dir):
    """Load a trained T-SAE from disk."""
    data = torch.load(
        os.path.join(tsae_dir, "temporal_sae.pt"), weights_only=False
    )
    tsae = TemporalSAE(
        data["d_model"], data["n_features"],
        data["n_high_level"], k=data.get("k", 20),
    )
    tsae.load_state_dict(data["tsae_state_dict"])
    tsae.eval()
    return tsae, data["normalize_scale"], data


def _load_conversations(hidden_states_dir):
    """Load all per-conversation hidden states from round files.

    Returns:
        hs_list: list of (n_turns, d_model) tensors
        labels: list of bool (unsafe)
        turns_of_breach: list of int|None
    """
    hs_list = []
    labels = []
    turns_of_breach = []

    round_files = sorted(
        f for f in os.listdir(hidden_states_dir)
        if f.startswith("round_") and f.endswith(".pt")
    )

    for fname in round_files:
        data = torch.load(
            os.path.join(hidden_states_dir, fname), weights_only=False
        )
        tob_list = data.get("turns_of_breach",
                            [None] * len(data["labels"]))
        for hs, label, tob in zip(
            data["hidden_states"], data["labels"], tob_list
        ):
            hs_list.append(hs)
            labels.append(bool(label))
            turns_of_breach.append(tob)

    return hs_list, labels, turns_of_breach


def _encode_conversations(tsae, scale, hs_list):
    """Encode all conversations through T-SAE.

    Returns list of (n_turns, n_features) feature tensors.
    """
    features_by_conv = []
    for hs in hs_list:
        x_norm = hs * scale
        with torch.no_grad():
            f = tsae.encode(x_norm)
        features_by_conv.append(f)
    return features_by_conv


def compute_smoothness(features_by_conv, hs_list, scale, n_high):
    """Compute smoothness metric Delta_s from Bhalla et al. Section 4.3.

    For each active feature i across a conversation:
      delta_i = max_{t} |f_i(x_t) - f_i(x_{t-1})| / ||x_t - x_{t-1}||_2

    Returns per-feature Delta_s and summary statistics.
    """
    n_features = features_by_conv[0].shape[-1]
    # Accumulate max normalized change per feature
    feat_max_changes = torch.zeros(n_features)
    feat_counts = torch.zeros(n_features)

    for f, hs in zip(features_by_conv, hs_list):
        if f.shape[0] < 2:
            continue
        x_norm = hs * scale
        # Input changes (denominators)
        input_diffs = (x_norm[1:] - x_norm[:-1]).norm(dim=-1)  # (T-1,)
        input_diffs = input_diffs.clamp(min=1e-8)

        # Feature changes
        feat_diffs = (f[1:] - f[:-1]).abs()  # (T-1, n_features)

        # Max normalized change per feature across this conversation
        normalized = feat_diffs / input_diffs.unsqueeze(-1)
        max_change = normalized.max(dim=0).values  # (n_features,)

        # Only count features that are active somewhere in this conv
        active = (f.abs().sum(dim=0) > 0)
        feat_max_changes += max_change * active.float()
        feat_counts += active.float()

    # Average over conversations where feature was active
    delta_s = torch.where(
        feat_counts > 0,
        feat_max_changes / feat_counts,
        torch.zeros_like(feat_max_changes),
    )

    high_smooth = delta_s[:n_high].mean().item()
    low_smooth = delta_s[n_high:].mean().item()

    return delta_s, high_smooth, low_smooth


def fit_probes(features_by_conv, labels, n_high):
    """Fit logistic probes on T-SAE features and compare.

    Probes:
        1. high_mean: high-level features, mean-pooled across turns
        2. all_mean: all features, mean-pooled across turns
        3. high_final: high-level features at the final turn only

    Uses stratified k-fold CV (up to 10 folds). Reports mean +/- std AUC.

    Returns dict of {probe_name: {"mean": float, "std": float, "k": int}}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    labels_np = np.array(labels, dtype=int)
    min_class = min(int(labels_np.sum()), int((1 - labels_np).sum()))
    if min_class < 2:
        print("  [Skip] Not enough class diversity for probes")
        return {}

    high_mean = torch.stack(
        [f[:, :n_high].mean(dim=0) for f in features_by_conv]
    ).numpy()
    all_mean = torch.stack(
        [f.mean(dim=0) for f in features_by_conv]
    ).numpy()
    high_final = torch.stack(
        [f[-1, :n_high] for f in features_by_conv]
    ).numpy()

    n_splits = min(10, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for name, x in [("high_mean", high_mean), ("all_mean", all_mean),
                     ("high_final", high_final)]:
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        scores = cross_val_score(clf, x, labels_np, cv=cv, scoring="roc_auc")
        results[name] = {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "k": n_splits,
        }

    return results


def analyze(hidden_states_dir, tsae_dir, output_dir=None):
    """Run full T-SAE analysis: smoothness, probes, report."""
    if output_dir is None:
        output_dir = tsae_dir

    tsae, scale, tsae_data = load_temporal_sae(tsae_dir)
    n_high = tsae_data["n_high_level"]

    print("  Loading conversations...")
    hs_list, labels, turns_of_breach = _load_conversations(hidden_states_dir)
    n_unsafe = sum(labels)
    n_safe = len(labels) - n_unsafe
    print(f"  {len(hs_list)} conversations ({n_unsafe} unsafe, {n_safe} safe)")

    print("  Encoding through T-SAE...")
    features_by_conv = _encode_conversations(tsae, scale, hs_list)

    # Smoothness (Bhalla et al. Section 4.3)
    print("  Computing smoothness metrics...")
    delta_s, high_smooth, low_smooth = compute_smoothness(
        features_by_conv, hs_list, scale, n_high
    )
    print(f"    High-level Delta_s: {high_smooth:.4f}")
    print(f"    Low-level Delta_s:  {low_smooth:.4f}")
    if high_smooth > 0:
        ratio = low_smooth / high_smooth
        expectation = "(high-level smoother)" if ratio > 1 else "(unexpected)"
        print(f"    Ratio (low/high): {ratio:.2f}x {expectation}")

    # Probes
    print("  Fitting probes...")
    probe_aucs = fit_probes(features_by_conv, labels, n_high)
    for name, result in probe_aucs.items():
        print(f"    {name:15s} AUC: {result['mean']:.4f} "
              f"+/- {result['std']:.4f} ({result['k']}-fold CV)")

    # Feature trajectories: mean high-level activation per turn
    unsafe_trajs = [
        f[:, :n_high].mean(dim=-1).numpy()
        for f, l in zip(features_by_conv, labels) if l
    ]
    safe_trajs = [
        f[:, :n_high].mean(dim=-1).numpy()
        for f, l in zip(features_by_conv, labels) if not l
    ]

    # Report
    report = {
        "n_conversations": len(features_by_conv),
        "n_unsafe": n_unsafe,
        "n_safe": n_safe,
        "n_high_level_features": n_high,
        "n_total_features": tsae_data["n_features"],
        "k": tsae_data.get("k", 20),
        "smoothness": {
            "high_level_delta_s": round(high_smooth, 4),
            "low_level_delta_s": round(low_smooth, 4),
            "ratio_low_over_high": (
                round(low_smooth / high_smooth, 4) if high_smooth > 0
                else None
            ),
        },
        "probe_aucs": probe_aucs,
        "mean_unsafe_trajectory": (
            np.mean(unsafe_trajs, axis=0).tolist() if unsafe_trajs else []
        ),
        "mean_safe_trajectory": (
            np.mean(safe_trajs, axis=0).tolist() if safe_trajs else []
        ),
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "temporal_analysis.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    # Plots
    try:
        _plot_trajectories(unsafe_trajs, safe_trajs, output_dir)
        _plot_smoothness(delta_s, n_high, output_dir)
        _plot_heatmaps(features_by_conv, labels, n_high, output_dir)
    except ImportError:
        print("  [Warning] matplotlib not available — skipping plots.")

    return report


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_trajectories(unsafe_trajs, safe_trajs, output_dir):
    """Mean high-level feature activation across turns."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    # Individual trajectories (faint)
    for traj in unsafe_trajs[:20]:
        ax.plot(traj, color="#ef4444", alpha=0.15, linewidth=0.8)
    for traj in safe_trajs[:20]:
        ax.plot(traj, color="#3b82f6", alpha=0.15, linewidth=0.8)

    # Mean trajectories
    if unsafe_trajs:
        mean_unsafe = np.mean(unsafe_trajs, axis=0)
        ax.plot(mean_unsafe, color="#ef4444", linewidth=2.5,
                label=f"Unsafe (n={len(unsafe_trajs)})")
    if safe_trajs:
        mean_safe = np.mean(safe_trajs, axis=0)
        ax.plot(mean_safe, color="#3b82f6", linewidth=2.5,
                label=f"Safe (n={len(safe_trajs)})")

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Mean High-Level Feature Activation", fontsize=12)
    ax.set_title("Safety Trajectory: High-Level T-SAE Features",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectories.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: trajectories.png")


def _plot_smoothness(delta_s, n_high, output_dir):
    """Histogram of Delta_s for high-level vs low-level features."""
    import matplotlib.pyplot as plt

    high_ds = delta_s[:n_high].numpy()
    low_ds = delta_s[n_high:].numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(high_ds, bins=50, alpha=0.7, color="#6366f1",
            label=f"High-level (n={n_high})", density=True)
    ax.hist(low_ds, bins=50, alpha=0.7, color="#f59e0b",
            label=f"Low-level (n={len(low_ds)})", density=True)
    ax.axvline(high_ds.mean(), color="#6366f1", linestyle="--", linewidth=2)
    ax.axvline(low_ds.mean(), color="#f59e0b", linestyle="--", linewidth=2)

    ax.set_xlabel("Smoothness (Delta_s)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Feature Smoothness: High-Level vs Low-Level",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "smoothness.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: smoothness.png")


def _plot_heatmaps(features_by_conv, labels, n_high, output_dir,
                   n_examples=5):
    """Feature activation heatmaps (high-level features x turns)."""
    import matplotlib.pyplot as plt

    unsafe_feats = [f for f, l in zip(features_by_conv, labels) if l]
    safe_feats = [f for f, l in zip(features_by_conv, labels) if not l]

    n_show = min(n_examples, len(unsafe_feats), len(safe_feats))
    if n_show == 0:
        return

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_show):
        # Unsafe
        ax = axes[0, col]
        f = unsafe_feats[col][:, :n_high].numpy().T
        ax.imshow(f, aspect="auto", cmap="Reds", interpolation="nearest")
        ax.set_xlabel("Turn")
        if col == 0:
            ax.set_ylabel("High-Level Feature")
        ax.set_title(f"Unsafe #{col+1}", fontsize=10)

        # Safe
        ax = axes[1, col]
        f = safe_feats[col][:, :n_high].numpy().T
        ax.imshow(f, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_xlabel("Turn")
        if col == 0:
            ax.set_ylabel("High-Level Feature")
        ax.set_title(f"Safe #{col+1}", fontsize=10)

    fig.suptitle("T-SAE High-Level Feature Heatmaps",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmaps.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: heatmaps.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Temporal SAE features"
    )
    parser.add_argument("--hidden-states-dir", required=True)
    parser.add_argument("--tsae-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    analyze(args.hidden_states_dir, args.tsae_dir, args.output_dir)
