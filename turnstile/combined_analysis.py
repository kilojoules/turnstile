"""Combined cross-experiment analysis for Turnstile.

Loads hidden states from BOTH baseline (direct) and stealth experiments,
encodes through BOTH T-SAEs, fits logistic probes, computes smoothness,
and compares against the per-turn SAE probe. This is the core comparison
table for the paper: does stealth training break the T-SAE's ability
to detect unsafe conversations?

Usage:
  python -m turnstile.combined_analysis
"""

import json
import os

import numpy as np
import torch

from turnstile.temporal_sae import TemporalSAE
from turnstile.probe import SparseAutoencoder
from turnstile.temporal_analysis import (
    _load_conversations,
    load_temporal_sae,
    _encode_conversations,
    compute_smoothness,
    fit_probes,
)

# ---------------------------------------------------------------------------
# Paths (all expanduser so ~ works)
# ---------------------------------------------------------------------------

BASELINE_HS_DIR = os.path.expanduser(
    "~/turnstile/experiments/direct_3b_v1/hidden_states"
)
STEALTH_HS_DIR = os.path.expanduser(
    "~/turnstile/experiments/stealth_3b_v1/hidden_states"
)

BASELINE_TSAE_DIR = os.path.expanduser(
    "~/turnstile/results/tsae/direct_3b_v1"
)
STEALTH_TSAE_DIR = os.path.expanduser(
    "~/turnstile/results/tsae/stealth_3b_v1"
)

PER_TURN_SAE_PATH = os.path.expanduser(
    "~/turnstile/results/probe/stealth_3b_v1/sae.pt"
)

OUTPUT_DIR = os.path.expanduser(
    "~/turnstile/results/analysis/combined"
)


# ---------------------------------------------------------------------------
# Per-turn probe on combined data
# ---------------------------------------------------------------------------

def fit_per_turn_probe(sae_path, hs_list, labels):
    """Load a per-turn SAE and fit a logistic probe on combined data.

    Flattens all conversations to individual turns (each turn inherits its
    conversation-level label), encodes through the SAE, and runs stratified
    k-fold CV with logistic regression.

    Returns dict with mean, std, k (or None if data is insufficient).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    sae_data = torch.load(sae_path, weights_only=False)
    sae = SparseAutoencoder(
        sae_data["d_model"], sae_data["n_features"], sae_data["l1_coeff"]
    )
    sae.load_state_dict(sae_data["sae_state_dict"])
    sae.eval()
    scale = sae_data["normalize_scale"]

    # Flatten conversations to individual turns
    all_states = []
    all_labels = []
    for hs, label in zip(hs_list, labels):
        for t in range(hs.shape[0]):
            all_states.append(hs[t])
            all_labels.append(int(label))

    x_raw = torch.stack(all_states)
    labels_np = np.array(all_labels, dtype=int)

    # Encode through SAE
    x_norm = x_raw * scale
    with torch.no_grad():
        features = sae.encode(x_norm).numpy()

    min_class = min(int(labels_np.sum()), int((1 - labels_np).sum()))
    if min_class < 2:
        print("  [per-turn probe] Not enough class diversity")
        return None

    n_splits = min(10, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    scores = cross_val_score(clf, features, labels_np, cv=cv, scoring="roc_auc")

    return {
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "k": n_splits,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_combined_analysis():
    """Run the full combined cross-experiment analysis."""
    print("=" * 70)
    print("  COMBINED CROSS-EXPERIMENT ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load hidden states from both experiments
    # ------------------------------------------------------------------
    print("\n[1/5] Loading hidden states...")

    hs_baseline, labels_baseline, _ = _load_conversations(BASELINE_HS_DIR)
    n_bl = len(hs_baseline)
    n_bl_unsafe = sum(labels_baseline)
    n_bl_safe = n_bl - n_bl_unsafe
    print(f"  Baseline: {n_bl} conversations "
          f"({n_bl_unsafe} unsafe, {n_bl_safe} safe)")

    hs_stealth, labels_stealth, _ = _load_conversations(STEALTH_HS_DIR)
    n_st = len(hs_stealth)
    n_st_unsafe = sum(labels_stealth)
    n_st_safe = n_st - n_st_unsafe
    print(f"  Stealth:  {n_st} conversations "
          f"({n_st_unsafe} unsafe, {n_st_safe} safe)")

    # Combine
    hs_all = hs_baseline + hs_stealth
    labels_all = labels_baseline + labels_stealth
    n_total = len(hs_all)
    n_unsafe = sum(labels_all)
    n_safe = n_total - n_unsafe
    print(f"  Combined: {n_total} conversations "
          f"({n_unsafe} unsafe, {n_safe} safe)")

    source_breakdown = {
        "baseline_safe": n_bl_safe,
        "baseline_unsafe": n_bl_unsafe,
        "stealth_safe": n_st_safe,
        "stealth_unsafe": n_st_unsafe,
    }

    # ------------------------------------------------------------------
    # 2. Load both T-SAEs
    # ------------------------------------------------------------------
    print("\n[2/5] Loading T-SAEs...")

    tsae_bl, scale_bl, tsae_bl_data = load_temporal_sae(BASELINE_TSAE_DIR)
    n_high_bl = tsae_bl_data["n_high_level"]
    print(f"  Baseline T-SAE: {tsae_bl_data['n_features']} features "
          f"({n_high_bl} high-level, k={tsae_bl_data.get('k', 20)})")

    tsae_st, scale_st, tsae_st_data = load_temporal_sae(STEALTH_TSAE_DIR)
    n_high_st = tsae_st_data["n_high_level"]
    print(f"  Stealth T-SAE:  {tsae_st_data['n_features']} features "
          f"({n_high_st} high-level, k={tsae_st_data.get('k', 20)})")

    # ------------------------------------------------------------------
    # 3. Encode all conversations through each T-SAE, fit probes
    # ------------------------------------------------------------------
    results = {}

    for name, tsae, scale, n_high in [
        ("baseline_tsae", tsae_bl, scale_bl, n_high_bl),
        ("stealth_tsae", tsae_st, scale_st, n_high_st),
    ]:
        print(f"\n[3/5] {name}: encoding + probes...")
        features_by_conv = _encode_conversations(tsae, scale, hs_all)

        # Probes
        print(f"  Fitting probes on combined data ({n_total} conversations)...")
        probe_aucs = fit_probes(features_by_conv, labels_all, n_high)
        for pname, result in probe_aucs.items():
            print(f"    {pname:15s} AUC: {result['mean']:.4f} "
                  f"+/- {result['std']:.4f} ({result['k']}-fold CV)")

        # Smoothness
        print(f"  Computing smoothness...")
        delta_s, high_smooth, low_smooth = compute_smoothness(
            features_by_conv, hs_all, scale, n_high
        )
        ratio = round(low_smooth / high_smooth, 4) if high_smooth > 0 else None
        print(f"    High-level Delta_s: {high_smooth:.4f}")
        print(f"    Low-level Delta_s:  {low_smooth:.4f}")
        if ratio is not None:
            print(f"    Ratio (low/high):   {ratio:.2f}x")

        results[name] = {
            "probe_aucs": probe_aucs,
            "smoothness": {
                "high_level_delta_s": round(high_smooth, 4),
                "low_level_delta_s": round(low_smooth, 4),
                "ratio": ratio,
            },
        }

    # ------------------------------------------------------------------
    # 4. Per-turn SAE probe on combined data
    # ------------------------------------------------------------------
    print("\n[4/5] Per-turn SAE probe on combined data...")
    per_turn_result = fit_per_turn_probe(PER_TURN_SAE_PATH, hs_all, labels_all)
    if per_turn_result is not None:
        print(f"  Per-turn probe AUC: {per_turn_result['mean']:.4f} "
              f"+/- {per_turn_result['std']:.4f} "
              f"({per_turn_result['k']}-fold CV)")
    else:
        print("  Per-turn probe: insufficient data")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    print("\n[5/5] Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = {
        "n_conversations": n_total,
        "n_unsafe": n_unsafe,
        "n_safe": n_safe,
        "source_breakdown": source_breakdown,
        "baseline_tsae": results["baseline_tsae"],
        "stealth_tsae": results["stealth_tsae"],
        "per_turn_probe": per_turn_result,
    }

    output_path = os.path.join(OUTPUT_DIR, "combined_analysis.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {output_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Data: {n_total} conversations "
          f"({n_unsafe} unsafe, {n_safe} safe)")
    print(f"    Baseline: {n_bl_safe} safe + {n_bl_unsafe} unsafe = {n_bl}")
    print(f"    Stealth:  {n_st_safe} safe + {n_st_unsafe} unsafe = {n_st}")

    # Probe comparison table
    print(f"\n  {'Probe':<25s} {'AUC (mean +/- std)':>25s}")
    print(f"  {'-' * 25} {'-' * 25}")

    for tsae_name, tsae_label in [
        ("baseline_tsae", "Baseline T-SAE"),
        ("stealth_tsae", "Stealth T-SAE"),
    ]:
        for probe_name in ["high_mean", "all_mean", "high_final"]:
            aucs = results[tsae_name]["probe_aucs"].get(probe_name)
            if aucs:
                label = f"{tsae_label} / {probe_name}"
                val = f"{aucs['mean']:.4f} +/- {aucs['std']:.4f} ({aucs['k']}f)"
                print(f"  {label:<25s} {val:>25s}")

    if per_turn_result:
        label = "Per-turn SAE"
        val = (f"{per_turn_result['mean']:.4f} +/- "
               f"{per_turn_result['std']:.4f} ({per_turn_result['k']}f)")
        print(f"  {label:<25s} {val:>25s}")

    # Smoothness comparison
    print(f"\n  {'Smoothness':<25s} {'High Delta_s':>12s} "
          f"{'Low Delta_s':>12s} {'Ratio':>8s}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 8}")
    for tsae_name, tsae_label in [
        ("baseline_tsae", "Baseline T-SAE"),
        ("stealth_tsae", "Stealth T-SAE"),
    ]:
        sm = results[tsae_name]["smoothness"]
        ratio_str = f"{sm['ratio']:.2f}x" if sm["ratio"] is not None else "N/A"
        print(f"  {tsae_label:<25s} {sm['high_level_delta_s']:>12.4f} "
              f"{sm['low_level_delta_s']:>12.4f} {ratio_str:>8s}")

    print("\n" + "=" * 70)
    return report


if __name__ == "__main__":
    run_combined_analysis()
