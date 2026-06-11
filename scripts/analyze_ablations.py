"""Ablation analysis: compare training schemes for stealth adversary.

Compares all conditions:
  - weighted (original, alpha=3)
  - alpha ablation (alpha=1,2,5)
  - iw_weighted (importance-weighted SFT)
  - probe_dpo (probe-aware DPO)
  - control (no stealth signal)

For each, reports:
  - ASR trajectory and mean
  - Probe evasion rate
  - Stealth ASR (success AND evasion)
  - Statistical comparisons

Usage:
    python scripts/analyze_ablations.py \
        --experiment-dir experiments \
        --seeds 42 123 456
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_condition(experiment_dir, name_pattern, seeds):
    """Load per-round metrics for a condition across seeds.

    Args:
        name_pattern: f-string pattern with {seed} placeholder,
            e.g. "stealth_s{seed}" or "alpha1_s{seed}"

    Returns dict: seed -> list of round dicts.
    """
    data = {}
    for seed in seeds:
        name = name_pattern.format(seed=seed)
        metrics_file = os.path.join(experiment_dir, name, "metrics.jsonl")
        if not os.path.exists(metrics_file):
            continue
        rounds = []
        with open(metrics_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rounds.append(json.loads(line))
        if rounds:
            data[seed] = rounds
    return data


def extract_metric(data, metric, n_rounds):
    """Extract a metric as seed -> np.array(n_rounds)."""
    result = {}
    for seed, rounds in data.items():
        vals = []
        for r in range(n_rounds):
            matching = [m for m in rounds if m["round"] == r]
            if matching and metric in matching[0]:
                vals.append(matching[0][metric])
            else:
                vals.append(np.nan)
        result[seed] = np.array(vals)
    return result


def mean_across_seeds(per_seed):
    """Mean across seeds, returns np.array(n_rounds)."""
    if not per_seed:
        return np.array([])
    return np.nanmean(list(per_seed.values()), axis=0)


def overall_mean(per_seed):
    """Scalar mean across all seeds and rounds."""
    if not per_seed:
        return np.nan
    return float(np.nanmean([np.nanmean(v) for v in per_seed.values()]))


# -----------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------

def compare_conditions(a_data, b_data, metric, n_rounds, label_a, label_b):
    """Run statistical comparison between two conditions."""
    a_vals = extract_metric(a_data, metric, n_rounds)
    b_vals = extract_metric(b_data, metric, n_rounds)

    a_mean = mean_across_seeds(a_vals)
    b_mean = mean_across_seeds(b_vals)

    if len(a_mean) == 0 or len(b_mean) == 0:
        print(f"  {label_a} vs {label_b}: insufficient data")
        return

    a_overall = float(np.nanmean(a_mean))
    b_overall = float(np.nanmean(b_mean))
    diff = a_overall - b_overall

    # Wilcoxon signed-rank (paired by round)
    valid = ~(np.isnan(a_mean) | np.isnan(b_mean))
    a_valid = a_mean[valid]
    b_valid = b_mean[valid]

    p_val = np.nan
    if len(a_valid) >= 5:
        from scipy.stats import wilcoxon
        try:
            _, p_val = wilcoxon(a_valid, b_valid, alternative="two-sided")
        except ValueError:
            pass  # all differences are zero

    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    print(f"  {label_a:>20s} vs {label_b:<20s}: "
          f"{a_overall:.1%} vs {b_overall:.1%} "
          f"(diff={diff:+.1%}, p={p_val:.3f}) {sig}")

    return {"a": a_overall, "b": b_overall, "diff": diff, "p": p_val}


def bootstrap_ci(a_data, b_data, metric, n_rounds, n_boot=10000):
    """Bootstrap 95% CI on mean metric difference (a - b)."""
    a_vals = extract_metric(a_data, metric, n_rounds)
    b_vals = extract_metric(b_data, metric, n_rounds)

    a_seeds = list(a_vals.values())
    b_seeds = list(b_vals.values())

    if not a_seeds or not b_seeds:
        return None, None

    diffs = []
    for _ in range(n_boot):
        ai = np.random.choice(len(a_seeds), size=len(a_seeds), replace=True)
        bi = np.random.choice(len(b_seeds), size=len(b_seeds), replace=True)
        a_boot = np.nanmean([a_seeds[i] for i in ai], axis=0)
        b_boot = np.nanmean([b_seeds[i] for i in bi], axis=0)
        diffs.append(np.nanmean(a_boot) - np.nanmean(b_boot))

    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


# -----------------------------------------------------------------------
# Alpha sensitivity
# -----------------------------------------------------------------------

def alpha_sensitivity(experiment_dir, seeds, n_rounds):
    """Analyze sensitivity to alpha across {1, 2, 3, 5}."""
    print("\n" + "=" * 60)
    print("ALPHA SENSITIVITY ANALYSIS")
    print("=" * 60)

    alphas = [1, 2, 3, 5]
    results = {}
    for alpha in alphas:
        if alpha == 3:
            pattern = "stealth_s{seed}"  # original experiment
        else:
            pattern = f"alpha{alpha}_s{{seed}}"
        data = load_condition(experiment_dir, pattern, seeds)
        if not data:
            print(f"  alpha={alpha}: no data")
            continue
        asr_vals = extract_metric(data, "asr", n_rounds)
        results[alpha] = {
            "mean_asr": overall_mean(data),
            "n_seeds": len(data),
            "per_round": mean_across_seeds(asr_vals),
        }

        # Stealth metrics (if present)
        stealth_vals = extract_metric(data, "stealth_asr", n_rounds)
        evasion_vals = extract_metric(data, "evasion_rate", n_rounds)
        results[alpha]["mean_stealth_asr"] = overall_mean(
            {k: v for k, v in stealth_vals.items() if not np.all(np.isnan(v))}
        ) if stealth_vals else np.nan
        results[alpha]["mean_evasion"] = overall_mean(
            {k: v for k, v in evasion_vals.items() if not np.all(np.isnan(v))}
        ) if evasion_vals else np.nan

    if not results:
        print("  No alpha ablation data found.")
        return

    print(f"\n{'Alpha':>6s}  {'Seeds':>5s}  {'ASR':>8s}  "
          f"{'Stealth':>8s}  {'Evasion':>8s}")
    print("-" * 45)
    for alpha in sorted(results):
        r = results[alpha]
        print(f"{alpha:6.0f}  {r['n_seeds']:5d}  {r['mean_asr']:8.1%}  "
              f"{r['mean_stealth_asr']:8.1%}  {r['mean_evasion']:8.1%}")

    # Check sensitivity: is max-min ASR difference > 2pp?
    asrs = [r["mean_asr"] for r in results.values()
            if not np.isnan(r["mean_asr"])]
    if len(asrs) >= 2:
        spread = max(asrs) - min(asrs)
        print(f"\nASR spread across alphas: {spread:.1%}")
        if spread < 0.02:
            print("  -> Results INSENSITIVE to alpha (spread < 2pp).")
            print("     The scheme is robust to this hyperparameter choice.")
        else:
            print(f"  -> Results sensitive to alpha (spread = {spread:.1%}).")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

CONDITIONS = {
    # name: (pattern, description)
    "control":     ("control_s{seed}",      "No stealth (SFT on all wins)"),
    "weighted":    ("stealth_s{seed}",       "Bootstrap-weighted SFT (alpha=3)"),
    "iw":          ("iw_s{seed}",            "Importance-weighted SFT (alpha=3)"),
    "pdpo":        ("pdpo_s{seed}",          "Probe-aware DPO"),
    "control_h":   ("control_hard_s{seed}",  "Control (hardened victim)"),
    "weighted_h":  ("stealth_hard_s{seed}",  "Bootstrap-weighted (hardened)"),
    "iw_h":        ("iw_hard_s{seed}",       "IW-SFT (hardened victim)"),
    "pdpo_h":      ("pdpo_hard_s{seed}",     "Probe-DPO (hardened victim)"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Ablation analysis: training scheme comparison"
    )
    parser.add_argument("--experiment-dir", default="experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--n-rounds", type=int, default=15)
    args = parser.parse_args()

    seeds = args.seeds
    n_rounds = args.n_rounds

    # Load all conditions
    loaded = {}
    print("=== Loading experiment data ===")
    for name, (pattern, desc) in CONDITIONS.items():
        data = load_condition(args.experiment_dir, pattern, seeds)
        if data:
            loaded[name] = data
            print(f"  {name:>12s}: {len(data)} seeds — {desc}")
        else:
            print(f"  {name:>12s}: no data")

    if not loaded:
        print("\n[ERROR] No experiment data found.")
        sys.exit(1)

    # ----- Summary Table -----
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY (mean ASR across all seeds and rounds)")
    print("=" * 60)

    metrics = ["asr", "stealth_asr", "evasion_rate"]
    headers = ["ASR", "Stealth ASR", "Evasion"]

    print(f"{'Condition':>12s}  {'Seeds':>5s}  ", end="")
    for h in headers:
        print(f"{h:>12s}  ", end="")
    print()
    print("-" * 60)

    for name in CONDITIONS:
        if name not in loaded:
            continue
        data = loaded[name]
        print(f"{name:>12s}  {len(data):5d}  ", end="")
        for m in metrics:
            vals = extract_metric(data, m, n_rounds)
            mean = overall_mean(data) if m == "asr" else np.nan
            if m != "asr":
                flat = [v for arr in vals.values() for v in arr if not np.isnan(v)]
                mean = np.mean(flat) if flat else np.nan
            print(f"{mean:12.1%}  ", end="")
        print()

    # ----- Per-Round ASR Trajectories -----
    print("\n" + "=" * 60)
    print("PER-ROUND ASR TRAJECTORIES")
    print("=" * 60)

    present = [n for n in CONDITIONS if n in loaded]
    header = f"{'Round':>5s}  " + "  ".join(f"{n:>12s}" for n in present)
    print(header)
    print("-" * len(header))

    for r in range(n_rounds):
        row = f"{r:5d}  "
        for name in present:
            vals = extract_metric(loaded[name], "asr", n_rounds)
            mean_arr = mean_across_seeds(vals)
            if r < len(mean_arr):
                row += f"{mean_arr[r]:12.1%}  "
            else:
                row += f"{'—':>12s}  "
        print(row)

    # ----- Frozen Victim Comparisons -----
    frozen = ["control", "weighted", "iw", "pdpo"]
    frozen_present = [n for n in frozen if n in loaded]
    if len(frozen_present) >= 2:
        print("\n" + "=" * 60)
        print("PAIRWISE COMPARISONS (frozen victim, ASR)")
        print("=" * 60)
        for i, a in enumerate(frozen_present):
            for b in frozen_present[i + 1:]:
                compare_conditions(
                    loaded[a], loaded[b], "asr", n_rounds,
                    a, b,
                )

    # ----- Hardened Victim Comparisons -----
    hardened = ["control_h", "weighted_h", "iw_h", "pdpo_h"]
    hardened_present = [n for n in hardened if n in loaded]
    if len(hardened_present) >= 2:
        print("\n" + "=" * 60)
        print("PAIRWISE COMPARISONS (hardened victim, ASR)")
        print("=" * 60)
        for i, a in enumerate(hardened_present):
            for b in hardened_present[i + 1:]:
                compare_conditions(
                    loaded[a], loaded[b], "asr", n_rounds,
                    a, b,
                )

    # ----- Key Hypotheses -----
    print("\n" + "=" * 60)
    print("KEY HYPOTHESIS TESTS")
    print("=" * 60)

    # H1: IW-weighted > bootstrap-weighted (same alpha, fixes dedup issue)
    if "iw" in loaded and "weighted" in loaded:
        print("\nH1: Importance-weighted SFT vs bootstrap-weighted SFT (frozen)")
        result = compare_conditions(
            loaded["iw"], loaded["weighted"], "asr", n_rounds,
            "iw_weighted", "weighted",
        )
        ci = bootstrap_ci(loaded["iw"], loaded["weighted"], "asr", n_rounds)
        if ci[0] is not None:
            print(f"     Bootstrap 95% CI on diff: [{ci[0]:+.2%}, {ci[1]:+.2%}]")

    # H2: Probe-DPO > bootstrap-weighted (better data efficiency)
    if "pdpo" in loaded and "weighted" in loaded:
        print("\nH2: Probe-aware DPO vs bootstrap-weighted SFT (frozen)")
        result = compare_conditions(
            loaded["pdpo"], loaded["weighted"], "asr", n_rounds,
            "probe_dpo", "weighted",
        )
        ci = bootstrap_ci(loaded["pdpo"], loaded["weighted"], "asr", n_rounds)
        if ci[0] is not None:
            print(f"     Bootstrap 95% CI on diff: [{ci[0]:+.2%}, {ci[1]:+.2%}]")

    # H3: Probe-DPO > control under hardening (the real test)
    if "pdpo_h" in loaded and "control_h" in loaded:
        print("\nH3: Probe-DPO vs control under hardened victim")
        result = compare_conditions(
            loaded["pdpo_h"], loaded["control_h"], "asr", n_rounds,
            "pdpo_hard", "control_hard",
        )
        ci = bootstrap_ci(
            loaded["pdpo_h"], loaded["control_h"], "asr", n_rounds
        )
        if ci[0] is not None:
            print(f"     Bootstrap 95% CI on diff: [{ci[0]:+.2%}, {ci[1]:+.2%}]")

    # H4: Probe evasion comparison (does training scheme affect evasion?)
    print("\n" + "=" * 60)
    print("PROBE EVASION COMPARISON")
    print("=" * 60)
    for name in present:
        data = loaded[name]
        vals = extract_metric(data, "evasion_rate", n_rounds)
        flat = [v for arr in vals.values() for v in arr if not np.isnan(v)]
        if flat:
            print(f"  {name:>12s}: evasion={np.mean(flat):.1%} "
                  f"+/- {np.std(flat):.1%} (n={len(flat)})")

    # ----- Alpha Sensitivity -----
    alpha_sensitivity(args.experiment_dir, seeds, n_rounds)

    # ----- Probe DPO Pair Statistics -----
    print("\n" + "=" * 60)
    print("PROBE-DPO PAIR STATISTICS (if available)")
    print("=" * 60)
    for seed in seeds:
        for suffix in ["", "_hard"]:
            name = f"pdpo{suffix}_s{seed}"
            data_dir = os.path.join(
                args.experiment_dir, name, "data"
            )
            if not os.path.isdir(data_dir):
                continue
            pair_files = [
                f for f in os.listdir(data_dir)
                if f.startswith("probe_dpo_pairs_round_")
            ]
            if not pair_files:
                continue
            # Read last round's pairs for summary
            last_file = sorted(pair_files)[-1]
            pairs = []
            with open(os.path.join(data_dir, last_file)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pairs.append(json.loads(line))
            types = defaultdict(int)
            for p in pairs:
                types[p.get("pair_type", "unknown")] += 1
            print(f"  {name} ({last_file}): {len(pairs)} pairs")
            for t, n in sorted(types.items()):
                print(f"    {t}: {n}")


if __name__ == "__main__":
    main()
