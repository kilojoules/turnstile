"""A/B analysis: stealth vs control experiment comparison.

Loads metrics from all experiment arms, runs statistical tests,
and generates comparison plots.

Usage:
    python scripts/analyze_ab.py \
        --experiment-dir experiments \
        --seeds 42 123 456
"""

import argparse
import json
import os
import sys

import numpy as np


def load_metrics(experiment_dir, condition, seeds):
    """Load per-round metrics for a condition across all seeds."""
    all_rounds = {}  # seed -> list of round metrics
    for seed in seeds:
        name = f"{condition}_s{seed}"
        metrics_file = os.path.join(experiment_dir, name, "metrics.jsonl")
        if not os.path.exists(metrics_file):
            print(f"  [WARN] Missing: {metrics_file}")
            continue
        rounds = []
        with open(metrics_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rounds.append(json.loads(line))
        all_rounds[seed] = rounds
        print(f"  {name}: {len(rounds)} rounds")
    return all_rounds


def extract_per_round_asr(all_rounds, n_rounds):
    """Extract ASR arrays aligned by round number.

    Returns dict: seed -> np.array of ASR per round.
    """
    result = {}
    for seed, rounds in all_rounds.items():
        asrs = []
        for r in range(n_rounds):
            matching = [m for m in rounds if m["round"] == r]
            if matching:
                asrs.append(matching[0]["asr"])
            else:
                asrs.append(np.nan)
        result[seed] = np.array(asrs)
    return result


def main():
    parser = argparse.ArgumentParser(description="A/B analysis: stealth vs control")
    parser.add_argument("--experiment-dir", default="experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--n-rounds", type=int, default=15)
    args = parser.parse_args()

    seeds = args.seeds
    n_rounds = args.n_rounds

    print("=== Loading stealth metrics ===")
    stealth_data = load_metrics(args.experiment_dir, "stealth", seeds)
    print("\n=== Loading control metrics ===")
    control_data = load_metrics(args.experiment_dir, "control", seeds)

    if not stealth_data or not control_data:
        print("\n[ERROR] Need data from both conditions.")
        sys.exit(1)

    # Extract per-round ASR
    stealth_asr = extract_per_round_asr(stealth_data, n_rounds)
    control_asr = extract_per_round_asr(control_data, n_rounds)

    # Mean ASR across seeds per round
    stealth_seeds = list(stealth_asr.values())
    control_seeds = list(control_asr.values())
    stealth_mean = np.nanmean(stealth_seeds, axis=0)
    control_mean = np.nanmean(control_seeds, axis=0)

    print(f"\n=== Per-Round ASR (mean across {len(seeds)} seeds) ===")
    print(f"{'Round':>5s}  {'Stealth':>8s}  {'Control':>8s}  {'Diff':>8s}")
    print("-" * 35)
    for r in range(n_rounds):
        s = stealth_mean[r]
        c = control_mean[r]
        d = s - c
        print(f"{r:5d}  {s:8.1%}  {c:8.1%}  {d:+8.1%}")

    # Overall means
    stealth_overall = np.nanmean(stealth_mean)
    control_overall = np.nanmean(control_mean)
    print(f"\n{'Mean':>5s}  {stealth_overall:8.1%}  {control_overall:8.1%}  "
          f"{stealth_overall - control_overall:+8.1%}")

    # Per-seed summaries
    print(f"\n=== Per-Seed Mean ASR ===")
    for seed in seeds:
        s = np.nanmean(stealth_asr.get(seed, [np.nan]))
        c = np.nanmean(control_asr.get(seed, [np.nan]))
        print(f"  Seed {seed}: stealth={s:.1%}, control={c:.1%}, diff={s-c:+.1%}")

    # --- Statistical Tests ---
    print(f"\n=== Statistical Tests ===")

    # Primary: Wilcoxon signed-rank on paired per-round ASR
    # (paired by round, each value is the mean across seeds for that round)
    from scipy.stats import wilcoxon, mannwhitneyu

    # Remove rounds where either is NaN
    valid = ~(np.isnan(stealth_mean) | np.isnan(control_mean))
    s_valid = stealth_mean[valid]
    c_valid = control_mean[valid]

    if len(s_valid) >= 5:
        stat, p_wilcox = wilcoxon(s_valid, c_valid, alternative="two-sided")
        print(f"Wilcoxon signed-rank (paired by round, N={len(s_valid)}):")
        print(f"  statistic={stat:.1f}, p={p_wilcox:.4f}")
        if p_wilcox < 0.05:
            print(f"  ** Significant at p<0.05 **")
        else:
            print(f"  Not significant (p>0.05)")
    else:
        print(f"  Too few valid rounds ({len(s_valid)}) for Wilcoxon test")

    # Secondary: Mann-Whitney U on all per-round ASR values (unpaired)
    # Pool all per-seed per-round values
    stealth_all = np.concatenate([v[~np.isnan(v)] for v in stealth_seeds])
    control_all = np.concatenate([v[~np.isnan(v)] for v in control_seeds])

    if len(stealth_all) >= 5 and len(control_all) >= 5:
        stat_mw, p_mw = mannwhitneyu(stealth_all, control_all,
                                      alternative="two-sided")
        print(f"\nMann-Whitney U (unpaired, N_stealth={len(stealth_all)}, "
              f"N_control={len(control_all)}):")
        print(f"  statistic={stat_mw:.1f}, p={p_mw:.4f}")

    # Effect size: Cohen's d
    pooled_std = np.sqrt((np.var(stealth_all) + np.var(control_all)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(stealth_all) - np.mean(control_all)) / pooled_std
        print(f"\nCohen's d: {cohens_d:.3f}")
        if abs(cohens_d) < 0.2:
            print("  (negligible effect)")
        elif abs(cohens_d) < 0.5:
            print("  (small effect)")
        elif abs(cohens_d) < 0.8:
            print("  (medium effect)")
        else:
            print("  (large effect)")

    # Bootstrap CI on mean ASR difference
    print(f"\n=== Bootstrap 95% CI on Mean ASR Difference ===")
    n_boot = 10000
    diffs = []
    for _ in range(n_boot):
        # Resample seeds with replacement
        boot_seeds = np.random.choice(len(seeds), size=len(seeds), replace=True)
        s_boot = np.nanmean([stealth_seeds[i] for i in boot_seeds], axis=0)
        c_boot = np.nanmean([control_seeds[i] for i in boot_seeds], axis=0)
        diffs.append(np.nanmean(s_boot) - np.nanmean(c_boot))

    diffs = np.array(diffs)
    ci_lo = np.percentile(diffs, 2.5)
    ci_hi = np.percentile(diffs, 97.5)
    print(f"  Mean diff: {np.mean(diffs):+.2%}")
    print(f"  95% CI: [{ci_lo:+.2%}, {ci_hi:+.2%}]")
    if ci_lo > 0:
        print("  ** CI excludes zero: stealth > control **")
    elif ci_hi < 0:
        print("  ** CI excludes zero: control > stealth **")
    else:
        print("  CI includes zero: no significant difference")

    # --- Probe Score Comparison ---
    print(f"\n=== Probe Score on Wins (stealth vs control) ===")
    for condition, data in [("stealth", stealth_data), ("control", control_data)]:
        probe_scores = []
        for seed, rounds in data.items():
            for m in rounds:
                ps = m.get("mean_probe_score_wins") or m.get("probe_wins")
                if ps is not None:
                    probe_scores.append(ps)
        if probe_scores:
            print(f"  {condition}: mean={np.mean(probe_scores):.3f}, "
                  f"std={np.std(probe_scores):.3f}, n={len(probe_scores)}")

    # --- Adaptation Dynamics ---
    print(f"\n=== Adaptation: ASR Around Probe Retraining (stealth only) ===")
    retrain_round = 4  # retrain_probe_every=5, triggers at round 4
    for seed, asrs_arr in stealth_asr.items():
        if len(asrs_arr) > retrain_round + 1:
            before = asrs_arr[retrain_round - 1] if retrain_round > 0 else np.nan
            at_retrain = asrs_arr[retrain_round]
            after = asrs_arr[retrain_round + 1]
            print(f"  Seed {seed}: before={before:.1%}, "
                  f"at_retrain={at_retrain:.1%}, after={after:.1%}")


if __name__ == "__main__":
    main()
