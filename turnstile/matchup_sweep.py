"""Run frozen-victim experiments across multiple adversary–victim matchups.

For each matchup: bootstrap adversary LoRA on seed data, then run the
multi-turn training loop with dual-judge (Llama Guard + 70B Together API)
and JailbreakBench behaviors.

Matchups (same as REDKWEEN):
  1B vs 8B   — small adversary, large victim
  3B vs 8B   — medium adversary, large victim
  8B vs 8B   — matched size
  8B vs 3B   — large adversary, small victim

Usage:
  python -m turnstile.matchup_sweep --together-key KEY
  python -m turnstile.matchup_sweep --together-key KEY --matchups 1Bv8B 3Bv8B
  python -m turnstile.matchup_sweep --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


MATCHUPS = [
    {
        "name": "1Bv8B",
        "adversary": "meta-llama/Llama-3.2-1B-Instruct",
        "victim": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "name": "3Bv8B",
        "adversary": "meta-llama/Llama-3.2-3B-Instruct",
        "victim": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "name": "8Bv8B",
        "adversary": "meta-llama/Llama-3.1-8B-Instruct",
        "victim": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "name": "8Bv3B",
        "adversary": "meta-llama/Llama-3.1-8B-Instruct",
        "victim": "meta-llama/Llama-3.2-3B-Instruct",
    },
]


def is_complete(output_dir, name, expected_rounds):
    """Check if an experiment already completed all rounds."""
    metrics_path = os.path.join(output_dir, name, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return False
    count = 0
    with open(metrics_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count >= expected_rounds


def read_metrics(output_dir, name):
    """Read metrics.jsonl for a completed experiment."""
    metrics_path = os.path.join(output_dir, name, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return None
    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_matchup(matchup, rounds, candidates, num_turns, seed, output_dir,
                together_key, num_seeds):
    """Bootstrap + run loop for one matchup."""
    name = matchup["name"]
    adv = matchup["adversary"]
    vic = matchup["victim"]
    exp_name = f"frozen_{name}"

    print(f"\n{'=' * 60}")
    print(f"MATCHUP: {name}")
    print(f"  Adversary: {adv}")
    print(f"  Victim:    {vic}")
    print(f"{'=' * 60}")

    # Check for resume
    if is_complete(output_dir, exp_name, rounds):
        print(f"  >> SKIPPING: already completed")
        return {"name": exp_name, "skipped": True, "returncode": 0}

    exp_dir = os.path.join(output_dir, exp_name)
    data_dir = os.path.join(exp_dir, "data")
    adapter_dir = os.path.join(exp_dir, "adapters")

    # Phase 0: Bootstrap adversary if no adapter exists
    adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        print(f"\n  --- BOOTSTRAP: Training {name} adversary LoRA ---")
        bootstrap_cmd = [
            sys.executable, "-m", "turnstile.bootstrap",
            "--adversary-model", adv,
            "--num-seeds", str(num_seeds),
            "--num-turns", "3",
            "--seed", str(seed),
            "--data-dir", data_dir,
            "--adapter-path", adapter_dir,
            "--lora-iters", "200",
        ]
        proc = subprocess.run(bootstrap_cmd, capture_output=False)
        if proc.returncode != 0:
            print(f"  >> BOOTSTRAP FAILED (rc={proc.returncode})")
            return {"name": exp_name, "skipped": False,
                    "returncode": proc.returncode, "phase": "bootstrap"}
    else:
        print(f"  Adapter exists, skipping bootstrap")

    # Phase 1: Run training loop
    print(f"\n  --- LOOP: {rounds} rounds × {candidates} candidates ---")
    loop_cmd = [
        sys.executable, "-m", "turnstile.loop",
        "--name", exp_name,
        "--adversary-model", adv,
        "--victim-model", vic,
        "--rounds", str(rounds),
        "--candidates", str(candidates),
        "--num-turns", str(num_turns),
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    if together_key:
        loop_cmd += ["--together-key", together_key]

    start = time.time()
    proc = subprocess.run(loop_cmd, capture_output=False)
    elapsed = time.time() - start

    result = {
        "name": exp_name,
        "matchup": name,
        "adversary": adv.split("/")[-1],
        "victim": vic.split("/")[-1],
        "skipped": False,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(elapsed / 3600, 2),
    }

    metrics = read_metrics(output_dir, exp_name)
    if metrics:
        asrs = [r["asr"] for r in metrics]
        result["mean_asr"] = round(sum(asrs) / len(asrs), 4)
        result["final_asr"] = metrics[-1]["asr"]
        result["peak_asr"] = max(asrs)
        result["num_rounds"] = len(metrics)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run frozen-victim experiments across model matchups"
    )
    parser.add_argument("--matchups", nargs="*", type=str, default=None,
                        help="Which matchups to run (e.g. 1Bv8B 3Bv8B)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--together-key", type=str, default="",
                        help="Together API key for dual judge")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.matchups:
        selected = [m for m in MATCHUPS if m["name"] in args.matchups]
    else:
        selected = MATCHUPS

    print("=" * 60)
    print("TURNSTILE MATCHUP SWEEP")
    print("=" * 60)
    print(f"Matchups: {[m['name'] for m in selected]}")
    print(f"Rounds: {args.rounds}, Candidates/round: {args.candidates}")
    print(f"Turns: {args.num_turns}, Seeds: {args.num_seeds}")
    print(f"Dual judge: {'YES' if args.together_key else 'NO (Llama Guard only)'}")
    print(f"Output: {args.output_dir}/")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for m in selected:
            print(f"\n  {m['name']}:")
            print(f"    adv={m['adversary']}")
            print(f"    vic={m['victim']}")
            print(f"    exp=frozen_{m['name']}")
        print(f"\n{len(selected)} matchups would run.")
        return

    sweep_start = time.time()
    results = []

    for m in selected:
        result = run_matchup(
            m, args.rounds, args.candidates, args.num_turns,
            args.seed, args.output_dir, args.together_key, args.num_seeds,
        )
        results.append(result)

        status = "OK" if result["returncode"] == 0 else "FAIL"
        if result.get("skipped"):
            status = "SKIP"
        print(f"\n  >> {result['name']}: {status}")

    total = time.time() - sweep_start

    # Summary
    print(f"\n{'=' * 60}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Name':<20} | {'Mean ASR':<10} | {'Peak ASR':<10} | "
          f"{'Final ASR':<10} | {'Time':<8} | Status")
    print("-" * 80)
    for r in results:
        mean = f"{r.get('mean_asr', '-')}"
        peak = f"{r.get('peak_asr', '-')}"
        final = f"{r.get('final_asr', '-')}"
        t = f"{r.get('elapsed_hours', 0)}h"
        st = "SKIP" if r.get("skipped") else (
            "OK" if r["returncode"] == 0 else "FAIL")
        print(f"{r['name']:<20} | {mean:<10} | {peak:<10} | "
              f"{final:<10} | {t:<8} | {st}")

    print(f"\nTotal: {total / 3600:.1f} hours")

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rounds": args.rounds,
        "candidates": args.candidates,
        "num_turns": args.num_turns,
        "dual_judge": bool(args.together_key),
        "total_elapsed_hours": round(total / 3600, 2),
        "results": results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "matchup_sweep_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {path}")


if __name__ == "__main__":
    main()
