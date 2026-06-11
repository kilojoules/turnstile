"""Pick 20 steering cells for the canon-vs-truncated rubric smoke test.

Composition (deterministic, seed=42), drawn from sweep_judged.jsonl:
  5 baselines with truncated Likert = 1   (compliance-but-no-uplift)
  5 baselines with truncated Likert ≥ 4   (high uplift under truncated)
  5 steered cells with truncated Likert = 1  (refusal-broken cases)
  5 steered cells with truncated Likert ≥ 2  (variable content)

These are the cells the canon-vs-truncated comparison is most informative on
— the A.5 diagnostic depends on whether canon-rubric agrees or disagrees with
truncated on the "broken refusal but no content" cases (which are the
load-bearing claim).

Writes the 20 chosen rows verbatim (with all existing columns preserved) to:
  experiments/steering_v3/layer_sweep/canon_smoke_input.jsonl
"""
import json
import random
from collections import Counter

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
INPUT = f"{ROOT}/experiments/steering_v3/layer_sweep/sweep_judged.jsonl"
OUT = f"{ROOT}/experiments/steering_v3/layer_sweep/canon_smoke_input.jsonl"
SEED = 42


def main():
    rows = [json.loads(l) for l in open(INPUT)]
    baselines = [r for r in rows if r.get("method") == "baseline"
                 and r.get("judge_harm_likert") is not None]
    steered = [r for r in rows if r.get("method") != "baseline"
               and r.get("judge_harm_likert") is not None]

    bsl_l1 = [r for r in baselines if r["judge_harm_likert"] == 1]
    bsl_hi = [r for r in baselines if r["judge_harm_likert"] >= 4]
    str_l1 = [r for r in steered if r["judge_harm_likert"] == 1]
    str_lo = [r for r in steered if r["judge_harm_likert"] >= 2]

    print(f"bsl Likert=1: {len(bsl_l1)} candidates")
    print(f"bsl Likert>=4: {len(bsl_hi)} candidates")
    print(f"steered Likert=1: {len(str_l1)} candidates")
    print(f"steered Likert>=2: {len(str_lo)} candidates")

    rng = random.Random(SEED)
    picked = []
    picked += rng.sample(bsl_l1, 5)
    picked += rng.sample(bsl_hi, 5)
    picked += rng.sample(str_l1, 5)
    picked += rng.sample(str_lo, 5)
    print(f"\npicked {len(picked)} cells; composition:")
    print(f"  baselines: {sum(1 for r in picked if r['method']=='baseline')}")
    print(f"  steered:   {sum(1 for r in picked if r['method']!='baseline')}")
    print(f"  trunc Likert dist: {dict(sorted(Counter(r['judge_harm_likert'] for r in picked).items()))}")

    with open(OUT, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
