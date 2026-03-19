"""Turnstile: Multi-turn adversarial red-teaming for agentic AI systems."""

import argparse
import json
from pathlib import Path

from turnstile.agents import Adversary, Judge, Target
from turnstile.episode import run_multi_turn, run_single_turn
from turnstile.goals import GOALS


def main():
    parser = argparse.ArgumentParser(description="Turnstile: multi-turn adversarial red-teaming")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--num-goals", type=int, default=None)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    goals = GOALS[: args.num_goals] if args.num_goals else GOALS
    adversary = Adversary(base_url=args.base_url, model=args.model)
    target = Target(base_url=args.base_url, model=args.model)
    judge = Judge(base_url=args.base_url, model=args.model)

    results = []
    for i, goal in enumerate(goals):
        print(f"[{i + 1}/{len(goals)}] {goal[:60]}...")
        single = run_single_turn(adversary, target, judge, goal)
        multi = run_multi_turn(adversary, target, judge, goal, args.num_turns)
        results.append(
            {
                "goal": goal,
                "single_turn": {
                    "score": single.score,
                    "conversation": single.conversation,
                    "duration": single.duration,
                },
                "multi_turn": {
                    "score": multi.score,
                    "turns": multi.num_turns,
                    "conversation": multi.conversation,
                    "duration": multi.duration,
                },
            }
        )
        delta = multi.score - single.score
        marker = "+" if delta > 0 else ""
        print(f"  Single: {single.score}/5 | Multi: {multi.score}/5 | Delta: {marker}{delta}")

    avg_single = sum(r["single_turn"]["score"] for r in results) / len(results)
    avg_multi = sum(r["multi_turn"]["score"] for r in results) / len(results)
    wins = sum(1 for r in results if r["multi_turn"]["score"] > r["single_turn"]["score"])

    print(f"\n{'=' * 50}")
    print(f"Results ({len(results)} goals, {args.num_turns} turns)")
    print(f"{'=' * 50}")
    print(f"Avg single-turn score: {avg_single:.2f}/5")
    print(f"Avg multi-turn score:  {avg_multi:.2f}/5")
    print(f"Avg delta:             {avg_multi - avg_single:+.2f}")
    print(f"Multi > Single:        {wins}/{len(results)}")

    output = Path(args.output)
    output.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
