"""Episode runners for single-turn and multi-turn red-teaming."""

import time
from dataclasses import dataclass, field

from turnstile.agents import Adversary, Judge, Target


@dataclass
class EpisodeResult:
    goal: str
    behavior: str
    category: str
    mode: str
    num_turns: int
    conversation: list = field(default_factory=list)
    score: int = 0
    duration: float = 0.0


def run_single_turn(adversary: Adversary, target: Target, judge: Judge, goal_entry: dict) -> EpisodeResult:
    goal = goal_entry["goal"]
    start = time.time()
    attack = adversary.generate_attack(goal)
    conversation = [{"role": "user", "content": attack}]
    response = target.generate_response(conversation)
    conversation.append({"role": "assistant", "content": response})
    score = judge.evaluate(goal, goal_entry["target"], conversation)
    return EpisodeResult(
        goal=goal, behavior=goal_entry["behavior"], category=goal_entry["category"],
        mode="single", num_turns=1,
        conversation=conversation, score=score, duration=time.time() - start,
    )


def run_multi_turn(
    adversary: Adversary, target: Target, judge: Judge, goal_entry: dict, num_turns: int = 5
) -> EpisodeResult:
    goal = goal_entry["goal"]
    start = time.time()
    conversation = []
    for _ in range(num_turns):
        attack = adversary.generate_attack(goal, conversation if conversation else None)
        conversation.append({"role": "user", "content": attack})
        response = target.generate_response(conversation)
        conversation.append({"role": "assistant", "content": response})
    score = judge.evaluate(goal, goal_entry["target"], conversation)
    return EpisodeResult(
        goal=goal, behavior=goal_entry["behavior"], category=goal_entry["category"],
        mode="multi", num_turns=num_turns,
        conversation=conversation, score=score, duration=time.time() - start,
    )
