"""Load adversarial goals from JailbreakBench (JBB-Behaviors).

JailbreakBench provides 100 standardized harmful behaviors across 10 categories
aligned with OpenAI usage policies. Using a community benchmark allows direct
comparison with published single-turn attack methods.
"""

import jailbreakbench as jbb


def load_goals(n=None):
    """Load harmful behavior goals from JailbreakBench.

    Returns list of dicts with keys: behavior, goal, category, target.
    """
    dataset = jbb.read_dataset()
    goals = []
    for behavior, goal, category, target in zip(
        dataset.behaviors, dataset.goals, dataset.categories, dataset.targets
    ):
        goals.append(
            {"behavior": behavior, "goal": goal, "category": category, "target": target}
        )
    if n is not None:
        goals = goals[:n]
    return goals
