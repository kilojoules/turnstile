"""Load adversarial goals from JailbreakBench (JBB-Behaviors).

JailbreakBench provides 100 standardized harmful behaviors across 10 categories
aligned with OpenAI usage policies. Using a community benchmark allows direct
comparison with published single-turn attack methods.
"""

import sys
import types


def _patch_litellm():
    """Patch missing litellm submodules to avoid import errors in jailbreakbench."""
    for mod_name in [
        "litellm", "litellm.llms", "litellm.llms.prompt_templates",
        "litellm.llms.prompt_templates.factory",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["litellm.llms.prompt_templates.factory"].custom_prompt = (
        lambda *a, **kw: None
    )


def load_goals(n=None):
    """Load harmful behavior goals from JailbreakBench.

    Returns list of dicts with keys: behavior, goal, category, target.
    """
    _patch_litellm()
    import jailbreakbench as jbb

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
