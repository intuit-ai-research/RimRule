from __future__ import annotations

import math

from rimrule.models import Rule


def rule_length(rule: Rule) -> int:
    """Return a rule's model cost: the count of symbols in its translation."""
    return rule.symbolic.size()


def mdl_score(
    rules: list[Rule], outcomes: list[bool], alpha: float
) -> tuple[float, float, float]:
    """Return ``(total, model_cost, data_cost)`` MDL score for a rule library.

    ``model_cost`` is ``alpha`` times the total rule length; ``data_cost`` is the
    Bernoulli negative log-likelihood of the correctness outcomes.
    """
    model_cost = alpha * sum(rule_length(r) for r in rules)
    n = len(outcomes)
    if n == 0:
        return model_cost, model_cost, 0.0
    k = sum(outcomes)
    p = k / n
    if p in (0.0, 1.0):
        data_cost = (
            0.0 if (p == 1.0 and k == n) or (p == 0.0 and k == 0) else float("inf")
        )
    else:
        data_cost = -(k * math.log(p) + (n - k) * math.log(1 - p))
    return model_cost + data_cost, model_cost, data_cost
