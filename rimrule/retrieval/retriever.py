from __future__ import annotations

from rimrule.models import SYMBOLIC_FIELDS, Rule, SymbolicRule


def symbolic_score(query: SymbolicRule, rule: SymbolicRule) -> float:
    """Score a query against a rule as the mean Jaccard overlap per axis.

    Axes on which the rule declares no symbols are skipped so that an
    under-specified rule is not penalised for its silence.
    """
    score = 0.0
    for field in SYMBOLIC_FIELDS:
        q = set(getattr(query, field))
        r = set(getattr(rule, field))
        if not r:
            continue
        score += len(q & r) / len(q | r) if q | r else 0
    return score


class SymbolicRetriever:
    """Select the top-k rules for a query by symbolic overlap and tool scope."""

    def __init__(self, rules: list[Rule], top_k: int = 5):
        self.rules = rules
        self.top_k = top_k

    def retrieve(self, query: SymbolicRule) -> list[Rule]:
        """Return the highest-scoring rules applicable to ``query``.

        A rule scoped to tool categories is skipped when it shares none with the
        query. Decomposition rules are always eligible so global guidance is not
        filtered out by tool scope.
        """
        eligible = []
        query_tools = set(query.tool_category)
        for rule in self.rules:
            scope = set(rule.symbolic.tool_category)
            if scope and query_tools and not scope.intersection(query_tools):
                continue
            score = symbolic_score(query, rule.symbolic)
            if score > 0 or rule.error_type.value == "decomposition":
                eligible.append((score, rule.id, rule))
        eligible.sort(key=lambda x: (-x[0], x[1]))
        return [x[2] for x in eligible[: self.top_k]]
