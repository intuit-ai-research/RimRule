from __future__ import annotations

from rimrule.models import Rule, SymbolicRule
from rimrule.retrieval.retriever import SymbolicRetriever


def affected_samples(
    old_rules: list[Rule],
    new_rules: list[Rule],
    query_symbols: list[SymbolicRule],
    top_k: int,
) -> set[int]:
    """Return sample indices whose retrieved rule set differs after an edit.

    Only these samples need re-evaluation when comparing ``old_rules`` to
    ``new_rules``; the rest keep their cached outcomes.
    """
    old = SymbolicRetriever(old_rules, top_k)
    new = SymbolicRetriever(new_rules, top_k)
    changed = set()
    for i, q in enumerate(query_symbols):
        a = [(r.id, r.text, r.symbolic.model_dump_json()) for r in old.retrieve(q)]
        b = [(r.id, r.text, r.symbolic.model_dump_json()) for r in new.retrieve(q)]
        if a != b:
            changed.add(i)
    return changed
