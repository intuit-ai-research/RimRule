from __future__ import annotations

import unittest

from rimrule.consolidation.affected import affected_samples
from rimrule.models import ErrorType, Rule, SymbolicRule


def _rule(rule_id: str, domain: str = "GENERAL") -> Rule:
    return Rule(
        id=rule_id,
        text=f"If {domain.lower()} then act.",
        error_type=ErrorType.ARGUMENTS,
        symbolic=SymbolicRule(
            domain=(domain,),
            qualifier=("ALWAYS",),
            action=("CHECK",),
            strength=("MUST",),
            tool_category=("SEARCH",),
        ),
        source_failure_ids=("f1",),
    )


def _query(domain: str = "GENERAL") -> SymbolicRule:
    return SymbolicRule(
        domain=(domain,),
        qualifier=("ALWAYS",),
        action=("CHECK",),
        strength=("MUST",),
        tool_category=("SEARCH",),
    )


class AffectedSamplesTest(unittest.TestCase):
    def test_identical_rules_no_change(self):
        rules = [_rule("r1")]
        result = affected_samples(rules, rules, [_query()], top_k=5)
        self.assertEqual(result, set())

    def test_removing_rule_changes_affected(self):
        old = [_rule("r1"), _rule("r2", "SPECIFIC")]
        new = [_rule("r1")]
        queries = [_query(), _query("SPECIFIC")]
        result = affected_samples(old, new, queries, top_k=5)
        self.assertIn(1, result)
