from __future__ import annotations

import unittest

from rimrule.induction.generator import RuleGenerator
from rimrule.models import (
    ErrorType,
    FailureExperience,
    RuleProposal,
    ToolHopSample,
    Trace,
)
from tests.unit._fakes import StructuredFakeChatModel


def _experience() -> FailureExperience:
    sample = ToolHopSample(id="s1", question="q", answer="a", tools=[])
    return FailureExperience(
        sample=sample,
        failed_trace=Trace(correct=False),
        reference_trace=Trace(correct=True),
    )


class RuleGeneratorTest(unittest.TestCase):
    def test_propose_returns_candidate(self):
        proposal = RuleProposal(
            rule="If the tool requires a date, then format it as YYYY-MM-DD.",
            error_type=ErrorType.ARGUMENTS,
        )
        gen = RuleGenerator(StructuredFakeChatModel([proposal]), max_words=80)
        rule = gen.propose(_experience())
        self.assertTrue(rule.text.startswith("If the tool"))
        self.assertEqual(rule.source_failure_id, "s1")
        self.assertTrue(rule.id.startswith("rule_"))

    def test_propose_rejects_long_rule(self):
        long_text = "If " + " ".join(["word"] * 90) + " then do it."
        proposal = RuleProposal(rule=long_text, error_type=ErrorType.ARGUMENTS)
        gen = RuleGenerator(StructuredFakeChatModel([proposal]), max_words=80)
        with self.assertRaisesRegex(ValueError, "words"):
            gen.propose(_experience())

    def test_propose_preserves_error_type(self):
        proposal = RuleProposal(
            rule="If the API returns an error, then retry with corrected parameters.",
            error_type=ErrorType.TOOL_SELECTION,
        )
        gen = RuleGenerator(StructuredFakeChatModel([proposal]), max_words=80)
        rule = gen.propose(_experience())
        self.assertEqual(rule.error_type, ErrorType.TOOL_SELECTION)
