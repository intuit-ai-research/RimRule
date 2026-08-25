from __future__ import annotations

import unittest

from rimrule.canonicalization.canonicalizer import Canonicalizer, ensure_unknown
from rimrule.models import UNKNOWN, CandidateRule, ErrorType, SymbolicRule, Vocabulary
from tests.unit._fakes import StructuredFakeChatModel


def _vocabulary() -> Vocabulary:
    return Vocabulary(
        domain=("GENERAL",),
        qualifier=("SCHEMA_MATCH",),
        action=("VERIFY_INPUT",),
        strength=("MUST",),
        tool_category=("SEARCH",),
    )


def _candidate() -> CandidateRule:
    return CandidateRule(
        id="r1",
        text="If calling a tool, then verify its input schema.",
        error_type=ErrorType.ARGUMENTS,
        source_failure_id="f1",
    )


def _translation(action: str) -> SymbolicRule:
    return SymbolicRule(
        domain=("GENERAL",),
        qualifier=("SCHEMA_MATCH",),
        action=(action,),
        strength=("MUST",),
        tool_category=("SEARCH",),
    )


class CanonicalizerTranslateTest(unittest.TestCase):
    def test_retries_unknown_label_then_accepts_allowed_label(self):
        llm = StructuredFakeChatModel(
            [_translation("MATCH_TOOL_INPUT_SCHEMA"), _translation("VERIFY_INPUT")]
        )
        rule = Canonicalizer(llm).translate(_candidate(), _vocabulary())
        self.assertEqual(rule.symbolic.action, ("VERIFY_INPUT",))

    def test_falls_back_to_unknown_after_retries(self):
        bad = _translation("MATCH_TOOL_INPUT_SCHEMA")
        llm = StructuredFakeChatModel([bad, bad, bad, bad])
        rule = Canonicalizer(llm, translation_attempts=3).translate(
            _candidate(), _vocabulary()
        )
        self.assertEqual(rule.symbolic.action, (UNKNOWN,))

    def test_final_attempt_error_is_logged_and_falls_back(self):
        bad = _translation("MATCH_TOOL_INPUT_SCHEMA")
        # Retries return bad output; the final fallback call raises.
        llm = StructuredFakeChatModel([bad, RuntimeError("llm down")])
        with self.assertLogs(
            "rimrule.canonicalization.canonicalizer", level="ERROR"
        ) as logs:
            rule = Canonicalizer(llm, translation_attempts=1).translate(
                _candidate(), _vocabulary()
            )
        # A failed final attempt yields an empty (all-UNKNOWN-eligible) rule,
        # and the error is logged rather than swallowed silently.
        self.assertEqual(rule.id, "r1")
        self.assertEqual(rule.symbolic.action, ())
        self.assertTrue(any("falling back" in m for m in logs.output))


class InduceVocabularyTest(unittest.TestCase):
    def test_picks_most_compact_vocabulary(self):
        small = Vocabulary(
            domain=("GENERAL",),
            qualifier=(),
            action=("VERIFY_INPUT",),
            strength=("MUST",),
            tool_category=("SEARCH",),
        )
        large = Vocabulary(
            domain=("GENERAL", "SPECIFIC", "EXTRA"),
            qualifier=("SCHEMA_MATCH",),
            action=("VERIFY_INPUT", "CHECK"),
            strength=("MUST",),
            tool_category=("SEARCH",),
        )
        llm = StructuredFakeChatModel([large, small])
        vocab = Canonicalizer(llm).induce_vocabulary([_candidate()], randomizations=2)
        self.assertNotIn("SPECIFIC", vocab.domain)


class EnsureUnknownTest(unittest.TestCase):
    def test_unknown_is_added_to_every_vocabulary_field(self):
        fixed = ensure_unknown(_vocabulary())
        for field in Vocabulary.model_fields:
            self.assertIn(UNKNOWN, getattr(fixed, field))
