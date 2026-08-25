from __future__ import annotations

import unittest

from rimrule.induction.checks import linguistic_check


class LinguisticCheckTest(unittest.TestCase):
    def test_valid_rule(self):
        ok, reason = linguistic_check(
            "If the tool requires an ID, then verify the format."
        )
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_too_many_words(self):
        rule = "If " + " ".join(["word"] * 80) + " then do it."
        ok, reason = linguistic_check(rule, max_words=80)
        self.assertFalse(ok)
        self.assertIsNotNone(reason)
        self.assertIn("words", reason or "")

    def test_missing_if_then(self):
        ok, reason = linguistic_check(
            "Always verify the tool input schema before calling."
        )
        self.assertFalse(ok)
        self.assertIsNotNone(reason)
        self.assertIn("if", reason or "")
        self.assertIn("then", reason or "")

    def test_too_short(self):
        ok, reason = linguistic_check("If x then y.")
        self.assertFalse(ok)
        self.assertIsNotNone(reason)
        self.assertIn("short", reason or "")
