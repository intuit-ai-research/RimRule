from __future__ import annotations

import unittest

from rimrule.toolhop.evaluator import extract_answer, is_correct


class EvaluatorTest(unittest.TestCase):
    def test_answer_tag(self):
        self.assertEqual(extract_answer("thinking <answer>42</answer>"), "42")
        self.assertTrue(is_correct("<answer>42.0</answer>", "42"))
