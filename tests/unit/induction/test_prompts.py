from __future__ import annotations

import unittest

from rimrule.induction.prompts import RULE_PROMPT


class RulePromptTest(unittest.TestCase):
    def test_rule_prompt_renders_literal_json_example(self):
        rendered = RULE_PROMPT.format(
            query="Who wrote the book?",
            tools="[]",
            failed="{}",
            reference="{}",
            previous="None",
        )
        self.assertIn(
            '{"rule": "If ... then ...", '
            '"error_type": "decomposition|tool_selection|arguments"}',
            rendered,
        )
        self.assertIn("Who wrote the book?", rendered)
