from __future__ import annotations

import unittest

from rimrule.consolidation.mdl import mdl_score, rule_length
from rimrule.models import ErrorType, Rule, SymbolicRule


class MdlTest(unittest.TestCase):
    def test_rule_length_and_mdl(self):
        r = Rule(
            id="r",
            text="If x then y",
            error_type=ErrorType.DECOMPOSITION,
            symbolic=SymbolicRule(domain=("D",), action=("A",), strength=("M",)),
        )
        self.assertEqual(rule_length(r), 3)
        total, model, data = mdl_score([r], [True, False], 0.5)
        self.assertEqual(model, 1.5)
        self.assertGreater(total, model)
        self.assertGreater(data, 0)
