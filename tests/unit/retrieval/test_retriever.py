from __future__ import annotations

import unittest

from rimrule.models import ErrorType, Rule, SymbolicRule
from rimrule.retrieval.retriever import SymbolicRetriever


class SymbolicRetrieverTest(unittest.TestCase):
    def test_scope_filter_and_decomposition_fallback(self):
        rules = [
            Rule(
                id="a",
                text="a",
                error_type=ErrorType.TOOL_SELECTION,
                symbolic=SymbolicRule(domain=("D",), tool_category=("CAT",)),
            ),
            Rule(
                id="b",
                text="b",
                error_type=ErrorType.DECOMPOSITION,
                symbolic=SymbolicRule(domain=("OTHER",)),
            ),
        ]
        got = SymbolicRetriever(rules, 5).retrieve(
            SymbolicRule(domain=("D",), tool_category=("CAT",))
        )
        self.assertEqual([r.id for r in got], ["a", "b"])
