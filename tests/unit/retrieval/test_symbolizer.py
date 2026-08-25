from __future__ import annotations

import unittest

from rimrule.models import SymbolicRule, ToolHopSample, Vocabulary
from rimrule.retrieval.symbolizer import QuerySymbolizer
from tests.unit._fakes import StructuredFakeChatModel


class QuerySymbolizerTest(unittest.TestCase):
    def test_symbolize(self):
        symbolic = SymbolicRule(
            domain=("GENERAL",),
            qualifier=("SCHEMA_MATCH",),
            action=("VERIFY_INPUT",),
            strength=("MUST",),
            tool_category=("SEARCH",),
        )
        vocab = Vocabulary(
            domain=("GENERAL",),
            qualifier=("SCHEMA_MATCH",),
            action=("VERIFY_INPUT",),
            strength=("MUST",),
            tool_category=("SEARCH",),
        )
        symbolizer = QuerySymbolizer(StructuredFakeChatModel([symbolic]), vocab)
        sample = ToolHopSample(id="s1", question="q", answer="a", tools=[])
        result = symbolizer.symbolize(sample)
        self.assertEqual(result.domain, ("GENERAL",))
        self.assertEqual(result.action, ("VERIFY_INPUT",))
