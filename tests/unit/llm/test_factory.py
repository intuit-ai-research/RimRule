from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_openai import ChatOpenAI

from rimrule.config import LLMConfig
from rimrule.llm.factory import get_llm


class GetLlmTest(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_openai_returns_chatopenai(self):
        llm = get_llm(LLMConfig(provider="openai", model="gpt-4o"))
        self.assertIsInstance(llm, ChatOpenAI)
        self.assertEqual(llm.model_name, "gpt-4o")

    def test_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            get_llm(LLMConfig(provider="anthropic", model="claude"))
