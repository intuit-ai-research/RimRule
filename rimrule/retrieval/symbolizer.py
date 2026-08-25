from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from rimrule.induction.prompts import QUERY_SYMBOL_PROMPT
from rimrule.models import SymbolicRule, ToolHopSample, Vocabulary


class QuerySymbolizer:
    """Translate a ToolHop query into the frozen symbolic vocabulary."""

    def __init__(self, llm: BaseChatModel, vocabulary: Vocabulary):
        self.llm = llm.with_structured_output(SymbolicRule, method="function_calling")
        self.vocabulary = vocabulary

    def symbolize(self, sample: ToolHopSample) -> SymbolicRule:
        """Return the symbolic representation of ``sample``'s query and tools."""
        prompt = QUERY_SYMBOL_PROMPT.format(
            vocabulary=self.vocabulary.model_dump_json(indent=2),
            query=sample.question,
            tools="\n".join(f"{t.name}: {t.description}" for t in sample.tools),
        )
        result = self.llm.invoke([HumanMessage(content=prompt)])
        if not isinstance(result, SymbolicRule):
            raise TypeError(f"Expected SymbolicRule, got {type(result).__name__}")
        return result
