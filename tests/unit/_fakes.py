from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


class FakeChatModel(BaseChatModel):
    """A chat model that replays queued AIMessages, one per invoke.

    Supports ``bind_tools`` (a no-op that returns the model) so it can stand in
    for a tool-calling ``ChatOpenAI`` in agent tests.
    """

    responses: list[AIMessage]
    index: int = 0

    def __init__(self, responses: list[AIMessage], **kwargs: Any):
        super().__init__(responses=responses, **kwargs)  # type: ignore[call-arg]

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


class StructuredFakeChatModel(BaseChatModel):
    """A chat model whose ``with_structured_output`` replays queued objects.

    Stands in for a ``ChatOpenAI`` used via ``.with_structured_output(Model)``.
    A queued ``Exception`` is raised instead of returned.
    """

    outputs: list[Any]
    index: int = 0

    def __init__(self, outputs: list[Any], **kwargs: Any):
        super().__init__(outputs=outputs, **kwargs)  # type: ignore[call-arg]

    @property
    def _llm_type(self) -> str:
        return "structured-fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def with_structured_output(
        self, schema: Any, **kwargs: Any
    ) -> Runnable[Any, BaseModel]:
        def _next(_: Any) -> Any:
            value = self.outputs[self.index]
            self.index += 1
            if isinstance(value, Exception):
                raise value
            return value

        return RunnableLambda(_next)
