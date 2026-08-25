from __future__ import annotations

import json
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from rimrule.models import ToolCall, ToolHopSample, Trace, TraceStep
from rimrule.toolhop.evaluator import is_correct
from rimrule.toolhop.executor import execute_tool

SYSTEM = """You are a tool-using agent. Solve the question only by calling the available tools.
Use tool feedback to continue step by step. Return the final answer as <answer>...</answer> and keep it short.
Dates must use YYYY-MM-DD; names use Firstname Lastname; numeric answers contain only the number."""


def _fallback_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:10]}"


class ToolHopAgent:
    """A ReAct-style agent that answers a ToolHop sample by calling its tools."""

    def __init__(
        self,
        llm: BaseChatModel,
        max_turns: int = 10,
        tool_timeout_seconds: float = 10.0,
    ):
        self.llm = llm
        self.max_turns = max_turns
        self.tool_timeout_seconds = tool_timeout_seconds

    def run(self, sample: ToolHopSample, rules: list[str] | None = None) -> Trace:
        """Run the agent on ``sample`` under optional guidance ``rules``.

        Returns:
            The full rollout, with ``correct`` set once the agent gives a final
            answer or exhausts its turn budget.
        """
        rule_text = "\n".join(f"Rule: {r}" for r in (rules or []))
        user = f"{rule_text}\nQuestion: {sample.question}".strip()
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=user),
        ]
        model = self.llm.bind_tools([t.chat_schema() for t in sample.tools])
        trace = Trace()
        for _ in range(self.max_turns):
            response = model.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError(f"Expected AIMessage, got {type(response).__name__}")
            messages.append(response)
            trace.steps.append(
                TraceStep(role="assistant", content=_text(response.content))
            )
            if not response.tool_calls:
                answer = _text(response.content)
                trace.final_answer = answer
                trace.correct = is_correct(answer, sample.answer)
                trace.terminated_reason = "final_answer"
                return trace
            for raw in response.tool_calls:
                call = ToolCall(
                    id=raw.get("id") or _fallback_call_id(),
                    name=raw["name"],
                    arguments=raw["args"],
                )
                try:
                    observation = execute_tool(
                        sample.functions,
                        call.name,
                        call.arguments,
                        self.tool_timeout_seconds,
                    )
                    content = json.dumps(observation, ensure_ascii=False)
                    step = TraceStep(
                        role="tool",
                        content=content,
                        tool_call=call,
                        observation=observation,
                    )
                except Exception as exc:
                    content = f"an error occurred when calling {call.name}: {exc}"
                    step = TraceStep(
                        role="tool", content=content, tool_call=call, error=str(exc)
                    )
                messages.append(ToolMessage(content=content, tool_call_id=call.id))
                trace.steps.append(step)
        trace.terminated_reason = "max_turns"
        trace.correct = False
        return trace


def _text(content: object) -> str:
    """Flatten LangChain message content (str or content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts)
    return str(content)


def build_reference_trace(sample: ToolHopSample) -> Trace:
    """Build a synthetic correct trace from a sample's ground-truth steps."""
    trace = Trace(
        correct=True, final_answer=sample.answer, terminated_reason="ground_truth"
    )
    for i, gt in enumerate(sample.ground_truth):
        call = ToolCall(id=f"gt_{i}", name=gt.tool_name, arguments=gt.arguments or {})
        trace.steps.append(
            TraceStep(
                role="assistant", content=f"Subtask: {gt.sub_question}", tool_call=call
            )
        )
        trace.steps.append(
            TraceStep(
                role="tool", content=gt.expected_answer, observation=gt.expected_answer
            )
        )
    return trace
