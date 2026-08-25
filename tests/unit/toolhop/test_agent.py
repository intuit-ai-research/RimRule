from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from rimrule.models import GroundTruthStep, ToolDefinition, ToolHopSample
from rimrule.toolhop.agent import ToolHopAgent, build_reference_trace
from tests.unit._fakes import FakeChatModel


def _sample() -> ToolHopSample:
    return ToolHopSample(
        id="s1",
        question="What is 2+3?",
        answer="5",
        tools=[
            ToolDefinition(
                name="add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            )
        ],
        functions=["def add(a, b):\n    return a + b\n"],
    )


def _tool_call(call_id: str, name: str = "add", **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class ToolHopAgentRunTest(unittest.TestCase):
    def test_final_answer_no_tools(self):
        llm = FakeChatModel([AIMessage(content="<answer>5</answer>")])
        trace = ToolHopAgent(llm, max_turns=3).run(_sample())
        self.assertTrue(trace.correct)
        self.assertEqual(trace.terminated_reason, "final_answer")

    def test_tool_call_then_answer(self):
        llm = FakeChatModel(
            [
                AIMessage(content="", tool_calls=[_tool_call("c1", a=2, b=3)]),
                AIMessage(content="<answer>5</answer>"),
            ]
        )
        trace = ToolHopAgent(llm, max_turns=5, tool_timeout_seconds=5.0).run(_sample())
        self.assertTrue(trace.correct)
        self.assertEqual(len(trace.steps), 3)

    def test_max_turns_exceeded(self):
        llm = FakeChatModel(
            [
                AIMessage(content="", tool_calls=[_tool_call(f"c{i}", a=1, b=1)])
                for i in range(5)
            ]
        )
        trace = ToolHopAgent(llm, max_turns=2, tool_timeout_seconds=5.0).run(_sample())
        self.assertFalse(trace.correct)
        self.assertEqual(trace.terminated_reason, "max_turns")

    def test_tool_error_captured(self):
        llm = FakeChatModel(
            [
                AIMessage(
                    content="", tool_calls=[_tool_call("c1", name="nonexistent")]
                ),
                AIMessage(content="<answer>error</answer>"),
            ]
        )
        trace = ToolHopAgent(llm, max_turns=3, tool_timeout_seconds=5.0).run(_sample())
        error_steps = [s for s in trace.steps if s.error is not None]
        self.assertEqual(len(error_steps), 1)

    def test_with_rules(self):
        llm = FakeChatModel([AIMessage(content="<answer>5</answer>")])
        trace = ToolHopAgent(llm, max_turns=3).run(
            _sample(), rules=["If adding, then check types."]
        )
        self.assertTrue(trace.correct)


class BuildReferenceTraceTest(unittest.TestCase):
    def test_build_reference_trace(self):
        sample = ToolHopSample(
            id="s1",
            question="q",
            answer="42",
            tools=[],
            ground_truth=[
                GroundTruthStep(
                    sub_question="sub",
                    expected_answer="42",
                    tool_name="calc",
                    arguments={"x": 1},
                )
            ],
        )
        trace = build_reference_trace(sample)
        self.assertTrue(trace.correct)
        self.assertEqual(trace.final_answer, "42")
        self.assertEqual(len(trace.steps), 2)
