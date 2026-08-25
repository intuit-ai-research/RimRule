from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

SYMBOLIC_FIELDS: tuple[str, ...] = (
    "domain",
    "qualifier",
    "action",
    "strength",
    "tool_category",
)
"""The five symbolic-vocabulary axes shared by :class:`SymbolicRule` and
:class:`Vocabulary`. Every place that iterates the axes must reference this
constant so a renamed or added axis cannot silently desync retrieval, MDL
scoring, and canonicalization."""

UNKNOWN = "UNKNOWN"
"""Closed-vocabulary fallback token for a symbol that cannot be translated."""


class ErrorType(StrEnum):
    """Root-cause category of a tool-use failure."""

    DECOMPOSITION = "decomposition"
    TOOL_SELECTION = "tool_selection"
    ARGUMENTS = "arguments"


class ToolDefinition(BaseModel):
    """A tool the agent may call, in JSON-schema function form.

    This is the dataset's tool schema; ``chat_schema`` renders it in the
    OpenAI-function shape that ``ChatOpenAI.bind_tools`` accepts.
    """

    name: str = Field(description="Unique tool name.")
    description: str = Field(default="", description="What the tool does.")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema of the tool's arguments."
    )

    def chat_schema(self) -> dict[str, Any]:
        """Return the tool as an OpenAI chat-completions function schema."""
        return {"type": "function", "function": self.model_dump()}


class ToolCall(BaseModel):
    """A concrete tool invocation with resolved arguments."""

    id: str = Field(description="Provider-assigned call id.")
    name: str = Field(description="Name of the invoked tool.")
    arguments: dict[str, Any] = Field(description="Arguments passed to the tool.")


class TraceStep(BaseModel):
    """One recorded turn of a rollout: a message plus any tool call and result."""

    role: str = Field(description="Message role: system/user/assistant/tool.")
    content: str = Field(default="", description="The message text.")
    tool_call: ToolCall | None = Field(
        default=None, description="The tool call made at this step, if any."
    )
    observation: Any = Field(default=None, description="The tool's returned value.")
    error: str | None = Field(
        default=None, description="Tool error text, if it failed."
    )


class Trace(BaseModel):
    """A full agent rollout on one sample."""

    steps: list[TraceStep] = Field(
        default_factory=list, description="Ordered steps of the rollout."
    )
    final_answer: str | None = Field(
        default=None, description="The agent's final answer, if it produced one."
    )
    correct: bool = Field(
        default=False, description="Whether the final answer matched the reference."
    )
    terminated_reason: str | None = Field(
        default=None, description="Why the rollout ended (final_answer/max_turns/...)."
    )


class GroundTruthStep(BaseModel):
    """One reference sub-task with its expected tool call and answer."""

    sub_question: str = Field(description="The decomposed sub-question.")
    expected_answer: str = Field(
        description="The reference answer to the sub-question."
    )
    tool_name: str = Field(description="The tool the reference uses for this step.")
    arguments: dict[str, Any] | None = Field(
        default=None, description="Reference tool arguments, if annotated."
    )


class ToolHopSample(BaseModel):
    """A single ToolHop task: question, answer, tools, and reference steps."""

    id: str = Field(description="Sample identifier.")
    question: str = Field(description="The multi-hop question to answer.")
    answer: str = Field(description="The ground-truth final answer.")
    tools: list[ToolDefinition] = Field(description="Tools available for this sample.")
    functions: list[str] = Field(
        default_factory=list, description="Python source of the executable tools."
    )
    ground_truth: list[GroundTruthStep] = Field(
        default_factory=list, description="Reference decomposition steps."
    )
    raw: dict[str, Any] = Field(
        default_factory=dict, exclude=True, description="Original dataset record."
    )


class FailureExperience(BaseModel):
    """A failed rollout paired with its reference trace, used for induction."""

    sample: ToolHopSample = Field(description="The sample that was attempted.")
    failed_trace: Trace = Field(description="The agent's failing rollout.")
    reference_trace: Trace = Field(description="The ground-truth rollout.")


class CandidateRule(BaseModel):
    """A natural-language rule proposed from one failure, before canonicalization."""

    id: str = Field(description="Stable rule identifier.")
    text: str = Field(description="The rule text in if-then form.")
    error_type: ErrorType = Field(description="The failure category it addresses.")
    source_failure_id: str = Field(description="Sample id the rule was induced from.")
    generation_round: int = Field(
        default=0, description="Which correction round proposed this rule."
    )


class RuleProposal(BaseModel):
    """Structured LLM output for a single induced rule."""

    rule: str = Field(description="The proposed rule in if-then form.")
    error_type: ErrorType = Field(description="The root-cause failure category.")


class SymbolicRule(BaseModel):
    """A rule (or query) expressed as symbols over the closed vocabulary axes."""

    domain: tuple[str, ...] = Field(default=(), description="Domain symbols.")
    qualifier: tuple[str, ...] = Field(default=(), description="Qualifier symbols.")
    action: tuple[str, ...] = Field(default=(), description="Action symbols.")
    strength: tuple[str, ...] = Field(default=(), description="Strength symbols.")
    tool_category: tuple[str, ...] = Field(
        default=(), description="Tool-category symbols."
    )

    def size(self) -> int:
        """Return the total number of symbols across all vocabulary axes."""
        return sum(len(getattr(self, field)) for field in SYMBOLIC_FIELDS)

    def has_unknown(self) -> bool:
        """Return whether any axis contains the closed-vocabulary UNKNOWN token."""
        return any(UNKNOWN in getattr(self, field) for field in SYMBOLIC_FIELDS)


class Rule(BaseModel):
    """A canonicalized rule: its text plus its symbolic translation."""

    id: str = Field(description="Stable rule identifier.")
    text: str = Field(description="The rule text in if-then form.")
    error_type: ErrorType = Field(description="The failure category it addresses.")
    symbolic: SymbolicRule = Field(description="The rule's symbolic translation.")
    source_failure_ids: tuple[str, ...] = Field(
        default=(), description="Sample ids that motivated this rule."
    )


class Vocabulary(BaseModel):
    """The closed set of allowed symbols on each canonicalization axis."""

    domain: tuple[str, ...] = Field(description="Allowed domain symbols.")
    qualifier: tuple[str, ...] = Field(description="Allowed qualifier symbols.")
    action: tuple[str, ...] = Field(description="Allowed action symbols.")
    strength: tuple[str, ...] = Field(description="Allowed strength symbols.")
    tool_category: tuple[str, ...] = Field(description="Allowed tool-category symbols.")
