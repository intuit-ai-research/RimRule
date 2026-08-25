from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rimrule.models import GroundTruthStep, ToolDefinition, ToolHopSample


class ToolHopSchemaError(ValueError):
    """Raised when a ToolHop dataset does not match the expected schema."""


def _parse_args(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def load_toolhop(path: str | Path, strict: bool = True) -> list[ToolHopSample]:
    """Load and validate ToolHop samples from a JSON file.

    Args:
        path: Path to the ToolHop JSON dataset.
        strict: If true, raise on any malformed sample; otherwise skip it.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ToolHopSchemaError: If the file is not a list, or a sample is invalid
            under ``strict``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ToolHop dataset not found: {path}. "
            "Expected default data/toolhop/ToolHop.json"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ToolHopSchemaError("ToolHop.json must contain a JSON list")
    samples = []
    errors = []
    for idx, raw in enumerate(data):
        try:
            tools_raw = raw.get("tools") or {}
            tool_items = (
                list(tools_raw.values())
                if isinstance(tools_raw, dict)
                else list(tools_raw)
            )
            tools = [ToolDefinition.model_validate(t) for t in tool_items]
            by_key = (
                tools_raw
                if isinstance(tools_raw, dict)
                else {t.name: t.model_dump() for t in tools}
            )
            sub_task = raw.get("sub_task") or {}
            arguments = raw.get("arguments") or []
            gt = []
            for i, (key, expected) in enumerate(sub_task.items()):
                tool_obj = by_key.get(key, {})
                tool_name = tool_obj.get("name", key)
                args = None
                if i < len(arguments) and arguments[i]:
                    fn = arguments[i][0].get("function", {})
                    if strict and fn.get("name") and fn.get("name") != tool_name:
                        raise ToolHopSchemaError(
                            f"argument tool mismatch: {fn.get('name')} != {tool_name}"
                        )
                    args = _parse_args(fn.get("arguments"))
                gt.append(
                    GroundTruthStep(
                        sub_question=str(key),
                        expected_answer=str(expected),
                        tool_name=tool_name,
                        arguments=args,
                    )
                )
            samples.append(
                ToolHopSample(
                    id=str(raw.get("id", idx)),
                    question=str(raw["question"]),
                    answer=str(raw["answer"]),
                    tools=tools,
                    functions=list(raw.get("functions") or []),
                    ground_truth=gt,
                    raw=raw,
                )
            )
        except Exception as exc:
            errors.append(f"sample {idx}: {exc}")
    if errors and strict:
        raise ToolHopSchemaError("\n".join(errors[:20]))
    return samples


def schema_summary(samples: list[ToolHopSample]) -> dict[str, int]:
    """Summarize reference-trace completeness across the loaded samples."""
    return {
        "samples": len(samples),
        "complete_reference_traces": sum(
            bool(s.ground_truth)
            and all(x.arguments is not None for x in s.ground_truth)
            for s in samples
        ),
        "missing_argument_annotations": sum(
            any(x.arguments is None for x in s.ground_truth) for s in samples
        ),
        "samples_without_reference_steps": sum(not s.ground_truth for s in samples),
    }
