from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_base(
    path: Path, _seen: frozenset[Path] = frozenset()
) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``base:`` parent by deep merge.

    Raises:
        ValueError: If the ``base:`` chain forms a cycle.
    """
    resolved = path.resolve()
    if resolved in _seen:
        raise ValueError(f"Cyclic config base reference at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = data.pop("base", None)
    if base_ref is None:
        return data
    base_path = (path.parent / base_ref).resolve()
    return _deep_merge(_load_yaml_with_base(base_path, _seen | {resolved}), data)


class LLMConfig(BaseModel):
    """Provider, model, and retry settings for one LLM role."""

    provider: str = "openai"
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    temperature: float = 0.0
    reasoning_effort: str | None = None
    max_output_tokens: int = 8192
    timeout_seconds: float = 120.0
    max_retries: int = 6
    retry_initial_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 60.0


class DatasetConfig(BaseModel):
    """Location and validation strictness of the ToolHop dataset.

    The dataset is not committed; fetch it into ``path`` from object storage
    before running the pipeline (see the README).
    """

    path: str = "data/toolhop/ToolHop.json"
    strict_schema: bool = True


class AgentConfig(BaseModel):
    """Runtime limits for the ToolHop agent loop."""

    max_turns: int = 10
    tool_timeout_seconds: float = 10.0
    scenario: str = "react"


class RimRuleConfig(BaseModel):
    """Top-level, YAML-driven configuration for the RimRule pipeline."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    agent_llm: LLMConfig
    rule_llm: LLMConfig | None = None
    vocabulary_llm: LLMConfig | None = None
    agent: AgentConfig = Field(default_factory=AgentConfig)
    artifacts_dir: str = "artifacts"
    max_rule_words: int = 80
    max_atomic_fixes_per_failure: int = 3
    vocabulary_randomizations: int = 5
    retrieval_top_k: int = 5
    alpha: float = 0.1

    @classmethod
    def load(cls, path: str | Path) -> RimRuleConfig:
        """Load and validate a configuration from a YAML file.

        A config may set ``base: <relative-path>`` to inherit from another YAML
        file; the child's keys are deep-merged over the base so shared defaults
        live in one place.
        """
        return cls.model_validate(_load_yaml_with_base(Path(path)))

    def with_artifacts_dir(self, path: str | Path) -> RimRuleConfig:
        """Return a copy writing all stage artifacts beneath ``path``."""
        resolved = str(Path(path).expanduser())
        return self.model_copy(update={"artifacts_dir": resolved})

    def artifact_path(self, name: str) -> Path:
        """Return the path to artifact ``name``, creating the directory if needed."""
        p = Path(self.artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / name
