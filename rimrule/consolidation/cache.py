from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from rimrule.models import Rule, ToolHopSample, Trace


class EvaluationCache:
    """SQLite-backed cache of agent rollouts keyed by (sample, rules, signature)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS evaluations "
            "(cache_key TEXT PRIMARY KEY, trace_json TEXT NOT NULL)"
        )

    @staticmethod
    def key(sample: ToolHopSample, rules: list[Rule], execution_signature: str) -> str:
        """Return the cache key for a sample under a specific rule set and signature."""
        payload = {
            "sample": sample.id,
            "rules": [[r.id, r.text, r.symbolic.model_dump()] for r in rules],
            "execution": execution_signature,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get_or_run_with_status(
        self,
        sample: ToolHopSample,
        rules: list[Rule],
        execution_signature: str,
        run: Callable[[], Trace],
    ) -> tuple[Trace, bool]:
        """Return ``(trace, was_cached)`` for progress and diagnostics."""
        key = self.key(sample, rules, execution_signature)
        row = self.db.execute(
            "SELECT trace_json FROM evaluations WHERE cache_key=?", (key,)
        ).fetchone()
        if row:
            return Trace.model_validate_json(row[0]), True
        trace = run()
        self.db.execute(
            "INSERT INTO evaluations VALUES (?,?)", (key, trace.model_dump_json())
        )
        self.db.commit()
        return trace, False

    def get_or_run(
        self,
        sample: ToolHopSample,
        rules: list[Rule],
        execution_signature: str,
        run: Callable[[], Trace],
    ) -> Trace:
        """Return the cached or freshly computed trace, discarding the hit flag."""
        trace, _ = self.get_or_run_with_status(sample, rules, execution_signature, run)
        return trace

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.db.close()

    def __enter__(self) -> EvaluationCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
