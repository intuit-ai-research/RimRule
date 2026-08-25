from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rimrule.consolidation.cache import EvaluationCache
from rimrule.models import ErrorType, Rule, SymbolicRule, ToolHopSample, Trace


def _sample() -> ToolHopSample:
    return ToolHopSample(id="s1", question="q", answer="a", tools=[])


def _rule() -> Rule:
    return Rule(
        id="r1",
        text="If x then y.",
        error_type=ErrorType.ARGUMENTS,
        symbolic=SymbolicRule(
            domain=("GENERAL",),
            qualifier=("ALWAYS",),
            action=("CHECK",),
            strength=("MUST",),
            tool_category=("SEARCH",),
        ),
        source_failure_ids=("f1",),
    )


class EvaluationCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _cache(self) -> EvaluationCache:
        cache = EvaluationCache(self.tmp_path / "eval.sqlite")
        self.addCleanup(cache.close)
        return cache

    def test_cache_miss_then_hit(self):
        cache = self._cache()
        sample, rules = _sample(), [_rule()]
        call_count = {"n": 0}

        def run() -> Trace:
            call_count["n"] += 1
            return Trace(correct=True, final_answer="a")

        trace1, cached1 = cache.get_or_run_with_status(sample, rules, "sig", run)
        self.assertFalse(cached1)
        self.assertTrue(trace1.correct)
        self.assertEqual(call_count["n"], 1)

        trace2, cached2 = cache.get_or_run_with_status(sample, rules, "sig", run)
        self.assertTrue(cached2)
        self.assertTrue(trace2.correct)
        self.assertEqual(call_count["n"], 1)

    def test_get_or_run_shorthand(self):
        cache = self._cache()
        trace = cache.get_or_run(
            _sample(), [_rule()], "sig", lambda: Trace(correct=False)
        )
        self.assertFalse(trace.correct)

    def test_close_and_context_manager(self):
        with EvaluationCache(self.tmp_path / "eval.sqlite") as cache:
            cache.get_or_run(_sample(), [_rule()], "sig", lambda: Trace(correct=True))
        # Connection is closed on exit; using it again raises.
        with self.assertRaises(sqlite3.ProgrammingError):
            cache.db.execute("SELECT 1")

    def test_different_signatures_miss(self):
        cache = self._cache()
        sample, rules = _sample(), [_rule()]
        cache.get_or_run(sample, rules, "sig_a", lambda: Trace(correct=True))
        _, cached = cache.get_or_run_with_status(
            sample, rules, "sig_b", lambda: Trace(correct=False)
        )
        self.assertFalse(cached)
