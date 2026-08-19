from __future__ import annotations

import contextlib
import io
import unittest

from rimrule.consolidation.optimizer import Consolidator
from rimrule.models import ErrorType, Rule, SymbolicRule, ToolHopSample


def _sample(i: int) -> ToolHopSample:
    return ToolHopSample(id=str(i), question=f"q{i}", answer="a", tools=[])


def _rule(rule_id: str, tool_cat: str = "SEARCH") -> Rule:
    return Rule(
        id=rule_id,
        text=f"If searching then check {rule_id}.",
        error_type=ErrorType.ARGUMENTS,
        symbolic=SymbolicRule(
            domain=("GENERAL",),
            qualifier=("ALWAYS",),
            action=("CHECK",),
            strength=("MUST",),
            tool_category=(tool_cat,),
        ),
        source_failure_ids=("f1",),
    )


class ConsolidatorEditTest(unittest.TestCase):
    def test_prunes_unhelpful_rule(self):
        samples = [_sample(0)]
        rules = [_rule("r1"), _rule("r2")]

        def evaluate(candidate_rules, changed, label):
            return [True]

        events: list[dict[str, object]] = []
        optimizer = Consolidator(
            samples,
            alpha=1.0,
            evaluate=evaluate,
            affected=lambda old, new: {0},
            on_event=events.append,
        )
        final_rules, log = optimizer.run(rules)
        self.assertLessEqual(len(final_rules), len(rules))
        self.assertTrue(any(e["event"] == "round_started" for e in events))
        self.assertTrue(any(e["event"] == "candidate_evaluated" for e in events))

    def test_generalizes_tool_category(self):
        samples = [_sample(0)]
        rules = [_rule("r1", "WEATHER")]
        call_count = {"n": 0}

        def evaluate(candidate_rules, changed, label):
            call_count["n"] += 1
            return [True]

        optimizer = Consolidator(
            samples, alpha=1.0, evaluate=evaluate, affected=lambda old, new: {0}
        )
        optimizer.run(rules)
        self.assertGreaterEqual(call_count["n"], 2)

    def test_skips_already_generalized(self):
        samples = [_sample(0)]
        rules = [_rule("r1", "SEARCH_CATEGORY")]
        optimizer = Consolidator(
            samples,
            alpha=1.0,
            evaluate=lambda candidate_rules, changed, label: [True],
            affected=lambda old, new: {0},
        )
        final_rules, _ = optimizer.run(rules)
        self.assertLessEqual(len(final_rules), 1)

    def test_calls_on_accept(self):
        samples = [_sample(0)]
        rules = [_rule("r1"), _rule("r2")]

        def evaluate(candidate_rules, changed, label):
            return [True] if len(candidate_rules) < 2 else [False]

        accepted: list[dict[str, object]] = []
        optimizer = Consolidator(
            samples,
            alpha=0.001,
            evaluate=evaluate,
            affected=lambda old, new: {0},
            on_accept=accepted.append,
        )
        _, log = optimizer.run(rules)
        self.assertGreater(len(log), 0)
        self.assertEqual(len(accepted), len(log))


class ConsolidatorProgressTest(unittest.TestCase):
    def test_emits_progress_and_stops(self):
        optimizer = Consolidator(
            samples=[],
            alpha=1.0,
            evaluate=lambda rules, changed, label: [],
            affected=lambda old, new: set(),
        )
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rules, log = optimizer.run([])
        self.assertEqual(rules, [])
        self.assertEqual(log, [])
        combined = out.getvalue() + err.getvalue()
        self.assertIn("Initial exact evaluation", combined)
        self.assertIn("Final library", combined)

    def test_emits_candidate_events(self):
        events: list[dict[str, object]] = []
        optimizer = Consolidator(
            samples=[],
            alpha=1.0,
            evaluate=lambda rules, changed, label: [],
            affected=lambda old, new: set(),
            on_event=events.append,
        )
        optimizer.run([])
        self.assertEqual(events[0]["event"], "initial_evaluation_started")
        self.assertTrue(any(event["event"] == "stopped" for event in events))
