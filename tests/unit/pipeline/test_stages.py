from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

from rimrule.config import DatasetConfig, LLMConfig, RimRuleConfig
from rimrule.models import (
    CandidateRule,
    ErrorType,
    FailureExperience,
    ToolHopSample,
    Trace,
)
from rimrule.pipeline import stages


def _sample(i: int) -> ToolHopSample:
    return ToolHopSample(
        id=str(i), question=f"q{i}", answer="a", tools=[], functions=[], ground_truth=[]
    )


def _failure(i: int) -> FailureExperience:
    return FailureExperience(
        sample=_sample(i),
        failed_trace=Trace(correct=False, final_answer="bad"),
        reference_trace=Trace(correct=True, final_answer="a"),
    )


class _Progress:
    """Minimal tqdm stand-in that records the postfixes it was given."""

    last: _Progress | None = None

    def __init__(self, rows, **kwargs):
        self.rows = rows
        self.postfixes: list[dict[str, object]] = []
        type(self).last = self

    def __iter__(self):
        return iter(self.rows)

    def set_postfix(self, **kwargs):
        kwargs.pop("refresh", None)
        self.postfixes.append(kwargs)


class _CollectAgent:
    outcomes = [True, False, False]

    def __init__(self, *args, **kwargs):
        self.index = 0

    def run(self, sample):
        correct = self.outcomes[self.index]
        self.index += 1
        return Trace(correct=correct, final_answer="ok" if correct else "bad")


class _RecordingAgent:
    calls: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    def run(self, sample):
        self.calls.append(sample.id)
        return Trace(correct=sample.id == "2")


class _FixingAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, sample, rules=None):
        return Trace(correct=True, final_answer="fixed")


class _FakeGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def propose(self, experience, previous):
        return CandidateRule(
            id=f"rule-{experience.sample.id}-{len(previous)}",
            text="If a tool result is needed, then call the relevant tool.",
            error_type=ErrorType.TOOL_SELECTION,
            source_failure_id=experience.sample.id,
            generation_round=len(previous),
        )


class _StageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        _Progress.last = None

    def _config(self, **kwargs) -> RimRuleConfig:
        return RimRuleConfig(
            dataset=DatasetConfig(path="unused.json", strict_schema=True),
            agent_llm=LLMConfig(provider="openai", model="fake"),
            artifacts_dir=str(self.tmp_path),
            **kwargs,
        )


@patch("rimrule.pipeline.stages.tqdm", lambda rows, **kwargs: _Progress(rows))
@patch("rimrule.pipeline.stages.get_llm", lambda config: object())
class CollectTest(_StageTest):
    @patch("rimrule.pipeline.stages.ToolHopAgent", _CollectAgent)
    @patch(
        "rimrule.pipeline.stages.load_toolhop",
        lambda *a, **k: [_sample(1), _sample(2), _sample(3)],
    )
    def test_collect_streams_each_failure(self):
        failures = stages.collect(self._config())
        path = self.tmp_path / "failures.jsonl"
        self.assertEqual(len(failures), 2)
        self.assertTrue(path.exists())
        self.assertEqual(
            len([line for line in path.read_text().splitlines() if line]), 2
        )

    @patch("rimrule.pipeline.stages.ToolHopAgent", _RecordingAgent)
    def test_collect_resumes_without_rerunning_completed(self):
        samples = [_sample(1), _sample(2), _sample(3)]
        prior = FailureExperience(
            sample=samples[0],
            failed_trace=Trace(correct=False),
            reference_trace=Trace(),
        )
        (self.tmp_path / "failures.jsonl").write_text(
            prior.model_dump_json() + "\n", encoding="utf-8"
        )
        (self.tmp_path / "collect_state.json").write_text(
            '{"status":"partial","completed_sample_ids":["1","2"],'
            '"successes":1,"failures":1,"total":3}',
            encoding="utf-8",
        )
        _RecordingAgent.calls = []
        with patch("rimrule.pipeline.stages.load_toolhop", lambda *a, **k: samples):
            failures = stages.collect(self._config())
        self.assertEqual(_RecordingAgent.calls, ["3"])
        self.assertEqual([f.sample.id for f in failures], ["1", "3"])
        state = (self.tmp_path / "collect_state.json").read_text(encoding="utf-8")
        self.assertIn('"status": "complete"', state)


@patch("rimrule.pipeline.stages.tqdm", lambda rows, **kwargs: _Progress(rows))
@patch("rimrule.pipeline.stages.get_llm", lambda config: object())
@patch("rimrule.pipeline.stages.ToolHopAgent", _FixingAgent)
@patch("rimrule.pipeline.stages.RuleGenerator", _FakeGenerator)
class InduceTest(_StageTest):
    def test_induce_streams_rules_and_reports_progress(self):
        config = self._config(max_atomic_fixes_per_failure=2)
        (self.tmp_path / "failures.jsonl").write_text(
            "\n".join(_failure(i).model_dump_json() for i in range(3)) + "\n",
            encoding="utf-8",
        )
        rules = stages.induce(config)
        self.assertEqual(len(rules), 3)
        output = self.tmp_path / "candidate_rules.jsonl"
        self.assertTrue(output.exists())
        self.assertEqual(
            len([line for line in output.read_text().splitlines() if line]), 3
        )
        self.assertIsNotNone(_Progress.last)
        progress = cast(_Progress, _Progress.last)
        self.assertEqual(
            progress.postfixes[-1],
            {"proposed": 3, "accepted": 3, "corrected": 3, "unchanged": 0},
        )


class _FailingAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, sample, rules=None):
        return Trace(correct=False, final_answer="still wrong")


class _RaisingGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def propose(self, experience, previous):
        raise ValueError("rule must contain if and then")


@patch("rimrule.pipeline.stages.tqdm", lambda rows, **kwargs: _Progress(rows))
@patch("rimrule.pipeline.stages.get_llm", lambda config: object())
@patch("rimrule.pipeline.stages.ToolHopAgent", _FailingAgent)
@patch("rimrule.pipeline.stages.RuleGenerator", _RaisingGenerator)
class InduceBadProposalTest(_StageTest):
    def test_invalid_proposal_is_skipped_not_crashing(self):
        config = self._config(max_atomic_fixes_per_failure=2)
        (self.tmp_path / "failures.jsonl").write_text(
            _failure(0).model_dump_json() + "\n", encoding="utf-8"
        )
        # Must not raise despite the generator raising ValueError.
        rules = stages.induce(config)
        self.assertEqual(rules, [])
        state = stages.load_json(self.tmp_path / "induce_state.json")
        self.assertEqual(state["status"], "complete")


def _candidate(i: int) -> CandidateRule:
    return CandidateRule(
        id=f"r{i}",
        text=f"If condition {i} holds, then take action {i}.",
        error_type=ErrorType.ARGUMENTS,
        source_failure_id=str(i),
    )


def _vocab():
    from rimrule.models import Vocabulary

    return Vocabulary(
        domain=("GENERAL", "UNKNOWN"),
        qualifier=("ALWAYS", "UNKNOWN"),
        action=("CHECK", "UNKNOWN"),
        strength=("MUST", "UNKNOWN"),
        tool_category=("SEARCH", "UNKNOWN"),
    )


def _symbolic_rule(candidate: CandidateRule):
    from rimrule.models import Rule, SymbolicRule

    return Rule(
        id=candidate.id,
        text=candidate.text,
        error_type=candidate.error_type,
        symbolic=SymbolicRule(
            domain=("GENERAL",),
            qualifier=("ALWAYS",),
            action=("CHECK",),
            strength=("MUST",),
            tool_category=("SEARCH",),
        ),
        source_failure_ids=(candidate.source_failure_id,),
    )


class _FakeCanonicalizer:
    def __init__(self, *args, **kwargs):
        pass

    def induce_vocabulary(self, candidates, randomizations):
        return _vocab()

    def translate(self, candidate, vocabulary):
        return _symbolic_rule(candidate)


@patch("rimrule.pipeline.stages.tqdm", lambda rows, **kwargs: _Progress(rows))
@patch("rimrule.pipeline.stages.get_llm", lambda config: object())
@patch("rimrule.pipeline.stages.Canonicalizer", _FakeCanonicalizer)
class CanonicalizeTest(_StageTest):
    def _seed_candidates(self, n: int) -> None:
        lines = "\n".join(_candidate(i).model_dump_json() for i in range(n))
        (self.tmp_path / "candidate_rules.jsonl").write_text(
            lines + "\n", encoding="utf-8"
        )

    def test_induces_vocab_and_translates_all(self):
        self._seed_candidates(3)
        vocab, rules = stages.canonicalize(self._config())
        self.assertIn("GENERAL", vocab.domain)
        self.assertEqual(len(rules), 3)
        self.assertTrue((self.tmp_path / "vocabulary.json").exists())
        out = self.tmp_path / "symbolic_rules.jsonl"
        self.assertEqual(len([x for x in out.read_text().splitlines() if x]), 3)

    def test_resumes_with_frozen_vocabulary_and_partial_rules(self):
        self._seed_candidates(3)
        # Pre-write the frozen vocab and one already-translated rule.
        from rimrule.io import dump_json

        dump_json(self.tmp_path / "vocabulary.json", _vocab())
        (self.tmp_path / "symbolic_rules.jsonl").write_text(
            _symbolic_rule(_candidate(0)).model_dump_json() + "\n", encoding="utf-8"
        )
        _, rules = stages.canonicalize(self._config())
        self.assertEqual(len(rules), 3)


class ConsolidateResumeTest(_StageTest):
    @patch("rimrule.pipeline.stages.tqdm", lambda rows, **kwargs: _Progress(rows))
    def test_returns_finished_artifact_without_recomputing(self):
        from rimrule.io import dump_json, write_jsonl

        final = [_symbolic_rule(_candidate(0)), _symbolic_rule(_candidate(1))]
        write_jsonl(self.tmp_path / "consolidated_rules.jsonl", final)
        dump_json(
            self.tmp_path / "consolidate_state.json",
            {"status": "complete", "accepted_edits": 0, "rules": 2},
        )
        # load_toolhop would raise on the unused.json path if we didn't short-circuit.
        result = stages.consolidate(self._config())
        self.assertEqual([r.id for r in result], ["r0", "r1"])


def _query_symbol():
    from rimrule.models import SymbolicRule

    return SymbolicRule(
        domain=("GENERAL",),
        qualifier=("ALWAYS",),
        action=("CHECK",),
        strength=("MUST",),
        tool_category=("SEARCH",),
    )


class _PruningConsolidator:
    """Fake optimizer that accepts one prune edit then finishes."""

    def __init__(self, samples, alpha, evaluate, affected, on_accept, on_event):
        self.on_accept = on_accept
        self.on_event = on_event

    def run(self, initial, round_offset=0):
        entry = {"round": round_offset + 1, "operation": "prune", "rule_id": "r0"}
        self.on_event({"event": "candidate_evaluated", "round": entry["round"]})
        self.on_accept(entry)
        self.on_event({"event": "edit_accepted", **entry})
        return initial[1:], [entry]


class _RaisingConsolidator:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, initial, round_offset=0):
        raise RuntimeError("optimizer boom")


class _ConsolidateRunTest(_StageTest):
    """Seed the artifacts consolidate() needs and stub the heavy collaborators."""

    def _seed(self, n: int = 3) -> None:
        from rimrule.io import append_jsonl, dump_json, write_jsonl

        rules = [_symbolic_rule(_candidate(i)) for i in range(n)]
        write_jsonl(self.tmp_path / "symbolic_rules.jsonl", rules)
        dump_json(self.tmp_path / "vocabulary.json", _vocab())
        # Pre-seed query symbols so symbolization is skipped entirely.
        for _ in range(n):
            append_jsonl(self.tmp_path / "query_symbols.jsonl", _query_symbol())


@patch("rimrule.pipeline.stages.ToolHopAgent", _FixingAgent)
@patch("rimrule.pipeline.stages.tqdm", lambda *a, **k: _Progress(a[0] if a else []))
@patch("rimrule.pipeline.stages.get_llm", lambda config: object())
@patch(
    "rimrule.pipeline.stages.load_toolhop",
    lambda *a, **k: [_sample(i) for i in range(3)],
)
@patch("rimrule.consolidation.optimizer.Consolidator", _PruningConsolidator)
class ConsolidateRunTest(_ConsolidateRunTest):
    def test_full_run_writes_final_artifact_and_complete_state(self):
        self._seed(3)
        result = stages.consolidate(self._config())
        # One prune edit removes r0, leaving r1, r2.
        self.assertEqual([r.id for r in result], ["r1", "r2"])
        self.assertTrue((self.tmp_path / "consolidated_rules.jsonl").exists())
        state = stages.load_json(self.tmp_path / "consolidate_state.json")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["accepted_edits"], 1)
        # Accepted edit was flushed to the authoritative JSONL log.
        log = self.tmp_path / "consolidation_log.jsonl"
        log_lines = [x for x in log.read_text().splitlines() if x]
        self.assertEqual(len(log_lines), 1)

    def test_jsonl_log_takes_precedence_over_json_on_resume(self):
        self._seed(3)
        from rimrule.io import append_json_line, dump_json

        # JSONL says r0 was already pruned; JSON (stale) says r1. JSONL must win.
        append_json_line(
            self.tmp_path / "consolidation_log.jsonl",
            {"round": 1, "operation": "prune", "rule_id": "r0"},
        )
        dump_json(
            self.tmp_path / "consolidation_log.json",
            [{"round": 1, "operation": "prune", "rule_id": "r1"}],
        )
        result = stages.consolidate(self._config())
        # Resume replays the JSONL prune of r0; r0 must be gone from the result.
        self.assertNotIn("r0", [r.id for r in result])


@patch("rimrule.pipeline.stages.ToolHopAgent", _FixingAgent)
@patch("rimrule.pipeline.stages.tqdm", lambda *a, **k: _Progress(a[0] if a else []))
@patch("rimrule.pipeline.stages.get_llm", lambda config: object())
@patch(
    "rimrule.pipeline.stages.load_toolhop",
    lambda *a, **k: [_sample(i) for i in range(3)],
)
@patch("rimrule.consolidation.optimizer.Consolidator", _RaisingConsolidator)
class ConsolidateErrorPathTest(_ConsolidateRunTest):
    def test_persists_partial_state_and_reraises_on_failure(self):
        self._seed(3)
        with self.assertRaisesRegex(RuntimeError, "optimizer boom"):
            stages.consolidate(self._config())
        state = stages.load_json(self.tmp_path / "consolidate_state.json")
        self.assertEqual(state["status"], "partial")
        self.assertEqual(state["error_type"], "RuntimeError")
        self.assertIn("resume_hint", state)


class ReplayEditsTest(unittest.TestCase):
    def test_prune_removes_the_named_rule(self):
        initial = [_symbolic_rule(_candidate(i)) for i in range(3)]
        replayed = stages._replay_edits(
            initial, [{"operation": "prune", "rule_id": "r1"}]
        )
        self.assertEqual([r.id for r in replayed], ["r0", "r2"])

    def test_generalize_adds_category_suffix(self):
        initial = [_symbolic_rule(_candidate(0))]
        replayed = stages._replay_edits(
            initial, [{"operation": "generalize", "rule_id": "r0"}]
        )
        self.assertEqual(replayed[0].symbolic.tool_category, ("SEARCH_CATEGORY",))

    def test_generalize_is_idempotent_on_already_suffixed(self):
        initial = [_symbolic_rule(_candidate(0))]
        once = stages._replay_edits(
            initial, [{"operation": "generalize", "rule_id": "r0"}]
        )
        twice = stages._replay_edits(
            once, [{"operation": "generalize", "rule_id": "r0"}]
        )
        self.assertEqual(twice[0].symbolic.tool_category, ("SEARCH_CATEGORY",))

    def test_edit_targeting_missing_rule_is_skipped(self):
        initial = [_symbolic_rule(_candidate(0))]
        replayed = stages._replay_edits(
            initial, [{"operation": "prune", "rule_id": "does-not-exist"}]
        )
        self.assertEqual([r.id for r in replayed], ["r0"])
