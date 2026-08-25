from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC

from tqdm.auto import tqdm

from rimrule.canonicalization import Canonicalizer
from rimrule.config import RimRuleConfig
from rimrule.induction import RuleGenerator
from rimrule.io import (
    append_json_line,
    append_jsonl,
    dump_json,
    ensure_jsonl,
    load_json,
    read_jsonl,
    write_jsonl,
)
from rimrule.llm import get_llm
from rimrule.models import (
    CandidateRule,
    FailureExperience,
    Rule,
    SymbolicRule,
    Vocabulary,
)
from rimrule.toolhop.agent import ToolHopAgent, build_reference_trace
from rimrule.toolhop.loader import load_toolhop, schema_summary

logger = logging.getLogger(__name__)


def _progress_write(message: str) -> None:
    writer = getattr(tqdm, "write", None)
    if writer is not None:
        writer(message)


def collect(config: RimRuleConfig) -> list[FailureExperience]:
    """Run the base agent over ToolHop and record its failures.

    Resumable: already-completed samples are read from the collect state file and
    skipped. Returns every failure experience gathered so far.
    """
    samples = load_toolhop(config.dataset.path, config.dataset.strict_schema)
    dump_json(config.artifact_path("toolhop_schema.json"), schema_summary(samples))

    failures_path = ensure_jsonl(config.artifact_path("failures.jsonl"))
    state_path = config.artifact_path("collect_state.json")
    failures = read_jsonl(failures_path, FailureExperience)
    state = load_json(state_path, {}) or {}
    completed_ids = set(state.get("completed_sample_ids", []))
    failure_ids = {failure.sample.id for failure in failures}
    # Recover older partial runs that wrote failures before state existed.
    completed_ids.update(failure_ids)
    successes = int(state.get("successes", 0))

    remaining = [sample for sample in samples if sample.id not in completed_ids]
    if not remaining:
        _progress_write(
            f"Resuming collection: all {len(samples)} samples already finished"
        )
        return failures

    _progress_write(
        f"Resuming collection: {len(completed_ids)}/{len(samples)} samples finished; "
        f"{len(remaining)} remaining"
    )
    agent = ToolHopAgent(
        get_llm(config.agent_llm),
        config.agent.max_turns,
        config.agent.tool_timeout_seconds,
    )
    progress = tqdm(
        remaining, desc="Collecting ToolHop traces", unit="sample", dynamic_ncols=True
    )
    for sample in progress:
        trace = agent.run(sample)
        if trace.correct:
            successes += 1
        else:
            failure = FailureExperience(
                sample=sample,
                failed_trace=trace,
                reference_trace=build_reference_trace(sample),
            )
            failures.append(failure)
            append_jsonl(failures_path, failure)
        completed_ids.add(sample.id)
        dump_json(
            state_path,
            {
                "status": "complete"
                if len(completed_ids) == len(samples)
                else "partial",
                "completed_sample_ids": sorted(completed_ids),
                "successes": successes,
                "failures": len(failures),
                "total": len(samples),
            },
        )
        progress.set_postfix(success=successes, failure=len(failures), refresh=True)
    return failures


def induce(config: RimRuleConfig) -> list[CandidateRule]:
    """Propose candidate rules from collected failures via iterative correction.

    For each failure, rules are proposed and re-tested up to the configured
    atomic-fix budget, keeping only those that change the rollout. Resumable per
    failure. Returns all accepted candidate rules.
    """
    failures = read_jsonl(config.artifact_path("failures.jsonl"), FailureExperience)
    rules_path = ensure_jsonl(config.artifact_path("candidate_rules.jsonl"))
    state_path = config.artifact_path("induce_state.json")
    accepted = read_jsonl(rules_path, CandidateRule)
    state = load_json(state_path, {}) or {}
    completed_ids = set(state.get("completed_failure_ids", []))
    grouped: dict[str, list[CandidateRule]] = defaultdict(list)
    for rule in accepted:
        grouped[rule.source_failure_id].append(rule)
    for rules in grouped.values():
        rules.sort(key=lambda rule: rule.generation_round)

    remaining = [
        failure for failure in failures if failure.sample.id not in completed_ids
    ]
    if not remaining:
        _progress_write(
            f"Resuming induction: all {len(failures)} failures already finished"
        )
        return accepted

    _progress_write(
        f"Resuming induction: {len(completed_ids)}/{len(failures)} failures finished; "
        f"{len(accepted)} rules already saved"
    )
    generator = RuleGenerator(
        get_llm(config.rule_llm or config.agent_llm), config.max_rule_words
    )
    agent = ToolHopAgent(
        get_llm(config.agent_llm),
        config.agent.max_turns,
        config.agent.tool_timeout_seconds,
    )
    proposed = int(state.get("proposed", len(accepted)))
    corrected = int(state.get("corrected", 0))
    unchanged = int(state.get("unchanged", 0))

    progress = tqdm(
        remaining,
        desc="Inducing RimRule candidates",
        unit="failure",
        dynamic_ncols=True,
    )
    for exp in progress:
        local = list(grouped.get(exp.sample.id, []))
        current = exp.failed_trace
        if local:
            # Reconstruct the last accepted local state after an interrupted failure.
            current = agent.run(exp.sample, [rule.text for rule in local])
        for generation_round in range(len(local), config.max_atomic_fixes_per_failure):
            if current.correct:
                corrected += 1
                break
            try:
                candidate = generator.propose(
                    exp.model_copy(update={"failed_trace": current}), local
                )
            except ValueError as exc:
                # A malformed proposal (fails the linguistic check) is skipped so
                # one bad rule does not abort the whole stage.
                logger.warning(
                    "Skipping invalid rule proposal for sample %s: %s",
                    exp.sample.id,
                    exc,
                )
                break
            candidate = candidate.model_copy(
                update={"generation_round": generation_round}
            )
            proposed += 1
            checked = agent.run(exp.sample, [rule.text for rule in local + [candidate]])
            if not checked.correct and checked.model_dump() == current.model_dump():
                unchanged += 1
                break
            local.append(candidate)
            accepted.append(candidate)
            append_jsonl(rules_path, candidate)
            current = checked
            if checked.correct:
                corrected += 1
                break

        completed_ids.add(exp.sample.id)
        dump_json(
            state_path,
            {
                "status": "complete"
                if len(completed_ids) == len(failures)
                else "partial",
                "completed_failure_ids": sorted(completed_ids),
                "proposed": proposed,
                "accepted": len(accepted),
                "corrected": corrected,
                "unchanged": unchanged,
                "total": len(failures),
            },
        )
        progress.set_postfix(
            proposed=proposed,
            accepted=len(accepted),
            corrected=corrected,
            unchanged=unchanged,
            refresh=True,
        )
    return accepted


def canonicalize(config: RimRuleConfig) -> tuple[Vocabulary, list[Rule]]:
    """Induce a symbolic vocabulary and translate every candidate rule into it.

    The vocabulary is frozen once and reused across resumed runs. Returns the
    vocabulary and the list of symbolic rules.
    """
    candidates = read_jsonl(
        config.artifact_path("candidate_rules.jsonl"), CandidateRule
    )
    vocab_path = config.artifact_path("vocabulary.json")
    rules_path = ensure_jsonl(config.artifact_path("symbolic_rules.jsonl"))
    state_path = config.artifact_path("canonicalize_state.json")
    can = Canonicalizer(
        get_llm(config.vocabulary_llm or config.rule_llm or config.agent_llm)
    )

    existing_vocab = load_json(vocab_path)
    if existing_vocab is not None:
        vocab = Vocabulary.model_validate(existing_vocab)
        _progress_write("Resuming canonicalization with existing frozen vocabulary")
    else:
        vocab = can.induce_vocabulary(candidates, config.vocabulary_randomizations)
        dump_json(vocab_path, vocab)

    rules = read_jsonl(rules_path, Rule)
    completed_ids = {rule.id for rule in rules}
    remaining = [
        candidate for candidate in candidates if candidate.id not in completed_ids
    ]
    if not remaining:
        _progress_write(
            f"Resuming canonicalization: all {len(candidates)} rules already finished"
        )
        return vocab, rules

    unknown_rules = sum(rule.symbolic.has_unknown() for rule in rules)
    progress = tqdm(
        remaining,
        desc="Canonicalizing RimRule candidates",
        unit="rule",
        dynamic_ncols=True,
    )
    for candidate in progress:
        rule = can.translate(candidate, vocab)
        rules.append(rule)
        append_jsonl(rules_path, rule)
        if rule.symbolic.has_unknown():
            unknown_rules += 1
        dump_json(
            state_path,
            {
                "status": "complete" if len(rules) == len(candidates) else "partial",
                "canonicalized": len(rules),
                "unknown": unknown_rules,
                "total": len(candidates),
            },
        )
        progress.set_postfix(
            canonicalized=len(rules), unknown=unknown_rules, refresh=True
        )
    return vocab, rules


def _replay_edits(initial: list[Rule], entries: list[dict]) -> list[Rule]:
    rules = list(initial)
    for entry in entries:
        idx = next(
            (i for i, rule in enumerate(rules) if rule.id == entry["rule_id"]), None
        )
        if idx is None:
            continue
        if entry["operation"] == "prune":
            rules = rules[:idx] + rules[idx + 1 :]
        elif entry["operation"] == "generalize":
            rule = rules[idx]
            cats = tuple(
                c if c.endswith("_CATEGORY") else f"{c}_CATEGORY"
                for c in rule.symbolic.tool_category
            )
            updated = rule.model_copy(
                update={
                    "symbolic": rule.symbolic.model_copy(update={"tool_category": cats})
                }
            )
            rules = rules[:idx] + [updated] + rules[idx + 1 :]
    return rules


def consolidate(config: RimRuleConfig) -> list[Rule]:
    """Optimize the symbolic rule library by greedy MDL minimization.

    Symbolizes queries, then repeatedly prunes or generalizes rules, reusing a
    SQLite evaluation cache so only samples with a changed retrieval set are
    re-run. Fully resumable from its log and state files. Returns the final rules.
    """
    from rimrule.consolidation.affected import affected_samples
    from rimrule.consolidation.cache import EvaluationCache
    from rimrule.consolidation.optimizer import Consolidator
    from rimrule.retrieval import QuerySymbolizer, SymbolicRetriever

    final_path = config.artifact_path("consolidated_rules.jsonl")
    final_state = load_json(config.artifact_path("consolidate_state.json"), {}) or {}
    if final_state.get("status") == "complete" and final_path.exists():
        final_rules = read_jsonl(final_path, Rule)
        _progress_write(
            f"Resuming consolidation: finished artifact has {len(final_rules)} rules"
        )
        return final_rules

    samples = load_toolhop(config.dataset.path, config.dataset.strict_schema)
    initial_rules = read_jsonl(config.artifact_path("symbolic_rules.jsonl"), Rule)
    vocab = Vocabulary.model_validate(
        load_json(config.artifact_path("vocabulary.json"))
    )
    _progress_write(
        f"Loaded {len(initial_rules)} symbolic rules and {len(samples)} ToolHop samples"
    )

    symbol_path = ensure_jsonl(config.artifact_path("query_symbols.jsonl"))
    query_symbols = read_jsonl(symbol_path, SymbolicRule)
    if len(query_symbols) > len(samples):
        raise ValueError("query_symbols.jsonl has more rows than the current dataset")
    if len(query_symbols) < len(samples):
        symbolizer = QuerySymbolizer(
            get_llm(config.vocabulary_llm or config.rule_llm or config.agent_llm),
            vocab,
        )
        remaining_samples = samples[len(query_symbols) :]
        _progress_write(
            f"Resuming symbolization: "
            f"{len(query_symbols)}/{len(samples)} samples finished"
        )
        for sample in tqdm(
            remaining_samples,
            desc="Symbolizing ToolHop samples",
            unit="sample",
            dynamic_ncols=True,
        ):
            symbolic = symbolizer.symbolize(sample)
            query_symbols.append(symbolic)
            append_jsonl(symbol_path, symbolic)

    agent = ToolHopAgent(
        get_llm(config.agent_llm),
        config.agent.max_turns,
        config.agent.tool_timeout_seconds,
    )
    cache = EvaluationCache(config.artifact_path("evaluation_cache.sqlite"))
    signature = (
        f"{config.agent_llm.provider}:{config.agent_llm.model}"
        f":turns={config.agent.max_turns}"
    )

    live_progress_path = config.artifact_path("consolidation_progress.json")

    def evaluate(
        candidate_rules: list[Rule], changed: set[int] | None, label: str
    ) -> list[bool]:
        retriever = SymbolicRetriever(candidate_rules, config.retrieval_top_k)
        outcomes: list[bool] = []
        affected_indices = set(range(len(samples))) if changed is None else changed
        cache_hits = executions = 0
        progress = tqdm(
            total=len(affected_indices),
            desc=label,
            unit="sample",
            leave=False,
            dynamic_ncols=True,
        )
        dump_json(
            live_progress_path,
            {
                "status": "evaluating",
                "label": label,
                "affected_total": len(affected_indices),
                "affected_completed": 0,
                "cache_hits": 0,
                "executions": 0,
                "correct_so_far": 0,
            },
        )
        for i, sample in enumerate(samples):
            selected = retriever.retrieve(query_symbols[i])
            trace, was_cached = cache.get_or_run_with_status(
                sample,
                selected,
                signature,
                lambda sample=sample, selected=selected: agent.run(  # type: ignore[misc]
                    sample, [r.text for r in selected]
                ),
            )
            outcomes.append(trace.correct)
            if i in affected_indices:
                cache_hits += int(was_cached)
                executions += int(not was_cached)
                progress.update(1)
                completed = cache_hits + executions
                progress.set_postfix(
                    cached=cache_hits,
                    executed=executions,
                    correct=sum(outcomes),
                    refresh=False,
                )
                dump_json(
                    live_progress_path,
                    {
                        "status": "evaluating",
                        "label": label,
                        "affected_total": len(affected_indices),
                        "affected_completed": completed,
                        "cache_hits": cache_hits,
                        "executions": executions,
                        "correct_so_far": sum(outcomes),
                        "sample_index": i,
                    },
                )
        progress.close()
        dump_json(
            live_progress_path,
            {
                "status": "evaluation_complete",
                "label": label,
                "affected_total": len(affected_indices),
                "affected_completed": len(affected_indices),
                "cache_hits": cache_hits,
                "executions": executions,
                "correct": sum(outcomes),
            },
        )
        return outcomes

    def affected(old: list[Rule], new: list[Rule]) -> set[int]:
        return affected_samples(old, new, query_symbols, config.retrieval_top_k)

    log_path = ensure_jsonl(config.artifact_path("consolidation_log.jsonl"))
    # JSONL is authoritative during a partial run because each accepted edit is
    # flushed there before the final JSON summary is replaced.
    prior_log: list[dict] = []
    if log_path.exists() and log_path.stat().st_size:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prior_log.append(json.loads(line))
    else:
        prior_log = load_json(config.artifact_path("consolidation_log.json"), []) or []
    resumed_rules = _replay_edits(initial_rules, prior_log)
    if prior_log:
        _progress_write(f"Resuming consolidation after {len(prior_log)} accepted edits")

    def persist_accepted_edit(entry: dict) -> None:
        append_json_line(log_path, entry)
        dump_json(
            config.artifact_path("consolidate_state.json"),
            {"status": "partial", "accepted_edits": int(entry["round"])},
        )

    detailed_log_path = ensure_jsonl(
        config.artifact_path("consolidation_detailed_log.jsonl")
    )

    def persist_event(event: dict) -> None:
        from datetime import datetime

        record = {"timestamp": datetime.now(UTC).isoformat(), **event}
        append_json_line(detailed_log_path, record)
        if event.get("event") == "candidate_evaluated":
            dump_json(
                config.artifact_path("consolidate_state.json"),
                {
                    "status": "partial",
                    "accepted_edits": len(prior_log),
                    "round": event.get("round"),
                    "candidate_index": event.get("candidate_index"),
                    "total_candidates": event.get("total_candidates"),
                    "best_delta_mdl": event.get("best_delta_mdl"),
                    "last_event": event.get("event"),
                },
            )

    optimizer = Consolidator(
        samples,
        config.alpha,
        evaluate,
        affected,
        on_accept=persist_accepted_edit,
        on_event=persist_event,
    )
    try:
        final_rules, new_log = optimizer.run(resumed_rules, round_offset=len(prior_log))
    except Exception as exc:
        accepted_edits = len(prior_log)
        if log_path.exists():
            accepted_edits = sum(
                1
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        dump_json(
            config.artifact_path("consolidate_state.json"),
            {
                "status": "partial",
                "accepted_edits": accepted_edits,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resume_hint": (
                    "Re-run the same run-consolidate command; "
                    "accepted edits and successful sample "
                    "evaluations are persisted."
                ),
            },
        )
        raise
    finally:
        cache.close()
    full_log = prior_log + new_log
    write_jsonl(final_path, final_rules)
    dump_json(config.artifact_path("consolidation_log.json"), full_log)
    dump_json(
        config.artifact_path("consolidate_state.json"),
        {
            "status": "complete",
            "accepted_edits": len(full_log),
            "rules": len(final_rules),
        },
    )
    return final_rules
