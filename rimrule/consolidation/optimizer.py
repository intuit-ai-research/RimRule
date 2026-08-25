from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from rimrule.consolidation.mdl import mdl_score
from rimrule.models import Rule, ToolHopSample


class EditResult(BaseModel):
    """The evaluated outcome of applying one candidate edit to the library."""

    model_config = ConfigDict(frozen=True)

    operation: str = Field(description="The edit applied: prune or generalize.")
    rule_id: str = Field(description="Id of the rule the edit targeted.")
    rules: list[Rule] = Field(description="The resulting rule library.")
    outcomes: list[bool] = Field(description="Per-sample correctness after the edit.")
    mdl: float = Field(description="MDL score of the resulting library.")
    delta_mdl: float = Field(description="Change in MDL relative to the current best.")
    affected_samples: int = Field(description="Number of re-evaluated samples.")


class Consolidator:
    """Greedy exact MDL optimizer with visible round/edit progress.

    ``evaluate`` receives a candidate library, affected sample indices, and a
    concise display label. It returns exact outcomes while reusing cached runs
    for prompts whose retrieved rule set did not change.
    """

    def __init__(
        self,
        samples: list[ToolHopSample],
        alpha: float,
        evaluate: Callable[[list[Rule], set[int] | None, str], list[bool]],
        affected: Callable[[list[Rule], list[Rule]], set[int]],
        on_accept: Callable[[dict], None] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.samples = samples
        self.alpha = alpha
        self.evaluate = evaluate
        self.affected = affected
        self.on_accept = on_accept
        self.on_event = on_event

    def _generalized(self, rules: list[Rule], idx: int) -> list[Rule] | None:
        rule = rules[idx]
        cats = rule.symbolic.tool_category
        if not cats or all(c.endswith("_CATEGORY") for c in cats):
            return None
        generalized = tuple(
            c if c.endswith("_CATEGORY") else f"{c}_CATEGORY" for c in cats
        )
        updated = rule.model_copy(
            update={
                "symbolic": rule.symbolic.model_copy(
                    update={"tool_category": generalized}
                )
            }
        )
        return rules[:idx] + [updated] + rules[idx + 1 :]

    def run(
        self, initial: list[Rule], round_offset: int = 0
    ) -> tuple[list[Rule], list[dict]]:
        """Greedily apply prune/generalize edits until none lowers the MDL score.

        Args:
            initial: The starting rule library.
            round_offset: Number of edits already applied in a resumed run.

        Returns:
            The optimized library and the log of accepted edits.
        """
        rules = list(initial)
        tqdm.write(
            f"Initial exact evaluation: {len(rules)} rules across "
            f"{len(self.samples)} samples"
        )
        if self.on_event is not None:
            self.on_event(
                {
                    "event": "initial_evaluation_started",
                    "rules": len(rules),
                    "samples": len(self.samples),
                }
            )
        outcomes = self.evaluate(rules, None, "Initial library")
        current = mdl_score(rules, outcomes, self.alpha)[0]
        correct = sum(outcomes)
        tqdm.write(f"Initial MDL={current:.6f}; correct={correct}/{len(outcomes)}")
        if self.on_event is not None:
            self.on_event(
                {
                    "event": "initial_evaluation_completed",
                    "rules": len(rules),
                    "samples": len(self.samples),
                    "mdl": current,
                    "correct": correct,
                }
            )

        log: list[dict] = []
        round_number = round_offset
        max_rounds = max(1, len(initial))
        rounds = tqdm(
            total=max_rounds,
            desc="Consolidating RimRule library",
            unit="round",
            dynamic_ncols=True,
        )

        while True:
            round_number += 1
            variants: list[tuple[str, Rule, list[Rule]]] = []
            for i, rule in enumerate(rules):
                variants.append(("prune", rule, rules[:i] + rules[i + 1 :]))
                generalized = self._generalized(rules, i)
                if generalized is not None:
                    variants.append(("generalize", rule, generalized))

            best: EditResult | None = None
            if self.on_event is not None:
                self.on_event(
                    {
                        "event": "round_started",
                        "round": round_number,
                        "rules": len(rules),
                        "mdl_before": current,
                        "correct_before": sum(outcomes),
                        "total_candidates": len(variants),
                    }
                )
            edit_progress = tqdm(
                variants,
                desc=f"Round {round_number}: evaluating edits",
                unit="edit",
                leave=False,
                dynamic_ncols=True,
            )
            for edit_index, (operation, rule, candidate) in enumerate(edit_progress, 1):
                changed = self.affected(rules, candidate)
                label = (
                    f"Round {round_number} {operation} {edit_index}/{len(variants)} "
                    f"({len(changed)} affected)"
                )
                candidate_outcomes = self.evaluate(candidate, changed, label)
                score = mdl_score(candidate, candidate_outcomes, self.alpha)[0]
                result = EditResult(
                    operation=operation,
                    rule_id=rule.id,
                    rules=candidate,
                    outcomes=candidate_outcomes,
                    mdl=score,
                    delta_mdl=score - current,
                    affected_samples=len(changed),
                )
                if best is None or (
                    result.delta_mdl,
                    result.operation,
                    result.rule_id,
                ) < (
                    best.delta_mdl,
                    best.operation,
                    best.rule_id,
                ):
                    best = result

                if self.on_event is not None:
                    self.on_event(
                        {
                            "event": "candidate_evaluated",
                            "round": round_number,
                            "candidate_index": edit_index,
                            "total_candidates": len(variants),
                            "operation": result.operation,
                            "rule_id": result.rule_id,
                            "affected_samples": result.affected_samples,
                            "mdl_before": current,
                            "mdl_after": result.mdl,
                            "delta_mdl": result.delta_mdl,
                            "correct_before": sum(outcomes),
                            "correct_after": sum(result.outcomes),
                            "rules_before": len(rules),
                            "rules_after": len(result.rules),
                            "best_operation": best.operation,
                            "best_rule_id": best.rule_id,
                            "best_delta_mdl": best.delta_mdl,
                        }
                    )

                edit_progress.set_postfix(
                    best_delta=(f"{best.delta_mdl:.6f}" if best else "n/a"),
                    affected=len(changed),
                    refresh=True,
                )

            if best is None or best.delta_mdl >= 0:
                reason = (
                    "no candidate edits"
                    if best is None
                    else (f"best delta MDL={best.delta_mdl:.6f}")
                )
                tqdm.write(
                    f"Stopping after {round_number - 1} total accepted edits: {reason}."
                )
                if self.on_event is not None:
                    self.on_event(
                        {
                            "event": "stopped",
                            "round": round_number,
                            "accepted_edits": round_number - 1,
                            "reason": reason,
                            "rules": len(rules),
                            "mdl": current,
                            "correct": sum(outcomes),
                        }
                    )
                break

            entry = {
                "round": round_number,
                "operation": best.operation,
                "rule_id": best.rule_id,
                "mdl_before": current,
                "mdl_after": best.mdl,
                "delta_mdl": best.delta_mdl,
                "affected_samples": best.affected_samples,
                "correct_before": sum(outcomes),
                "correct_after": sum(best.outcomes),
                "rules_before": len(rules),
                "rules_after": len(best.rules),
            }
            log.append(entry)
            if self.on_accept is not None:
                self.on_accept(entry)
            if self.on_event is not None:
                self.on_event({"event": "edit_accepted", **entry})

            tqdm.write(
                f"Accepted round {round_number}: {best.operation} rule "
                f"{best.rule_id}; delta MDL={best.delta_mdl:.6f}; "
                f"rules={len(rules)}->{len(best.rules)}; "
                f"correct={sum(outcomes)}->{sum(best.outcomes)}; "
                f"affected={best.affected_samples}"
            )
            rules, outcomes, current = best.rules, best.outcomes, best.mdl
            rounds.update(1)
            rounds.set_postfix(
                rules=len(rules),
                mdl=f"{current:.6f}",
                correct=sum(outcomes),
                refresh=True,
            )

        rounds.close()
        tqdm.write(
            f"Final library: {len(rules)} rules; MDL={current:.6f}; "
            f"correct={sum(outcomes)}/{len(outcomes)}"
        )
        return rules, log
