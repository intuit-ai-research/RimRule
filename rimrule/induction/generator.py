from __future__ import annotations

import hashlib
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from rimrule.induction.checks import linguistic_check
from rimrule.induction.prompts import RULE_PROMPT
from rimrule.models import CandidateRule, FailureExperience, RuleProposal


class RuleGenerator:
    """Propose one atomic corrective rule from a failed tool-use experience."""

    def __init__(self, llm: BaseChatModel, max_words: int = 80):
        self.llm = llm.with_structured_output(RuleProposal, method="function_calling")
        self.max_words = max_words

    def propose(
        self, experience: FailureExperience, previous: list[CandidateRule] | None = None
    ) -> CandidateRule:
        """Return a validated candidate rule for ``experience``.

        Raises:
            ValueError: If the proposed rule fails the linguistic shape check.
        """
        prompt = RULE_PROMPT.format(
            query=experience.sample.question,
            tools=json.dumps(
                [t.model_dump() for t in experience.sample.tools],
                ensure_ascii=False,
                indent=2,
            ),
            failed=experience.failed_trace.model_dump_json(indent=2),
            reference=experience.reference_trace.model_dump_json(indent=2),
            previous="\n".join(r.text for r in (previous or [])) or "None",
        )
        proposal = self.llm.invoke([HumanMessage(content=prompt)])
        if not isinstance(proposal, RuleProposal):
            raise TypeError(f"Expected RuleProposal, got {type(proposal).__name__}")
        ok, reason = linguistic_check(proposal.rule, self.max_words)
        if not ok:
            raise ValueError(reason)
        digest = hashlib.sha256(
            (experience.sample.id + proposal.rule).encode()
        ).hexdigest()[:16]
        return CandidateRule(
            id=f"rule_{digest}",
            text=proposal.rule,
            error_type=proposal.error_type,
            source_failure_id=experience.sample.id,
            generation_round=len(previous or []),
        )
