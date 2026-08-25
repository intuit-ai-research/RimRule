from __future__ import annotations

import logging
import random

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from rimrule.induction.prompts import TRANSLATE_PROMPT, VOCAB_PROMPT
from rimrule.models import (
    SYMBOLIC_FIELDS,
    UNKNOWN,
    CandidateRule,
    Rule,
    SymbolicRule,
    Vocabulary,
)

logger = logging.getLogger(__name__)


def ensure_unknown(vocabulary: Vocabulary) -> Vocabulary:
    """Add a deterministic fallback token to every frozen vocabulary field."""
    data: dict[str, tuple[str, ...]] = {}
    for field in SYMBOLIC_FIELDS:
        values = tuple(
            dict.fromkeys(str(value) for value in getattr(vocabulary, field))
        )
        data[field] = values if UNKNOWN in values else values + (UNKNOWN,)
    return Vocabulary.model_validate(data)


class Canonicalizer:
    """Induce a closed symbolic vocabulary and translate rules into it."""

    def __init__(self, llm: BaseChatModel, translation_attempts: int = 3):
        self.vocab_llm = llm.with_structured_output(
            Vocabulary, method="function_calling"
        )
        self.rule_llm = llm.with_structured_output(
            SymbolicRule, method="function_calling"
        )
        self.translation_attempts = max(1, translation_attempts)

    def induce_vocabulary(
        self,
        rules: list[CandidateRule],
        randomizations: int = 5,
        seed: int = 0,
    ) -> Vocabulary:
        """Induce several candidate vocabularies and keep the most compact one.

        The rule order is shuffled per candidate so the chosen vocabulary does
        not depend on the arbitrary order rules were collected in.
        """
        rng = random.Random(seed)
        candidates: list[Vocabulary] = []
        for _ in range(randomizations):
            ordered = list(rules)
            rng.shuffle(ordered)
            prompt = VOCAB_PROMPT.format(
                rules="\n".join(f"- {rule.text}" for rule in ordered)
            )
            vocab = self.vocab_llm.invoke([HumanMessage(content=prompt)])
            if not isinstance(vocab, Vocabulary):
                raise TypeError(f"Expected Vocabulary, got {type(vocab).__name__}")
            candidates.append(ensure_unknown(vocab))
        return min(
            candidates,
            key=lambda value: (
                sum(len(getattr(value, f)) for f in SYMBOLIC_FIELDS),
                value.model_dump_json(),
            ),
        )

    def _constrain(
        self,
        symbolic: SymbolicRule,
        vocabulary: Vocabulary,
        *,
        fallback_unknown: bool,
    ) -> SymbolicRule:
        """Keep only in-vocabulary symbols; raise or fall back on unknown ones."""
        normalized: dict[str, tuple[str, ...]] = {}
        for field in SYMBOLIC_FIELDS:
            allowed = set(getattr(vocabulary, field))
            values: list[str] = []
            unknown: list[str] = []
            for value in getattr(symbolic, field):
                if value in allowed:
                    values.append(value)
                else:
                    unknown.append(value)
                    if fallback_unknown:
                        values.append(UNKNOWN)
            if unknown and not fallback_unknown:
                raise ValueError(
                    f"Translation used unknown {field} values: {sorted(set(unknown))}"
                )
            normalized[field] = tuple(dict.fromkeys(values))
        return SymbolicRule.model_validate(normalized)

    def _build_rule(self, candidate: CandidateRule, symbolic: SymbolicRule) -> Rule:
        return Rule(
            id=candidate.id,
            text=candidate.text,
            error_type=candidate.error_type,
            symbolic=symbolic,
            source_failure_ids=(candidate.source_failure_id,),
        )

    def translate(self, candidate: CandidateRule, vocabulary: Vocabulary) -> Rule:
        """Translate a rule into the closed vocabulary, retrying on invalid output.

        Each retry feeds the validation error back to the model. After the retry
        budget is exhausted, any still-invalid symbol is replaced by the explicit
        UNKNOWN token so long canonicalization runs never stall.
        """
        vocabulary = ensure_unknown(vocabulary)
        base_prompt = TRANSLATE_PROMPT.format(
            vocabulary=vocabulary.model_dump_json(indent=2),
            rule=candidate.text,
        )
        prompt = base_prompt
        for _attempt in range(self.translation_attempts):
            try:
                symbolic = self.rule_llm.invoke([HumanMessage(content=prompt)])
                if not isinstance(symbolic, SymbolicRule):
                    raise TypeError(
                        f"Expected SymbolicRule, got {type(symbolic).__name__}"
                    )
                constrained = self._constrain(
                    symbolic, vocabulary, fallback_unknown=False
                )
                return self._build_rule(candidate, constrained)
            except (ValueError, KeyError, TypeError) as exc:
                prompt = (
                    f"{base_prompt}\n\n"
                    f"Your previous translation was invalid: {exc}. "
                    "Retry using only exact labels present in the vocabulary. "
                    "Do not invent synonyms."
                )

        try:
            symbolic = self.rule_llm.invoke([HumanMessage(content=prompt)])
            if not isinstance(symbolic, SymbolicRule):
                raise TypeError(f"Expected SymbolicRule, got {type(symbolic).__name__}")
        except Exception:
            logger.exception(
                "Final translation attempt for rule %s failed; falling back to "
                "the closed-vocabulary UNKNOWN token",
                candidate.id,
            )
            symbolic = SymbolicRule()
        constrained = self._constrain(symbolic, vocabulary, fallback_unknown=True)
        return self._build_rule(candidate, constrained)
