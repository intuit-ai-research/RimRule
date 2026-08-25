from __future__ import annotations

import re


def linguistic_check(rule: str, max_words: int = 80) -> tuple[bool, str | None]:
    """Check a rule's surface form: length bounds and an explicit if-then shape.

    Returns:
        ``(True, None)`` if the rule passes, else ``(False, reason)``.
    """
    words = rule.split()
    if len(words) > max_words:
        return False, f"rule has {len(words)} words; maximum is {max_words}"
    if not re.search(r"\bif\b", rule, re.I) or not re.search(r"\bthen\b", rule, re.I):
        return False, "rule must contain if and then"
    if len(words) < 5:
        return False, "rule is too short"
    return True, None
