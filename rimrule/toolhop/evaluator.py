from __future__ import annotations

import re


def extract_answer(text: str | None) -> str:
    """Return the content of an ``<answer>`` tag, or the whole text if absent."""
    text = text or ""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip()


def normalize_answer(value: str) -> str:
    """Canonicalize an answer for comparison (case, commas, trailing .0, spaces)."""
    value = value.strip().lower().replace(",", "")
    if value.endswith(".0"):
        value = value[:-2]
    return re.sub(r"\s+", " ", value).strip(" .")


def is_correct(prediction: str | None, expected: str) -> bool:
    """Return whether the extracted, normalized prediction matches ``expected``."""
    return normalize_answer(extract_answer(prediction)) == normalize_answer(expected)
