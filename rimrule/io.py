from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> None:
    """Overwrite a JSONL file with the given validated records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def append_jsonl(path: str | Path, row: BaseModel) -> None:
    """Append one validated record and make it visible immediately."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(row.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_json_line(path: str | Path, row: dict[str, Any]) -> None:
    """Append one plain-dict record as a JSONL line and fsync it to disk.

    Used for the authoritative consolidation log, so the line must survive a
    crash immediately after an accepted edit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ensure_jsonl(path: str | Path) -> Path:
    """Create a JSONL artifact without truncating an existing partial run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def reset_jsonl(path: str | Path) -> Path:
    """Truncate a JSONL artifact to empty, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def read_jsonl[T: BaseModel](path: str | Path, model: type[T]) -> list[T]:
    """Read completed JSONL records, tolerating a torn final line."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[T] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except Exception:
            if index == len(lines) - 1:
                break
            raise
    return rows


def dump_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump()
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load_json(path: str | Path, default: Any = None) -> Any:
    """Return the parsed JSON at ``path``, or ``default`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
