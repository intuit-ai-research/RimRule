from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pydantic
from pydantic import BaseModel, Field

from rimrule.io import (
    append_jsonl,
    dump_json,
    ensure_jsonl,
    load_json,
    read_jsonl,
    reset_jsonl,
    write_jsonl,
)


class _Item(BaseModel):
    name: str = Field(description="item name")
    value: int = Field(description="item value")


class IoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_write_and_read_jsonl(self):
        path = self.tmp_path / "items.jsonl"
        write_jsonl(path, [_Item(name="a", value=1), _Item(name="b", value=2)])
        loaded = read_jsonl(path, _Item)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].name, "a")
        self.assertEqual(loaded[1].value, 2)

    def test_append_jsonl(self):
        path = self.tmp_path / "items.jsonl"
        append_jsonl(path, _Item(name="first", value=1))
        append_jsonl(path, _Item(name="second", value=2))
        self.assertEqual(len(read_jsonl(path, _Item)), 2)

    def test_ensure_jsonl_creates_file(self):
        path = self.tmp_path / "sub" / "new.jsonl"
        result = ensure_jsonl(path)
        self.assertTrue(result.exists())
        self.assertEqual(result.read_text(), "")

    def test_ensure_jsonl_does_not_truncate(self):
        path = self.tmp_path / "existing.jsonl"
        path.write_text("existing\n")
        ensure_jsonl(path)
        self.assertEqual(path.read_text(), "existing\n")

    def test_reset_jsonl(self):
        path = self.tmp_path / "items.jsonl"
        path.write_text("old data\n")
        reset_jsonl(path)
        self.assertEqual(path.read_text(), "")

    def test_read_jsonl_missing_file(self):
        self.assertEqual(read_jsonl(self.tmp_path / "nope.jsonl", _Item), [])

    def test_read_jsonl_tolerates_torn_last_line(self):
        path = self.tmp_path / "items.jsonl"
        path.write_text(
            _Item(name="ok", value=1).model_dump_json() + "\n{broken", encoding="utf-8"
        )
        loaded = read_jsonl(path, _Item)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].name, "ok")

    def test_read_jsonl_skips_blank_lines(self):
        path = self.tmp_path / "items.jsonl"
        good = _Item(name="ok", value=1).model_dump_json()
        path.write_text(f"\n{good}\n\n", encoding="utf-8")
        self.assertEqual(len(read_jsonl(path, _Item)), 1)

    def test_read_jsonl_raises_on_mid_file_corruption(self):
        path = self.tmp_path / "items.jsonl"
        good = _Item(name="ok", value=1).model_dump_json()
        path.write_text(f"{good}\n{{broken\n{good}\n", encoding="utf-8")
        with self.assertRaises(pydantic.ValidationError):
            read_jsonl(path, _Item)

    def test_dump_and_load_json(self):
        path = self.tmp_path / "data.json"
        dump_json(path, {"key": "value"})
        self.assertEqual(load_json(path), {"key": "value"})

    def test_dump_json_with_model(self):
        path = self.tmp_path / "item.json"
        dump_json(path, _Item(name="test", value=42))
        loaded = load_json(path)
        self.assertEqual(loaded["name"], "test")
        self.assertEqual(loaded["value"], 42)

    def test_load_json_missing_file(self):
        self.assertIsNone(load_json(self.tmp_path / "nope.json"))
        self.assertEqual(load_json(self.tmp_path / "nope.json", "default"), "default")

    def test_dump_json_atomic(self):
        path = self.tmp_path / "data.json"
        dump_json(path, {"v": 1})
        dump_json(path, {"v": 2})
        self.assertEqual(load_json(path), {"v": 2})
        self.assertEqual(list(self.tmp_path.glob("*.tmp")), [])
