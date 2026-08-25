from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rimrule.toolhop.loader import (
    ToolHopSchemaError,
    _parse_args,
    load_toolhop,
    schema_summary,
)


class ParseArgsTest(unittest.TestCase):
    def test_dict(self):
        self.assertEqual(_parse_args({"x": 1}), {"x": 1})

    def test_json_string(self):
        self.assertEqual(_parse_args('{"x": 1}'), {"x": 1})

    def test_none(self):
        self.assertIsNone(_parse_args(None))

    def test_non_dict_json(self):
        self.assertIsNone(_parse_args("[1, 2]"))

    def test_invalid_json(self):
        self.assertIsNone(_parse_args("not json"))

    def test_non_string_non_dict(self):
        self.assertIsNone(_parse_args(42))


class LoadToolhopTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, data: object) -> Path:
        path = self.tmp_path / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            load_toolhop(self.tmp_path / "missing.json")

    def test_non_list(self):
        path = self.tmp_path / "data.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with self.assertRaisesRegex(ToolHopSchemaError, "JSON list"):
            load_toolhop(path)

    def test_valid_sample(self):
        path = self._write(
            [
                {
                    "id": "1",
                    "question": "What is 2+2?",
                    "answer": "4",
                    "tools": {
                        "calc": {
                            "name": "calc",
                            "description": "Calculator",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    },
                    "sub_task": {"calc": "4"},
                    "arguments": [
                        [{"function": {"name": "calc", "arguments": '{"x": 2}'}}]
                    ],
                }
            ]
        )
        samples = load_toolhop(path, strict=True)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].id, "1")
        self.assertEqual(len(samples[0].tools), 1)
        self.assertEqual(len(samples[0].ground_truth), 1)
        self.assertEqual(samples[0].ground_truth[0].arguments, {"x": 2})

    def test_tools_as_list(self):
        path = self._write(
            [
                {
                    "id": "1",
                    "question": "q",
                    "answer": "a",
                    "tools": [
                        {
                            "name": "fn",
                            "description": "A function",
                            "parameters": {"type": "object"},
                        }
                    ],
                    "sub_task": {},
                }
            ]
        )
        samples = load_toolhop(path, strict=True)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].tools[0].name, "fn")

    def test_strict_rejects_bad_samples(self):
        path = self._write([{"no_question": True}])
        with self.assertRaises(ToolHopSchemaError):
            load_toolhop(path, strict=True)

    def test_non_strict_skips_bad_samples(self):
        path = self._write(
            [
                {"no_question": True},
                {
                    "id": "ok",
                    "question": "q",
                    "answer": "a",
                    "tools": {},
                    "sub_task": {},
                },
            ]
        )
        samples = load_toolhop(path, strict=False)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].id, "ok")

    def test_supports_missing_arguments(self):
        path = self.tmp_path / "ToolHop.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "1",
                        "question": "q",
                        "answer": "a",
                        "tools": {
                            "step": {
                                "name": "tool",
                                "description": "d",
                                "parameters": {},
                            }
                        },
                        "functions": [],
                        "sub_task": {"step": "x"},
                    }
                ]
            ),
            encoding="utf-8",
        )
        samples = load_toolhop(path)
        self.assertIsNone(samples[0].ground_truth[0].arguments)
        self.assertEqual(schema_summary(samples)["missing_argument_annotations"], 1)
