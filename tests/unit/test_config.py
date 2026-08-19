from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from rimrule.config import LLMConfig, RimRuleConfig, _deep_merge, _load_yaml_with_base


class DeepMergeTest(unittest.TestCase):
    def test_override_wins_and_nested_dicts_merge(self):
        base = {"a": 1, "nested": {"x": 1, "y": 2}}
        override = {"a": 9, "nested": {"y": 20, "z": 30}}
        merged = _deep_merge(base, override)
        self.assertEqual(merged["a"], 9)
        self.assertEqual(merged["nested"], {"x": 1, "y": 20, "z": 30})

    def test_does_not_mutate_inputs(self):
        base = {"nested": {"x": 1}}
        _deep_merge(base, {"nested": {"y": 2}})
        self.assertEqual(base, {"nested": {"x": 1}})

    def test_non_dict_over_dict_replaces(self):
        merged = _deep_merge({"k": {"x": 1}}, {"k": "scalar"})
        self.assertEqual(merged["k"], "scalar")


class LoadYamlWithBaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name: str, body: str) -> Path:
        path = self.tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_child_overrides_base(self):
        self._write("base.yaml", "alpha: 0.1\nretrieval_top_k: 5\n")
        child = self._write("child.yaml", "base: base.yaml\nalpha: 0.9\n")
        data = _load_yaml_with_base(child)
        self.assertEqual(data["alpha"], 0.9)
        self.assertEqual(data["retrieval_top_k"], 5)
        self.assertNotIn("base", data)

    def test_multi_level_base_chain(self):
        self._write("root.yaml", "alpha: 0.1\na: 1\n")
        self._write("mid.yaml", "base: root.yaml\na: 2\nb: 2\n")
        leaf = self._write("leaf.yaml", "base: mid.yaml\nb: 3\n")
        data = _load_yaml_with_base(leaf)
        self.assertEqual(data["alpha"], 0.1)
        self.assertEqual(data["a"], 2)
        self.assertEqual(data["b"], 3)

    def test_self_referential_base_raises(self):
        cycle = self._write("cycle.yaml", "base: cycle.yaml\nalpha: 0.1\n")
        with self.assertRaisesRegex(ValueError, "Cyclic config base"):
            _load_yaml_with_base(cycle)

    def test_indirect_cycle_raises(self):
        self._write("a.yaml", "base: b.yaml\n")
        b = self._write("b.yaml", "base: a.yaml\n")
        with self.assertRaisesRegex(ValueError, "Cyclic config base"):
            _load_yaml_with_base(b)


class RimRuleConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _config_file(self) -> Path:
        path = self.tmp_path / "cfg.yaml"
        path.write_text(
            "agent_llm:\n  provider: openai\n  model: gpt-4o\n"
            "rule_llm:\n  provider: openai\n  model: gpt-4o-mini\n",
            encoding="utf-8",
        )
        return path

    def test_load_reads_and_validates(self):
        config = RimRuleConfig.load(self._config_file())
        self.assertEqual(config.agent_llm.model, "gpt-4o")
        self.assertIsNotNone(config.rule_llm)
        rule_llm = cast(LLMConfig, config.rule_llm)
        self.assertEqual(rule_llm.provider, "openai")

    def test_with_artifacts_dir_override(self):
        config = RimRuleConfig.load(self._config_file())
        updated = config.with_artifacts_dir(self.tmp_path / "run7")
        self.assertEqual(updated.artifacts_dir, str(self.tmp_path / "run7"))
        self.assertEqual(updated.artifact_path("x.json").parent, self.tmp_path / "run7")
