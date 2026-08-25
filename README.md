# RimRule

![Supported Python versions](https://shields.io/badge/python-3.12-green?logo=python)
![License](https://shields.io/badge/license-Apache--2.0-blue)

An implementation of [**RimRule**](https://arxiv.org/abs/2601.00086) -- MDL-guided rule learning for tool-using agents, with ToolHop as the included example workflow.

The pipeline is split into resumable stages:

1. **collect** -- run the base agent on ToolHop and record zero-shot failures
2. **induce** -- propose and locally evaluate candidate rules from failure traces
3. **canonicalize** -- translate rules into RimRule's symbolic representation
4. **consolidate** -- greedy MDL optimization of the rule library

Each stage writes partial results as it runs. Re-running the same command with the same `--fld` resumes from existing artifacts instead of starting over.

## Requirements

- Python 3.12
- A local copy of ToolHop at `data/toolhop/ToolHop.json` (not committed to the repo)
- An OpenAI API key

All LLM calls go through LangChain (`langchain-openai` `ChatOpenAI`). Any
OpenAI-compatible endpoint works: set `base_url` in your config to point at a
gateway or self-hosted server, and `api_key_env` to name the environment variable
holding its key.

## Setup

```bash
uv sync --all-extras
export OPENAI_API_KEY='sk-...'
```

### Verify

```bash
rimrule --help
```

## ToolHop data

The dataset is **not committed** to the repo (`data/toolhop/` is gitignored). ToolHop
is published on [Hugging Face](https://huggingface.co/datasets/bytedance-research/ToolHop).
Download it to:

```text
data/toolhop/ToolHop.json
```

(Set the `dataset.path` in your config to the file you fetched, e.g.
`data/toolhop/ToolHop30.json` for a 30-sample subset.)

Validate before making any LLM calls:

```bash
rimrule validate-data \
  --config configs/toolhop.yaml \
  --fld artifacts/demo_run
```

## Running the pipeline

Every command accepts `--fld <artifact-directory>`. When omitted, the config's `artifacts_dir` is used (defaults to `artifacts`). Use the **same** `--fld` value for every stage in one run.

Configs share defaults through `configs/base.yaml`; `configs/toolhop.yaml` sets
`base: base.yaml` and overrides only its LLM roles. Copy it to a new file and
adjust `dataset.path` for a different data subset.

```bash
# Step by step
rimrule run-collect \
  --config configs/toolhop.yaml \
  --fld artifacts/demo_run

rimrule run-induce \
  --config configs/toolhop.yaml \
  --fld artifacts/demo_run

rimrule run-canonicalize \
  --config configs/toolhop.yaml \
  --fld artifacts/demo_run

rimrule run-consolidate \
  --config configs/toolhop.yaml \
  --fld artifacts/demo_run

# Or run the full pipeline in one command
rimrule run-all \
  --config configs/toolhop.yaml \
  --fld artifacts/demo_run
```

For a first experiment, running stages separately is recommended to make intermediate artifacts easier to inspect.

## Key artifacts

| Stage | Outputs |
|-------|---------|
| collect | `failures.jsonl`, `collect_state.json` |
| induce | `candidate_rules.jsonl`, `induce_state.json` |
| canonicalize | `vocabulary.json`, `symbolic_rules.jsonl`, `canonicalize_state.json` |
| consolidate | `consolidated_rules.jsonl`, `evaluation_cache.sqlite`, `consolidation_checkpoint.json`, `consolidation_log.jsonl` |

## Local development

```bash
# Install all dependencies
uv sync --all-extras

# Run linting (ruff + mypy)
tox -e lint

# Run tests with coverage
tox -e py312

# Run everything
tox
```

## Contributing

See [Contribution Guidelines](./CONTRIBUTING.md) and our
[Code of Conduct](./CODE_OF_CONDUCT.md).

## License

Apache-2.0. See [LICENSE](./LICENSE).
