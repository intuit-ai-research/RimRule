# CLAUDE.md

## Development Commands
- `uv sync --all-extras` -- install all dependencies
- `tox -e lint` -- run linting (ruff check + ruff format + mypy)
- `tox -e py312` -- run tests with coverage
- `tox` -- run all environments
- `rimrule --help` -- CLI entry point for pipeline stages

## Architecture
- `rimrule/` -- main library code
  - `models.py` -- Pydantic models (ToolHopSample, Rule, SymbolicRule, Trace, etc.)
  - `config.py` -- RimRuleConfig (YAML-driven), LLMConfig, AgentConfig
  - `io.py` -- JSONL/JSON persistence helpers (append, read, atomic write)
  - `pipeline/stages.py` -- orchestrates the four pipeline stages with resumable state
  - `llm/` -- LLM client factory (OpenAI-compatible) with retry logic
  - `induction/` -- rule proposal from failure experiences (prompts, linguistic checks)
  - `canonicalization/` -- vocabulary induction and symbolic translation
  - `consolidation/` -- greedy MDL optimizer with evaluation caching
  - `retrieval/` -- symbolic rule retriever and query symbolizer
  - `toolhop/` -- ToolHop agent, tool executor (sandboxed subprocess), evaluator, dataset loader
- `configs/` -- YAML pipeline configs
- `data/toolhop/` -- ToolHop dataset files
- `scripts/cli.py` -- Typer CLI: validate-data, run-collect, run-induce, run-canonicalize, run-consolidate, run-all
- `tests/unit/` -- unit tests (network blocked)
- `tests/integration/` -- integration tests

## Pipeline Stages
1. `collect` -- run base agent on ToolHop, record failures
2. `induce` -- propose candidate rules from failure traces
3. `canonicalize` -- translate rules into symbolic vocabulary
4. `consolidate` -- greedy MDL optimization of the rule library

Each stage is resumable via state files in the artifacts directory.

## Code Conventions
Summarized here; `CONTRIBUTING.md` is the authoritative version.

- Python 3.12; use `list[str]`, `float | None` (not `typing.List`, `typing.Optional`)
- Type hints on all public functions including return types
- Pydantic v2 `BaseModel` for structured data with `Field(description=...)`
- `logging.getLogger(__name__)` at module level; never `print()`
- Private modules prefixed with underscore; define `__all__` in `__init__.py`
- Google-style docstrings on public API functions only
- Conventional commit format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

## Rules
- Never commit `.env` files or auth JSON; use `.env.tmpl` for templates
- All PRs must pass `tox` before merge
- Coverage must not drop below 80%
- Ruff is the only linter/formatter (no black, no isort)
- `unittest.TestCase` for sync tests, `unittest.IsolatedAsyncioTestCase` for async
- Use `@patch` decorators, never `with patch(...)` context managers
- No secrets, tokens, or PII in code or logs
