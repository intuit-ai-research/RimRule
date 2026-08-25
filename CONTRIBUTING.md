# Contributing to RimRule

Thanks for your interest in contributing. This document covers how to get set up,
what we expect from a change, and how to get it reviewed.

## Code of Conduct

This project is governed by our [Code of Conduct](./CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it. Please report unacceptable behavior
to TechOpenSource@intuit.com.

## Getting started

RimRule uses [uv](https://docs.astral.sh/uv/) for dependency management and
[tox](https://tox.wiki/) to run checks.

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/RimRule.git
cd RimRule
git remote add upstream https://github.com/intuit-ai-research/RimRule.git

# Install all dependencies (Python 3.12)
uv sync --all-extras

# Optional: install pre-commit hooks (ruff check + format on commit)
uv run pre-commit install
```

See the [README](./README.md#local-development) for the full local development loop.

## Making a change

1. Create a branch off `main`: `git checkout -b my-change`
2. Make your change, with tests.
3. Run the full gate locally: `tox`
4. Push to your fork and open a pull request.

### Commit messages

Use [conventional commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
`docs:`, `refactor:`, `test:`, `chore:`.

## What we expect

Before a pull request can be merged:

- **`tox` passes.** This runs `ruff check`, `ruff format --check`, `mypy`, and the
  test suite with coverage.
- **New code has tests.** Unit tests live in `tests/unit/`, mirroring the package
  layout. Use `unittest.TestCase` (or `unittest.IsolatedAsyncioTestCase` for async)
  and `@patch` decorators rather than context managers.
- **Coverage does not drop below 80%.** This is enforced by the coverage gate.
- **Type hints on all public functions**, including return types. The project
  targets Python 3.12: use `list[str]` and `float | None`, not `typing.List` or
  `typing.Optional`.
- **No secrets, tokens, or PII** in code, tests, logs, or fixtures. Never commit
  `.env` files or credential JSON; `.env.tmpl` is the template.

Unit tests must not make network calls -- the root `conftest.py` blocks socket
connections to enforce this. A test that legitimately needs the network belongs in
`tests/integration/`; if it hits real services, mark it `@pytest.mark.e2e`, which is
excluded from the default run.

Never call a real LLM in a unit test. To stand in for a chat model, use
`FakeChatModel` from `tests/unit/_fakes.py`, which replays a queued list of
`AIMessage`s one per invocation and supports `bind_tools` for agent tests. Patch at
the import site (`@patch("rimrule.module.dependency")`), not the definition site, and
use `AsyncMock` for async callables.

Build test data with short module-level factory helpers (e.g. `_make_result(score=0.9)`),
kept private to each test file. Assert with `self.assert*` methods rather than bare
`assert`. Don't write tests for Pydantic models themselves -- their correctness is
covered indirectly by the logic that uses them.

## Code style

Ruff is the only linter and formatter -- no black, no isort. Line length is 88, and
the enabled rule sets are `E, W, F, B, C4, UP, I`.

- Pydantic v2 `BaseModel` for structured data, with `Field(description=...)` on every
  field. Use `ConfigDict(frozen=True)` for immutable objects. For models, that
  `Field` description replaces a docstring.
- `logging.getLogger(__name__)` at module level; never `print()` for operational
  output. Use `logger.exception()` inside exception handlers -- it captures the
  traceback automatically.
- Google-style docstrings on public API functions only. Private functions need none
  when the name and types are clear.
- Comment the *why*, not the *what*. Skip comments that restate the code.
- Underscore-prefix private modules (`_fakes.py`, `_parser.py`) and define `__all__`
  in each `__init__.py`. Import from the defining module rather than through
  re-exports.
- `asyncio.gather` for concurrent independent I/O, `asyncio.to_thread` for CPU-bound
  work inside async code. Bound concurrent LLM calls with an `asyncio.Semaphore`
  rather than an unbounded `gather` over a large list.
- No emojis in code or commit messages.

All LLM calls go through LangChain (`langchain-openai`'s `ChatOpenAI`) -- never the
OpenAI SDK directly or raw HTTP. Client construction is centralized in `get_llm()`
in `rimrule/llm/factory.py`; don't scatter `ChatOpenAI(...)` across the codebase. Use
`.with_structured_output(Model, method="function_calling")` for structured output.

Pipeline configuration is YAML-driven through `RimRuleConfig.load()`. A config may
set `base: <relative-path>` to inherit shared defaults from another file (see
`configs/base.yaml`). Constants that don't vary per run belong in module-level
constants, not in config models.

## Reporting bugs

Open a GitHub issue. A useful report includes what you ran, what you expected, what
happened, and the versions of Python and RimRule you are on. For pipeline issues,
the relevant stage and the contents of the stage's state file in your `--fld`
directory are usually the fastest path to a diagnosis.

Please do not open a public issue for a suspected security vulnerability. Report
it privately to TechOpenSource@intuit.com instead.

## Licensing

RimRule is licensed under the [Apache License 2.0](./LICENSE). By submitting a
contribution, you agree that it is licensed under the same terms, as described in
section 5 of that license:

> Unless You explicitly state otherwise, any Contribution intentionally submitted
> for inclusion in the Work by You to the Licensor shall be under the terms and
> conditions of this License, without any additional terms or conditions.

Only contribute code you have the right to license this way.
