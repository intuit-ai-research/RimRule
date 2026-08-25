from __future__ import annotations

import logging
from pathlib import Path

import typer

from rimrule.config import RimRuleConfig
from rimrule.pipeline.stages import canonicalize, collect, consolidate, induce
from rimrule.toolhop.loader import load_toolhop, schema_summary

app = typer.Typer(no_args_is_help=True)


def configure_quiet_http_logging() -> None:
    """Silence noisy HTTP and auth loggers so progress bars stay readable."""
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)


ArtifactsOption = typer.Option(
    None,
    "--fld",
    help="Artifact directory. Defaults to the config value (normally 'artifacts').",
    file_okay=False,
    dir_okay=True,
    resolve_path=False,
)


def cfg(path: Path, fld: Path | None = None) -> RimRuleConfig:
    """Load config from ``path`` and apply the artifacts-dir override."""
    config = RimRuleConfig.load(path)
    if fld is not None:
        config = config.with_artifacts_dir(fld)
    return config


@app.command("validate-data")
def validate_data(
    config: Path = typer.Option(Path("configs/toolhop.yaml"), "--config"),
    fld: Path | None = ArtifactsOption,
) -> None:
    """Validate the ToolHop dataset and print a reference-trace summary."""
    c = cfg(config, fld=fld)
    typer.echo(schema_summary(load_toolhop(c.dataset.path, c.dataset.strict_schema)))


@app.command()
def run_collect(
    config: Path = typer.Option(Path("configs/toolhop.yaml"), "--config"),
    fld: Path | None = ArtifactsOption,
) -> None:
    """Stage 1: run the base agent on ToolHop and record failures."""
    configure_quiet_http_logging()
    failures = collect(cfg(config, fld))
    typer.echo(f"Collection complete: {len(failures)} failures saved")


@app.command()
def run_induce(
    config: Path = typer.Option(Path("configs/toolhop.yaml"), "--config"),
    fld: Path | None = ArtifactsOption,
) -> None:
    """Stage 2: propose candidate rules from recorded failure traces."""
    configure_quiet_http_logging()
    typer.echo(f"Accepted {len(induce(cfg(config, fld)))} candidate rules")


@app.command()
def run_canonicalize(
    config: Path = typer.Option(Path("configs/toolhop.yaml"), "--config"),
    fld: Path | None = ArtifactsOption,
) -> None:
    """Stage 3: translate candidate rules into the symbolic vocabulary."""
    configure_quiet_http_logging()
    _, rules = canonicalize(cfg(config, fld))
    typer.echo(f"Canonicalized {len(rules)} rules")


@app.command()
def run_consolidate(
    config: Path = typer.Option(Path("configs/toolhop.yaml"), "--config"),
    fld: Path | None = ArtifactsOption,
) -> None:
    """Stage 4: MDL-optimize the symbolic rule library."""
    configure_quiet_http_logging()
    typer.echo(f"Consolidated to {len(consolidate(cfg(config, fld)))} rules")


@app.command()
def run_all(
    config: Path = typer.Option(Path("configs/toolhop.yaml"), "--config"),
    fld: Path | None = ArtifactsOption,
) -> None:
    """Run all four pipeline stages end to end."""
    configure_quiet_http_logging()
    c = cfg(config, fld)
    collect(c)
    induce(c)
    canonicalize(c)
    consolidate(c)


if __name__ == "__main__":
    app()
