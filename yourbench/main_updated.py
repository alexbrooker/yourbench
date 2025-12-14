#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models."""

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

# Use structured logging if available, fallback to loguru
try:
    from yourbench.utils.logging import (
        configure_logging,
        get_logger,
        LogLevel
    )
    STRUCTURED_LOGGING_AVAILABLE = True
except ImportError:
    from loguru import logger
    STRUCTURED_LOGGING_AVAILABLE = False

load_dotenv()

app = typer.Typer(
    name="yourbench",
    help="YourBench - Dynamic Evaluation Set Generation with Large Language Models.",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


def setup_logging(debug: bool = False):
    """Set up logging based on available system."""
    if STRUCTURED_LOGGING_AVAILABLE:
        level = LogLevel.DEBUG if debug else LogLevel.INFO
        env_level = os.getenv("YOURBENCH_LOG_LEVEL", "").upper()
        if env_level and hasattr(LogLevel, env_level):
            level = LogLevel[env_level]
        
        # Set up structured logging
        configure_logging(
            level=level,
            console=True,
            file_path=None  # Will use default based on timestamp
        )
        return get_logger()
    else:
        # Fallback to loguru
        logger.remove()
        level = "DEBUG" if debug else os.getenv("YOURBENCH_LOG_LEVEL", "INFO")
        logger.add(sys.stderr, level=level)
        return logger


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to YAML config file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Run YourBench pipeline with a config file."""
    logger = setup_logging(debug)

    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(1)

    if config_file.suffix not in {".yaml", ".yml"}:
        logger.error(f"Config must be a YAML file (.yaml or .yml): {config_path}")
        raise typer.Exit(1)

    logger.info(f"Running with config: {config_file}")

    from yourbench.conf.loader import load_config
    from yourbench.pipeline.handler_updated import run_pipeline_with_config
    from yourbench.utils.dataset_engine import upload_dataset_card

    try:
        config = load_config(config_file)
        if debug:
            config.debug = True
        run_pipeline_with_config(config, debug=debug)
        try:
            upload_dataset_card(config)
        except Exception as e:
            logger.warning(f"Failed to upload dataset card: {e}")
    except Exception as e:
        if STRUCTURED_LOGGING_AVAILABLE:
            logger.exception(f"Pipeline failed", exc=e)
        else:
            logger.exception(f"Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command("version")
def version_command() -> None:
    """Show YourBench version."""
    from importlib.metadata import version as get_version

    try:
        v = get_version("yourbench")
        print(f"YourBench version: {v}")
    except Exception:
        print("YourBench version: development")


def main() -> None:
    """Entry point for the CLI."""
    # Handle version flag
    if len(sys.argv) == 2 and sys.argv[1] in {"--version", "-v"}:
        version_command()
        return

    app()


if __name__ == "__main__":
    main()
