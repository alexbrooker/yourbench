#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models."""

from __future__ import annotations
import sys
from typing import Optional
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from rich.table import Table
from rich.console import Console

from yourbench.analysis import run_analysis
from yourbench.pipeline.handler import run_pipeline
from yourbench.config_builder import create_yourbench_config, save_config


load_dotenv()

app = typer.Typer(
    name="yourbench",
    help="YourBench - Dynamic Evaluation Set Generation with Large Language Models.",
    pretty_exceptions_show_locals=False,
)
console = Console()


@app.command()
def run(
    config_path: Optional[Path] = typer.Argument(
        None,
        help="Path to configuration file (YAML/JSON)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="[LEGACY] Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    plot_stage_timing: bool = typer.Option(
        False,
        "--plot-stage-timing",
        help="Generate stage timing chart",
    ),
) -> None:
    """Run the YourBench pipeline with a configuration file."""
    # Handle both new positional and legacy --config
    final_config = config_path or config

    if not final_config:
        console.print("[red]Error:[/red] Please provide a configuration file")
        console.print("Usage: yourbench run CONFIG_FILE")
        raise typer.Exit(1)

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    logger.info(f"Running pipeline with config: {final_config}")

    try:
        run_pipeline(
            config_file_path=str(final_config),
            debug=debug,
            plot_stage_timing=plot_stage_timing,
        )
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command()
def create(
    output: Path = typer.Argument(
        "config.yaml",
        help="Output configuration file path",
    ),
    simple: bool = typer.Option(
        False,
        "--simple",
        "-s",
        help="Create a simple configuration with minimal options",
    ),
) -> None:
    """Create a new YourBench configuration file interactively."""
    try:
        config = create_yourbench_config(simple=simple)
        save_config(config, output)
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        if config.pipeline_config.ingestion.run:
            src_dir = config.pipeline_config.ingestion.source_documents_dir
            console.print(f"1. Place your documents in: {src_dir}")
        console.print(f"2. Run: [cyan]yourbench run {output}[/cyan]")
        
        # Remind about .env if API keys were used
        if any(m.api_key and m.api_key.startswith("$") for m in config.model_list):
            console.print("\n[yellow]Don't forget to update your .env file with actual API keys![/yellow]")
    
    except Exception as e:
        logger.error(f"Configuration creation failed: {e}")
        console.print(f"[red]Error: Could not create configuration[/red]")
        console.print(f"[red]Details: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    analysis_name: str = typer.Argument(..., help="Name of the analysis to run"),
    args: list[str] = typer.Argument(None, help="Additional arguments"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Run a specific analysis by name."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    logger.info(f"Running analysis '{analysis_name}' with arguments: {args}")

    try:
        run_analysis(analysis_name, args, debug=debug)
    except Exception as e:
        logger.exception(f"Analysis '{analysis_name}' failed: {e}")
        raise typer.Exit(1)


@app.command()
def gui() -> None:
    """Launch the Gradio UI (not yet implemented)."""
    logger.error("GUI support is not yet implemented")
    raise typer.Exit(1)


@app.command()
def help() -> None:
    """Show detailed help information for all YourBench commands."""
    console.print("[bold green]YourBench CLI Help[/bold green]\n")

    console.print("YourBench is a dynamic evaluation set generation tool using Large Language Models.")
    console.print("It converts documents into comprehensive evaluation datasets with questions and answers.\n")

    # Commands table
    table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", width=12)
    table.add_column("Description", style="white", width=50)
    table.add_column("Usage", style="green", width=30)

    commands = [
        (
            "run",
            "Execute the YourBench pipeline with a configuration file. Processes documents through ingestion, summarization, chunking, and question generation stages.",
            "yourbench run config.yaml",
        ),
        (
            "create",
            "Interactive configuration file creator. Guides you through setting up models, pipeline stages, and Hugging Face integration.",
            "yourbench create [--simple]",
        ),
        (
            "analyze",
            "Run specific analysis scripts on generated datasets. Includes various evaluation and visualization tools.",
            "yourbench analyze ANALYSIS_NAME",
        ),
        ("gui", "Launch the Gradio web interface for YourBench (not yet implemented).", "yourbench gui"),
        ("help", "Show this detailed help information about all commands.", "yourbench help"),
    ]

    for cmd, desc, usage in commands:
        table.add_row(cmd, desc, usage)

    console.print(table)

    # Quick start section
    console.print("\n[bold cyan]Quick Start:[/bold cyan]")
    console.print("1. [green]yourbench create[/green] - Create a configuration file")
    console.print("2. Place documents in [yellow]data/raw/[/yellow] directory")
    console.print("3. [green]yourbench run config.yaml[/green] - Process documents")

    # Examples section
    console.print("\n[bold cyan]Examples:[/bold cyan]")
    console.print("• Create simple config:    [green]yourbench create --simple[/green]")
    console.print("• Run with debug:          [green]yourbench run config.yaml --debug[/green]")
    console.print("• Show stage timing:       [green]yourbench run config.yaml --plot-stage-timing[/green]")
    console.print("• Run citation analysis:   [green]yourbench analyze citation_score[/green]")

    console.print("\n[bold cyan]For More Help:[/bold cyan]")
    console.print("• Use [green]yourbench COMMAND --help[/green] for command-specific options")
    console.print("• Visit the documentation for detailed guides and examples")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
