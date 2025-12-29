"""Pipeline orchestrator for Yourbench with Rich progress tracking."""

import time
import importlib

from loguru import logger
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn

from yourbench.conf.loader import get_enabled_stages


# Map stage names to module paths for stages that live in subfolders
_STAGE_MODULE_MAP = {
    "single_hop_question_generation": "question_generation.single_hop",
    "multi_hop_question_generation": "question_generation.multi_hop",
    "cross_document_question_generation": "question_generation.cross_document",
}

# Human-readable stage names
_STAGE_DISPLAY_NAMES = {
    "ingestion": "Document Ingestion",
    "summarization": "Summarization",
    "chunking": "Chunking",
    "single_hop_question_generation": "Single-Hop Questions",
    "multi_hop_question_generation": "Multi-Hop Questions",
    "cross_document_question_generation": "Cross-Document Questions",
    "question_rewriting": "Question Rewriting",
    "prepare_lighteval": "LightEval Preparation",
    "citation_score_filtering": "Citation Filtering",
}


def _get_stage_function(stage: str):
    """Get the function for a pipeline stage."""
    module_path = _STAGE_MODULE_MAP.get(stage, stage)
    module = importlib.import_module(f"yourbench.pipeline.{module_path}")
    return module.run


def run_stage(stage: str, config) -> float:
    """Run a single pipeline stage, return elapsed time."""
    logger.info(f"Running {stage}")
    start = time.perf_counter()
    try:
        _get_stage_function(stage)(config)
        return time.perf_counter() - start
    except Exception:
        logger.exception(f"Error in {stage}")
        raise


def run_pipeline(config_path: str, debug: bool = False) -> None:
    """Run the full pipeline from a config file path."""
    from yourbench.conf.loader import load_config

    config = load_config(config_path)
    if debug:
        config.debug = True

    run_pipeline_with_config(config, debug=debug)


def run_pipeline_with_config(config, debug: bool = False) -> None:
    """Run the pipeline with a pre-loaded config object."""
    if debug:
        config.debug = True

    enabled = get_enabled_stages(config)
    if not enabled:
        logger.warning("No pipeline stages enabled")
        return

    logger.info(f"Running stages: {', '.join(enabled)}")

    for stage in enabled:
        elapsed = run_stage(stage, config)
        logger.success(f"Completed {stage} in {elapsed:.2f}s")

    # Upload dataset card
    try:
        from yourbench.utils.dataset_card import upload_dataset_card

        upload_dataset_card(config)
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")


def run_pipeline_with_progress(config, debug: bool = False, quiet: bool = False, console: Console = None) -> None:
    """Run the pipeline with Rich progress tracking."""
    if debug:
        config.debug = True

    if console is None:
        console = Console()

    enabled = get_enabled_stages(config)
    if not enabled:
        logger.warning("No pipeline stages enabled")
        return

    logger.info(f"Running stages: {', '.join(enabled)}")

    stage_times = {}

    if quiet:
        # Quiet mode: no progress display
        for stage in enabled:
            elapsed = run_stage(stage, config)
            stage_times[stage] = elapsed
            logger.success(f"Completed {stage} in {elapsed:.2f}s")
    else:
        # Progress display mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            overall_task = progress.add_task(f"[cyan]Pipeline ({len(enabled)} stages)", total=len(enabled))

            for i, stage in enumerate(enabled):
                display_name = _STAGE_DISPLAY_NAMES.get(stage, stage)
                stage_task = progress.add_task(
                    f"[green]{display_name}",
                    total=None,  # Indeterminate
                )

                try:
                    elapsed = run_stage(stage, config)
                    stage_times[stage] = elapsed
                    progress.update(
                        stage_task, completed=True, description=f"[green]\u2713 {display_name} ({elapsed:.1f}s)"
                    )
                    progress.remove_task(stage_task)
                except Exception as e:
                    progress.update(stage_task, description=f"[red]\u2717 {display_name} (failed)")
                    raise

                progress.update(overall_task, advance=1)

        # Print stage timing summary
        console.print()
        console.print("[bold]Stage Timing:[/bold]")
        total_time = sum(stage_times.values())
        for stage, elapsed in stage_times.items():
            display_name = _STAGE_DISPLAY_NAMES.get(stage, stage)
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            console.print(f"  [cyan]{display_name}:[/cyan] {elapsed:.2f}s ({pct:.0f}%)")
        console.print(f"  [bold]Total:[/bold] {total_time:.2f}s")

    # Upload dataset card
    try:
        from yourbench.utils.dataset_card import upload_dataset_card

        upload_dataset_card(config)
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")
