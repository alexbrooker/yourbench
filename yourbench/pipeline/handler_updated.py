"""Enhanced pipeline orchestrator with optional structured logging support."""

import os
import time
import importlib
import uuid

from loguru import logger as loguru_logger
from yourbench.conf.loader import get_enabled_stages

# Check if structured logging is enabled via environment
USE_STRUCTURED_LOGGING = os.environ.get('YOURBENCH_STRUCTURED_LOGGING', 'false').lower() == 'true'

if USE_STRUCTURED_LOGGING:
    try:
        from yourbench.utils.logging import (
            get_logger,
            get_context,
            get_metrics_collector
        )
        from yourbench.utils.logging.integration import (
            setup_pipeline_logging,
            redirect_loguru_to_structured
        )
        STRUCTURED_LOGGING_AVAILABLE = True
    except ImportError:
        STRUCTURED_LOGGING_AVAILABLE = False
        USE_STRUCTURED_LOGGING = False
else:
    STRUCTURED_LOGGING_AVAILABLE = False


# Use the appropriate logger
if USE_STRUCTURED_LOGGING and STRUCTURED_LOGGING_AVAILABLE:
    logger = get_logger()
else:
    logger = loguru_logger


# Stage function overrides for question generation
_stage_overrides = {}


def _get_stage_overrides():
    """Lazy load question generation functions."""
    global _stage_overrides
    if not _stage_overrides:
        from yourbench.pipeline.question_generation import (
            run_multi_hop,
            run_single_shot,
            run_cross_document,
        )

        _stage_overrides = {
            "single_shot_question_generation": run_single_shot,
            "multi_hop_question_generation": run_multi_hop,
            "cross_document_question_generation": run_cross_document,
        }
    return _stage_overrides


def _get_stage_function(stage: str):
    """Get the function for a pipeline stage."""
    overrides = _get_stage_overrides()
    if stage in overrides:
        return overrides[stage]

    # Handle legacy name
    if stage == "lighteval":
        logger.warning("'lighteval' is deprecated, use 'prepare_lighteval'")
        stage = "prepare_lighteval"

    module = importlib.import_module(f"yourbench.pipeline.{stage}")
    return module.run


def run_stage(stage: str, config) -> float:
    """Run a single pipeline stage, return elapsed time."""
    if USE_STRUCTURED_LOGGING and STRUCTURED_LOGGING_AVAILABLE:
        # Use structured logging with context
        context = get_context()
        context.start_stage(stage)
        logger.info(f"Starting {stage}")
        
        start = time.perf_counter()
        try:
            _get_stage_function(stage)(config)
            elapsed = time.perf_counter() - start
            
            context.update_metrics(duration_seconds=elapsed)
            metrics = context.complete_stage()
            logger.success(f"Completed {stage}", extra={'metrics': metrics})
            
            return elapsed
        except Exception as e:
            elapsed = time.perf_counter() - start
            context.update_metrics(duration_seconds=elapsed, error_count=1)
            metrics = context.fail_stage(e)
            logger.error(f"Failed {stage}", error=e, extra={'metrics': metrics})
            raise
        finally:
            context.reset_stage()
    else:
        # Use regular loguru logging
        logger.info(f"Running {stage}")
        start = time.perf_counter()
        try:
            _get_stage_function(stage)(config)
            elapsed = time.perf_counter() - start
            return elapsed
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

    # Set up structured logging if enabled
    if USE_STRUCTURED_LOGGING and STRUCTURED_LOGGING_AVAILABLE:
        run_id = str(uuid.uuid4())[:8]
        structured_logger = setup_pipeline_logging(config, run_id=run_id, debug=debug)
        redirect_loguru_to_structured(structured_logger)
        logger.info(f"Pipeline run started with structured logging (run_id: {run_id})")

    enabled = get_enabled_stages(config)
    if not enabled:
        logger.warning("No pipeline stages enabled")
        return

    logger.info(f"Running stages: {', '.join(enabled)}")

    elapsed_times = {}
    failed_stage = None
    
    try:
        for stage in enabled:
            elapsed = run_stage(stage, config)
            elapsed_times[stage] = elapsed
            logger.success(f"Completed {stage} in {elapsed:.2f}s")
    
    except Exception as e:
        failed_stage = stage if 'stage' in locals() else 'unknown'
        logger.error(f"Pipeline failed at stage: {failed_stage}")
        raise
    
    finally:
        # Log summary if using structured logging
        if USE_STRUCTURED_LOGGING and STRUCTURED_LOGGING_AVAILABLE:
            total_time = sum(elapsed_times.values())
            collector = get_metrics_collector()
            summary = collector.get_pipeline_summary()
            
            logger.info(
                f"Pipeline {'completed' if not failed_stage else 'failed'}",
                extra={
                    'total_time': total_time,
                    'stages_completed': len(elapsed_times),
                    'stages_total': len(enabled),
                    'failed_stage': failed_stage,
                    'summary': summary
                }
            )

    # Upload dataset card
    try:
        from yourbench.utils.dataset_engine import upload_dataset_card
        upload_dataset_card(config)
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")
