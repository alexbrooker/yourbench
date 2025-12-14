"""
Integration helpers for the new logging system.

Provides utilities to integrate with the existing codebase.
"""
import sys
import uuid
from typing import Optional
from pathlib import Path

from .core import setup_logger, Logger
from .types import LogLevel
from .context import with_context


def setup_pipeline_logging(
    config,
    run_id: Optional[str] = None,
    debug: bool = False
) -> Logger:
    """
    Set up logging for a pipeline run.
    
    Args:
        config: YourBench configuration object
        run_id: Optional run ID (will generate if not provided)
        debug: Whether to enable debug logging
    
    Returns:
        Configured logger instance
    """
    # Generate run ID if not provided
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    # Determine log level
    if debug:
        level = LogLevel.DEBUG
    elif hasattr(config, 'log_level'):
        level = LogLevel[config.log_level.upper()]
    else:
        level = LogLevel.INFO
    
    # Determine log file path
    if hasattr(config, 'output_dir'):
        output_dir = Path(config.output_dir)
    elif hasattr(config, 'pipeline') and hasattr(config.pipeline, 'ingestion'):
        output_dir = Path(config.pipeline.ingestion.output_dir)
    else:
        output_dir = Path('output')
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_run_{run_id}.jsonl"
    
    # Set up logger
    logger = setup_logger(
        name="yourbench.pipeline",
        level=level,
        console=True,
        file_path=log_file,
        run_id=run_id
    )
    
    # Log initial configuration info
    logger.info(
        f"Pipeline run started",
        extra={
            'run_id': run_id,
            'debug': debug,
            'log_level': level.value,
            'log_file': str(log_file),
            'dataset_name': getattr(config.hf_configuration, 'hf_dataset_name', 'unknown')
        }
    )
    
    return logger


def redirect_loguru_to_structured(logger: Logger):
    """
    Redirect loguru logs to our structured logger.
    
    This is a compatibility layer for existing code that uses loguru.
    
    Args:
        logger: Our structured logger instance
    """
    try:
        from loguru import logger as loguru_logger
        
        # Remove default loguru handler
        loguru_logger.remove()
        
        # Add custom sink that redirects to our logger
        def structured_sink(message):
            record = message.record
            level_map = {
                'TRACE': logger.trace,
                'DEBUG': logger.debug,
                'INFO': logger.info,
                'SUCCESS': logger.success,
                'WARNING': logger.warning,
                'ERROR': logger.error,
                'CRITICAL': logger.critical,
            }
            
            log_func = level_map.get(record['level'].name, logger.info)
            log_func(
                record['message'],
                extra={
                    'loguru_module': record['module'],
                    'loguru_function': record['function'],
                    'loguru_line': record['line']
                }
            )
        
        loguru_logger.add(structured_sink)
        
    except ImportError:
        # loguru not available, skip redirection
        pass


def get_progress_callback(logger: Logger, stage: str):
    """
    Create a progress callback function for a stage.
    
    Args:
        logger: Logger instance
        stage: Stage name
    
    Returns:
        Progress callback function
    """
    from .context import get_context
    
    def progress_callback(message: str, **metrics):
        context = get_context()
        context.add_progress(message)
        if metrics:
            context.update_metrics(**metrics)
        logger.debug(f"{stage}: {message}", extra={'metrics': metrics})
    
    return progress_callback
