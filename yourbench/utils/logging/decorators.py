"""
Decorators for enhanced logging in YourBench pipeline.

Provides decorators for stage tracking, timing, and error handling.
"""
import functools
import time
import inspect
from typing import Any, Callable, Optional, TypeVar, cast
from datetime import datetime

from .types import LogLevel, StageStatus, ProgressCallback
from .context import get_context, update_context
from .core import Logger

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


def log_stage(
    stage_name: Optional[str] = None,
    logger: Optional[Logger] = None,
    track_metrics: bool = True,
    report_progress: bool = True
) -> Callable[[F], F]:
    """
    Decorator to log pipeline stage execution.
    
    Args:
        stage_name: Name of the stage (defaults to function name)
        logger: Logger instance to use
        track_metrics: Whether to track metrics
        report_progress: Whether to enable progress reporting
    
    Example:
        @log_stage("ingestion")
        def ingest_documents(config):
            # Progress will be automatically tracked
            pass
    """
    def decorator(func: F) -> F:
        # Get stage name from parameter or function name
        name = stage_name or func.__name__.replace('_', ' ').title()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create logger
            log = logger
            if log is None:
                # Try to get from module
                module = inspect.getmodule(func)
                if hasattr(module, 'logger'):
                    log = getattr(module, 'logger')
                else:
                    # Create a default logger
                    from . import get_logger
                    log = get_logger()
            
            # Set up context
            context = get_context()
            context.start_stage(name)
            
            # Create progress callback
            def progress_callback(message: str, **metrics: Any) -> None:
                if report_progress:
                    context.add_progress(message)
                    if metrics:
                        context.update_metrics(**metrics)
                    log.debug(f"{name}: {message}", extra={'metrics': metrics})
            
            # Inject progress callback if function accepts it
            sig = inspect.signature(func)
            if '_progress_callback' in sig.parameters or 'progress_callback' in sig.parameters:
                kwargs['progress_callback'] = progress_callback
            
            # Track start time
            start_time = time.time()
            items_processed = 0
            items_failed = 0
            
            try:
                log.info(f"Starting {name}")
                context.stage_status = StageStatus.IN_PROGRESS
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Track metrics
                if track_metrics:
                    duration = time.time() - start_time
                    context.update_metrics(
                        duration_seconds=duration,
                        items_processed=items_processed,
                        items_failed=items_failed
                    )
                
                # Mark success
                metrics = context.complete_stage()
                log.success(
                    f"Completed {name}",
                    extra={
                        'metrics': metrics,
                        'progress_count': len(context.progress)
                    }
                )
                
                return result
                
            except Exception as e:
                # Track failure
                duration = time.time() - start_time
                context.update_metrics(
                    duration_seconds=duration,
                    items_processed=items_processed,
                    items_failed=items_failed,
                    error_count=1
                )
                
                metrics = context.fail_stage(e)
                
                # Log with context about what succeeded before failure
                log.error(
                    f"Failed {name} after {len(context.progress)} steps",
                    error=e,
                    extra={
                        'metrics': metrics,
                        'last_progress': context.progress[-1] if context.progress else None,
                        'progress_count': len(context.progress)
                    }
                )
                
                # Re-raise the exception
                raise
                
            finally:
                # Clean up context
                context.reset_stage()
        
        return cast(F, wrapper)
    
    return decorator


def log_timing(name: Optional[str] = None, logger: Optional[Logger] = None) -> Callable[[F], F]:
    """
    Simple decorator to log execution time.
    
    Args:
        name: Name for the timing (defaults to function name)
        logger: Logger instance to use
    """
    def decorator(func: F) -> F:
        timing_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            log = logger
            if log is None:
                from . import get_logger
                log = get_logger()
            
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                log.debug(f"{timing_name} took {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                log.debug(f"{timing_name} failed after {duration:.3f}s")
                raise
        
        return cast(F, wrapper)
    
    return decorator


def log_errors(
    name: Optional[str] = None,
    logger: Optional[Logger] = None,
    reraise: bool = True,
    default_return: Any = None
) -> Callable[[F], F]:
    """
    Decorator to log exceptions with context.
    
    Args:
        name: Name for error context (defaults to function name)
        logger: Logger instance to use
        reraise: Whether to re-raise the exception
        default_return: Value to return if exception is not re-raised
    """
    def decorator(func: F) -> F:
        error_context = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            log = logger
            if log is None:
                from . import get_logger
                log = get_logger()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log with full context
                log.exception(
                    f"Error in {error_context}",
                    exc=e,
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return cast(F, wrapper)
    
    return decorator


def log_async_stage(
    stage_name: Optional[str] = None,
    logger: Optional[Logger] = None
) -> Callable[[F], F]:
    """
    Decorator for async pipeline stages.
    
    Similar to log_stage but for async functions.
    """
    def decorator(func: F) -> F:
        name = stage_name or func.__name__.replace('_', ' ').title()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get logger
            log = logger
            if log is None:
                from . import get_logger
                log = get_logger()
            
            context = get_context()
            context.start_stage(name)
            
            start_time = time.time()
            
            try:
                log.info(f"Starting async {name}")
                context.stage_status = StageStatus.IN_PROGRESS
                
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                context.update_metrics(duration_seconds=duration)
                
                metrics = context.complete_stage()
                log.success(f"Completed async {name}", extra={'metrics': metrics})
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                context.update_metrics(duration_seconds=duration, error_count=1)
                
                metrics = context.fail_stage(e)
                log.error(f"Failed async {name}", error=e, extra={'metrics': metrics})
                raise
                
            finally:
                context.reset_stage()
        
        return cast(F, async_wrapper)
    
    return decorator
