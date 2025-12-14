"""
YourBench structured logging system.

Provides clean, structured logging for the YourBench pipeline with:
- Thread-safe context management
- Stage tracking and metrics
- Console and file output
- Decorators for easy integration
"""
from typing import Optional, Union
from pathlib import Path

# Import types
from .types import (
    LogLevel,
    StageStatus,
    MetricsDict,
    LogRecord,
    StageMetrics,
    LogExtra,
    ErrorInfo
)

# Import core components
from .context import (
    LoggingContext,
    get_context,
    set_context,
    reset_context,
    with_context,
    update_context
)

from .core import (
    Logger,
    setup_logger,
    ConsoleHandler,
    FileHandler
)

from .formatters import (
    ConsoleFormatter,
    JSONFormatter,
    MinimalFormatter
)

from .decorators import (
    log_stage,
    log_timing,
    log_errors,
    log_async_stage
)

from .metrics import (
    MetricsCollector,
    StageMetricsSummary,
    get_metrics_collector,
    set_metrics_collector,
    reset_metrics_collector,
    collect_stage_metrics
)

# Global logger instance
_global_logger: Optional[Logger] = None


def get_logger(name: str = "yourbench") -> Logger:
    """
    Get or create the global logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = setup_default_logger(name)
    
    return _global_logger


def setup_default_logger(
    name: str = "yourbench",
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None
) -> Logger:
    """
    Set up the default logger configuration.
    
    Args:
        name: Logger name
        level: Log level
        log_file: Optional log file path
    
    Returns:
        Configured logger
    """
    import os
    from datetime import datetime
    
    # Determine log level from environment
    env_level = os.environ.get('YOURBENCH_LOG_LEVEL', '').upper()
    if env_level and hasattr(LogLevel, env_level):
        level = LogLevel[env_level]
    elif isinstance(level, str):
        level = LogLevel[level.upper()]
    
    # Determine log file from environment or default
    if log_file is None:
        log_file = os.environ.get('YOURBENCH_LOG_FILE')
    
    if log_file is None:
        # Create default log file in logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"yourbench_{timestamp}.jsonl"
    
    # Create logger
    logger = setup_logger(
        name=name,
        level=level,
        console=True,
        file_path=log_file
    )
    
    # Set as global logger
    global _global_logger
    _global_logger = logger
    
    return logger


def configure_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    console: bool = True,
    file_path: Optional[Union[str, Path]] = None,
    json_console: bool = False,
    run_id: Optional[str] = None
) -> Logger:
    """
    Configure the global logger with custom settings.
    
    Args:
        level: Log level
        console: Whether to log to console
        file_path: Optional file path for logs
        json_console: Use JSON format for console output
        run_id: Optional run ID for context
    
    Returns:
        Configured logger
    """
    if isinstance(level, str):
        level = LogLevel[level.upper()]
    
    logger = Logger("yourbench", level)
    
    # Add console handler
    if console:
        if json_console:
            formatter = JSONFormatter()
        else:
            formatter = ConsoleFormatter()
        console_handler = ConsoleHandler(formatter=formatter)
        logger.add_handler(console_handler)
    
    # Add file handler
    if file_path:
        file_handler = FileHandler(file_path, formatter=JSONFormatter())
        logger.add_handler(file_handler)
    
    # Set up context
    if run_id:
        with_context(run_id=run_id)
    
    # Set as global logger
    global _global_logger
    _global_logger = logger
    
    return logger


# Convenience functions that use the global logger
def trace(message: str, **kwargs):
    """Log at TRACE level using global logger."""
    get_logger().trace(message, **kwargs)


def debug(message: str, **kwargs):
    """Log at DEBUG level using global logger."""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log at INFO level using global logger."""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log at WARNING level using global logger."""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log at ERROR level using global logger."""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log at CRITICAL level using global logger."""
    get_logger().critical(message, **kwargs)


def success(message: str, **kwargs):
    """Log success using global logger."""
    get_logger().success(message, **kwargs)


def exception(message: str, exc: Optional[Exception] = None, **kwargs):
    """Log exception using global logger."""
    get_logger().exception(message, exc, **kwargs)


# Export all public symbols
__all__ = [
    # Types
    'LogLevel',
    'StageStatus',
    'MetricsDict',
    'LogRecord',
    'StageMetrics',
    'LogExtra',
    'ErrorInfo',
    
    # Context
    'LoggingContext',
    'get_context',
    'set_context',
    'reset_context',
    'with_context',
    'update_context',
    
    # Core
    'Logger',
    'setup_logger',
    'ConsoleHandler',
    'FileHandler',
    
    # Formatters
    'ConsoleFormatter',
    'JSONFormatter',
    'MinimalFormatter',
    
    # Decorators
    'log_stage',
    'log_timing',
    'log_errors',
    'log_async_stage',
    
    # Metrics
    'MetricsCollector',
    'StageMetricsSummary',
    'get_metrics_collector',
    'set_metrics_collector',
    'reset_metrics_collector',
    'collect_stage_metrics',
    
    # Public API
    'get_logger',
    'setup_default_logger',
    'configure_logging',
    
    # Convenience functions
    'trace',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'success',
    'exception',
]
