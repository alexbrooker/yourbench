"""
Type definitions for the YourBench logging system.

This module provides type definitions for type safety and better IDE support.
"""
from typing import Dict, Any, Optional, Union, TypedDict, Literal, Protocol
from datetime import datetime
from enum import Enum


class LogLevel(str, Enum):
    """Log severity levels."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"


class StageStatus(str, Enum):
    """Pipeline stage execution status."""
    NOT_STARTED = "NOT_STARTED"
    STARTING = "STARTING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class MetricsDict(TypedDict, total=False):
    """Type definition for stage metrics."""
    duration_seconds: float
    items_processed: int
    items_failed: int
    tokens_used: int
    api_calls: int
    cost_usd: float
    error_count: int
    warnings: list[str]


class LogRecord(TypedDict):
    """Structure of a log record."""
    timestamp: str
    level: str
    message: str
    stage: Optional[str]
    run_id: Optional[str]
    correlation_id: Optional[str]
    extra: Optional[Dict[str, Any]]
    metrics: Optional[MetricsDict]
    error: Optional[Dict[str, Any]]


class LogFormatter(Protocol):
    """Protocol for log formatters."""
    
    def format(self, record: LogRecord) -> str:
        """Format a log record into a string."""
        ...


class LogHandler(Protocol):
    """Protocol for log handlers."""
    
    def emit(self, record: LogRecord) -> None:
        """Emit a log record."""
        ...
    
    def flush(self) -> None:
        """Flush any buffered records."""
        ...
    
    def close(self) -> None:
        """Clean up handler resources."""
        ...


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""
    
    def __call__(self, message: str, **kwargs: Any) -> None:
        """Report progress."""
        ...


# Type aliases
StageMetrics = Dict[str, MetricsDict]
LogExtra = Dict[str, Any]
ErrorInfo = Dict[str, Any]
