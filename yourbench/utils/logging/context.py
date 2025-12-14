"""
Thread-safe context management using contextvars.

Provides context-aware logging that supports async operations and prevents
state leakage between concurrent executions.
"""
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from .types import StageStatus, MetricsDict


@dataclass
class LoggingContext:
    """Thread-local context for logging."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage: Optional[str] = None
    stage_status: StageStatus = StageStatus.NOT_STARTED
    stage_started_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    metrics: MetricsDict = field(default_factory=dict)
    progress: list[str] = field(default_factory=list)
    
    def reset_stage(self) -> None:
        """Reset stage-specific context."""
        self.stage = None
        self.stage_status = StageStatus.NOT_STARTED
        self.stage_started_at = None
        self.metrics = {}
        self.progress = []
    
    def start_stage(self, stage_name: str) -> None:
        """Start a new stage."""
        self.stage = stage_name
        self.stage_status = StageStatus.STARTING
        self.stage_started_at = datetime.utcnow()
        self.metrics = {}
        self.progress = []
    
    def update_metrics(self, **kwargs: Any) -> None:
        """Update stage metrics."""
        self.metrics.update(kwargs)
    
    def add_progress(self, message: str) -> None:
        """Add a progress message."""
        self.progress.append(message)
    
    def complete_stage(self) -> MetricsDict:
        """Mark stage as complete and return final metrics."""
        self.stage_status = StageStatus.COMPLETED
        if self.stage_started_at:
            duration = (datetime.utcnow() - self.stage_started_at).total_seconds()
            self.metrics['duration_seconds'] = duration
        return self.metrics.copy()
    
    def fail_stage(self, error: Optional[Exception] = None) -> MetricsDict:
        """Mark stage as failed and return metrics."""
        self.stage_status = StageStatus.FAILED
        if self.stage_started_at:
            duration = (datetime.utcnow() - self.stage_started_at).total_seconds()
            self.metrics['duration_seconds'] = duration
        if error:
            self.metrics['error_count'] = self.metrics.get('error_count', 0) + 1
        return self.metrics.copy()


# Context variable for thread-safe context management
_logging_context: ContextVar[Optional[LoggingContext]] = ContextVar(
    'logging_context',
    default=None
)


def get_context() -> LoggingContext:
    """Get or create the current logging context."""
    context = _logging_context.get()
    if context is None:
        context = LoggingContext()
        _logging_context.set(context)
    return context


def set_context(context: LoggingContext) -> None:
    """Set the current logging context."""
    _logging_context.set(context)


def reset_context() -> None:
    """Reset the logging context."""
    _logging_context.set(None)


def with_context(**kwargs: Any) -> LoggingContext:
    """Create a new context with the given attributes."""
    context = LoggingContext(**kwargs)
    _logging_context.set(context)
    return context


def update_context(**kwargs: Any) -> None:
    """Update the current context with new values."""
    context = get_context()
    for key, value in kwargs.items():
        if hasattr(context, key):
            setattr(context, key, value)
        else:
            context.extra[key] = value
