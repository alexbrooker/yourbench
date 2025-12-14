"""
Core logger implementation for YourBench.

Provides the main Logger class with thread-safe operations and multiple handlers.
"""
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
import threading
from contextlib import contextmanager

from .types import LogLevel, LogRecord, LogHandler, LogFormatter, ErrorInfo
from .context import get_context, reset_context, with_context
from .formatters import ConsoleFormatter, JSONFormatter


class ConsoleHandler:
    """Handler for console output."""
    
    def __init__(self, formatter: Optional[LogFormatter] = None, stream=None):
        self.formatter = formatter or ConsoleFormatter()
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()
    
    def emit(self, record: LogRecord) -> None:
        """Write record to console."""
        try:
            with self._lock:
                formatted = self.formatter.format(record)
                self.stream.write(formatted + '\n')
                self.stream.flush()
        except Exception:
            # Logging should never crash the application
            pass
    
    def flush(self) -> None:
        """Flush the stream."""
        try:
            self.stream.flush()
        except Exception:
            pass
    
    def close(self) -> None:
        """Close handler (no-op for console)."""
        pass


class FileHandler:
    """Handler for file output."""
    
    def __init__(self, filename: Union[str, Path], formatter: Optional[LogFormatter] = None):
        self.filename = Path(filename)
        self.formatter = formatter or JSONFormatter()
        self._file = None
        self._lock = threading.Lock()
        self._ensure_file()
    
    def _ensure_file(self) -> None:
        """Ensure the log file is open."""
        if self._file is None or self._file.closed:
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.filename, 'a', encoding='utf-8')
    
    def emit(self, record: LogRecord) -> None:
        """Write record to file."""
        try:
            with self._lock:
                self._ensure_file()
                formatted = self.formatter.format(record)
                self._file.write(formatted + '\n')
                self._file.flush()
        except Exception:
            # Logging should never crash the application
            pass
    
    def flush(self) -> None:
        """Flush the file."""
        try:
            if self._file:
                self._file.flush()
        except Exception:
            pass
    
    def close(self) -> None:
        """Close the file."""
        try:
            if self._file:
                self._file.close()
                self._file = None
        except Exception:
            pass


class Logger:
    """Main logger class for YourBench."""
    
    def __init__(self, name: str = "yourbench", level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self.handlers: List[LogHandler] = []
        self._lock = threading.Lock()
    
    def add_handler(self, handler: LogHandler) -> None:
        """Add a log handler."""
        with self._lock:
            self.handlers.append(handler)
    
    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a log handler."""
        with self._lock:
            if handler in self.handlers:
                self.handlers.remove(handler)
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if a level should be logged."""
        level_values = {
            LogLevel.TRACE: 10,
            LogLevel.DEBUG: 20,
            LogLevel.INFO: 30,
            LogLevel.WARNING: 40,
            LogLevel.ERROR: 50,
            LogLevel.CRITICAL: 60,
            LogLevel.SUCCESS: 30,  # Same as INFO
        }
        return level_values.get(level, 30) >= level_values.get(self.level, 30)
    
    def _create_record(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> LogRecord:
        """Create a log record."""
        context = get_context()
        
        record: LogRecord = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level.value,
            'message': message,
            'stage': context.stage,
            'run_id': context.run_id,
            'correlation_id': context.correlation_id,
            'extra': extra or context.extra,
            'metrics': context.metrics if context.metrics else None,
            'error': None
        }
        
        if error:
            record['error'] = {
                'type': error.__class__.__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }
        
        return record
    
    def _emit(self, record: LogRecord) -> None:
        """Emit a record to all handlers."""
        for handler in self.handlers:
            try:
                handler.emit(record)
            except Exception:
                # Handler errors should not propagate
                pass
    
    def log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """Log a message at the given level."""
        if not self._should_log(level):
            return
        
        error = kwargs.pop('error', None)
        extra = kwargs.pop('extra', None)
        
        record = self._create_record(level, message, extra, error)
        self._emit(record)
    
    def trace(self, message: str, **kwargs: Any) -> None:
        """Log at TRACE level."""
        self.log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def success(self, message: str, **kwargs: Any) -> None:
        """Log a success message."""
        self.log(LogLevel.SUCCESS, message, **kwargs)
    
    def exception(self, message: str, exc: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        if exc is None:
            # Try to get from current exception context
            exc_info = sys.exc_info()
            if exc_info[0]:
                exc = exc_info[1]
        
        self.error(message, error=exc, **kwargs)
    
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for logging a pipeline stage."""
        context = get_context()
        context.start_stage(stage_name)
        
        self.info(f"Starting {stage_name}")
        
        try:
            yield context
            metrics = context.complete_stage()
            self.success(f"Completed {stage_name}", extra={'metrics': metrics})
        except Exception as e:
            metrics = context.fail_stage(e)
            self.error(f"Failed {stage_name}", error=e, extra={'metrics': metrics})
            raise
        finally:
            context.reset_stage()
    
    def flush(self) -> None:
        """Flush all handlers."""
        for handler in self.handlers:
            try:
                handler.flush()
            except Exception:
                pass
    
    def close(self) -> None:
        """Close all handlers."""
        for handler in self.handlers:
            try:
                handler.close()
            except Exception:
                pass


def setup_logger(
    name: str = "yourbench",
    level: Union[str, LogLevel] = LogLevel.INFO,
    console: bool = True,
    file_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None
) -> Logger:
    """Set up a logger with default configuration."""
    if isinstance(level, str):
        level = LogLevel[level.upper()]
    
    logger = Logger(name, level)
    
    # Add console handler
    if console:
        console_handler = ConsoleHandler(formatter=ConsoleFormatter())
        logger.add_handler(console_handler)
    
    # Add file handler
    if file_path:
        file_handler = FileHandler(file_path, formatter=JSONFormatter())
        logger.add_handler(file_handler)
    
    # Set up context
    if run_id:
        with_context(run_id=run_id)
    
    return logger
