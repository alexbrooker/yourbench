"""
Log formatters for console and file output.

Provides clean, readable console output and structured JSON for file logging.
"""
import json
from datetime import datetime
from typing import Optional, Any, Dict
import sys

from .types import LogRecord, LogLevel, StageStatus
from .context import get_context


class ConsoleFormatter:
    """Format logs for console output with clean, readable format."""
    
    # Status indicators (no emojis, just clean text)
    STATUS_INDICATORS = {
        LogLevel.TRACE: "[TRACE]",
        LogLevel.DEBUG: "[DEBUG]",
        LogLevel.INFO: "[INFO ]",
        LogLevel.WARNING: "[WARN ]",
        LogLevel.ERROR: "[ERROR]",
        LogLevel.CRITICAL: "[CRIT ]",
        LogLevel.SUCCESS: "[ OK  ]",
    }
    
    STAGE_STATUS_INDICATORS = {
        StageStatus.NOT_STARTED: "[....]",
        StageStatus.STARTING: "[START]",
        StageStatus.IN_PROGRESS: "[>...]",
        StageStatus.COMPLETED: "[DONE ]",
        StageStatus.FAILED: "[FAIL ]",
        StageStatus.SKIPPED: "[SKIP ]",
    }
    
    def __init__(self, show_timestamp: bool = True, show_stage: bool = True):
        self.show_timestamp = show_timestamp
        self.show_stage = show_stage
        # Track maximum stage name length for dynamic padding
        self.max_stage_len = 20
    
    def format(self, record: LogRecord) -> str:
        """Format a log record for console output."""
        parts = []
        
        # Timestamp
        if self.show_timestamp:
            timestamp = record.get('timestamp', datetime.utcnow().isoformat())
            # Format as HH:MM:SS for readability
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
            parts.append(f"[{time_str}]")
        
        # Level indicator
        level = record.get('level', 'INFO')
        indicator = self.STATUS_INDICATORS.get(LogLevel(level), "[?????]")
        parts.append(indicator)
        
        # Stage info
        if self.show_stage and record.get('stage'):
            stage = record['stage']
            # Update max length for better formatting
            self.max_stage_len = max(self.max_stage_len, len(stage))
            # Dynamic padding
            stage_str = f"{stage:<{min(self.max_stage_len, 25)}}"
            
            # Add stage status if available
            context = get_context()
            if context and context.stage == stage:
                status_indicator = self.STAGE_STATUS_INDICATORS.get(
                    context.stage_status,
                    "[?????]"
                )
                parts.append(f"{status_indicator} {stage_str}")
            else:
                parts.append(f"       {stage_str}")
        
        # Main message
        message = record.get('message', '')
        parts.append(message)
        
        # Add metrics if present
        metrics = record.get('metrics', {})
        if metrics:
            metric_strs = []
            if 'duration_seconds' in metrics:
                metric_strs.append(f"duration={metrics['duration_seconds']:.2f}s")
            if 'items_processed' in metrics:
                metric_strs.append(f"items={metrics['items_processed']}")
            if 'tokens_used' in metrics:
                metric_strs.append(f"tokens={metrics['tokens_used']}")
            if 'cost_usd' in metrics:
                metric_strs.append(f"cost=${metrics['cost_usd']:.4f}")
            if metric_strs:
                parts.append(f"({', '.join(metric_strs)})")
        
        # Add error info if present
        error = record.get('error')
        if error:
            parts.append(f"ERROR: {error.get('type', 'Unknown')}: {error.get('message', 'No details')}")
        
        return ' '.join(parts)


class JSONFormatter:
    """Format logs as JSON for structured file output."""
    
    def __init__(self, pretty: bool = False, sanitize: bool = True):
        self.pretty = pretty
        self.sanitize = sanitize
    
    def format(self, record: LogRecord) -> str:
        """Format a log record as JSON."""
        # Create a copy to avoid modifying the original
        output = dict(record)
        
        # Ensure timestamp is present
        if 'timestamp' not in output:
            output['timestamp'] = datetime.utcnow().isoformat()
        
        # Add context info if available
        context = get_context()
        if context:
            if not output.get('run_id'):
                output['run_id'] = context.run_id
            if not output.get('correlation_id') and context.correlation_id:
                output['correlation_id'] = context.correlation_id
            if context.progress and not output.get('progress'):
                output['progress'] = context.progress
        
        # Sanitize if requested
        if self.sanitize:
            output = self._sanitize_value(output)
        
        # Convert to JSON
        if self.pretty:
            return json.dumps(output, indent=2, default=str, ensure_ascii=False)
        else:
            return json.dumps(output, default=str, ensure_ascii=False)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize values for JSON serialization."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_value(v) for v in value]
        elif hasattr(value, '__dict__'):
            return self._sanitize_value(value.__dict__)
        else:
            return str(value)


class MinimalFormatter:
    """Minimal formatter for testing and debugging."""
    
    def format(self, record: LogRecord) -> str:
        """Format with just level and message."""
        level = record.get('level', 'INFO')
        message = record.get('message', '')
        return f"{level}: {message}"
