"""
Metrics collection and aggregation for YourBench logging.

Provides thread-safe metrics tracking without global state.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import threading
from collections import defaultdict

from .types import MetricsDict, StageMetrics
from .context import get_context


@dataclass
class StageMetricsSummary:
    """Summary of metrics for a pipeline stage."""
    stage_name: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_duration_seconds: float = 0.0
    avg_duration_seconds: float = 0.0
    min_duration_seconds: Optional[float] = None
    max_duration_seconds: Optional[float] = None
    total_items_processed: int = 0
    total_items_failed: int = 0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    total_cost_usd: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MetricsCollector:
    """
    Thread-safe metrics collector with context isolation.
    
    This class avoids global state by using instance variables and locks.
    Each pipeline run can have its own MetricsCollector instance.
    """
    
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or get_context().run_id
        self._metrics: Dict[str, List[MetricsDict]] = defaultdict(list)
        self._lock = threading.Lock()
        self.start_time = datetime.utcnow()
    
    def add_stage_metrics(
        self,
        stage_name: str,
        metrics: MetricsDict,
        success: bool = True
    ) -> None:
        """
        Add metrics for a stage execution.
        
        Args:
            stage_name: Name of the stage
            metrics: Metrics dictionary
            success: Whether the stage succeeded
        """
        with self._lock:
            # Add success indicator
            metrics_copy = dict(metrics)
            metrics_copy['success'] = success
            metrics_copy['timestamp'] = datetime.utcnow().isoformat()
            
            self._metrics[stage_name].append(metrics_copy)
    
    def get_stage_summary(self, stage_name: str) -> StageMetricsSummary:
        """
        Get summary statistics for a stage.
        
        Args:
            stage_name: Name of the stage
        
        Returns:
            Summary of all metrics for the stage
        """
        with self._lock:
            stage_metrics = self._metrics.get(stage_name, [])
            
            if not stage_metrics:
                return StageMetricsSummary(stage_name=stage_name)
            
            summary = StageMetricsSummary(stage_name=stage_name)
            summary.total_runs = len(stage_metrics)
            
            durations = []
            
            for metric in stage_metrics:
                # Count success/failure
                if metric.get('success', True):
                    summary.successful_runs += 1
                else:
                    summary.failed_runs += 1
                
                # Aggregate durations
                if 'duration_seconds' in metric:
                    duration = metric['duration_seconds']
                    durations.append(duration)
                    summary.total_duration_seconds += duration
                
                # Aggregate items
                summary.total_items_processed += metric.get('items_processed', 0)
                summary.total_items_failed += metric.get('items_failed', 0)
                
                # Aggregate API usage
                summary.total_tokens_used += metric.get('tokens_used', 0)
                summary.total_api_calls += metric.get('api_calls', 0)
                summary.total_cost_usd += metric.get('cost_usd', 0.0)
                
                # Collect warnings
                if 'warnings' in metric:
                    summary.warnings.extend(metric['warnings'])
            
            # Calculate duration stats
            if durations:
                summary.avg_duration_seconds = summary.total_duration_seconds / len(durations)
                summary.min_duration_seconds = min(durations)
                summary.max_duration_seconds = max(durations)
            
            return summary
    
    def get_all_summaries(self) -> Dict[str, StageMetricsSummary]:
        """
        Get summaries for all stages.
        
        Returns:
            Dictionary mapping stage names to summaries
        """
        with self._lock:
            return {
                stage_name: self.get_stage_summary(stage_name)
                for stage_name in self._metrics.keys()
            }
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get overall pipeline summary.
        
        Returns:
            Dictionary with pipeline-level metrics
        """
        summaries = self.get_all_summaries()
        
        total_duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(),
            'total_duration_seconds': total_duration,
            'stages_run': len(summaries),
            'stages_succeeded': sum(1 for s in summaries.values() if s.failed_runs == 0),
            'stages_failed': sum(1 for s in summaries.values() if s.failed_runs > 0),
            'total_items_processed': sum(s.total_items_processed for s in summaries.values()),
            'total_items_failed': sum(s.total_items_failed for s in summaries.values()),
            'total_tokens_used': sum(s.total_tokens_used for s in summaries.values()),
            'total_api_calls': sum(s.total_api_calls for s in summaries.values()),
            'total_cost_usd': sum(s.total_cost_usd for s in summaries.values()),
            'stage_summaries': {
                name: {
                    'total_runs': s.total_runs,
                    'successful_runs': s.successful_runs,
                    'failed_runs': s.failed_runs,
                    'avg_duration_seconds': s.avg_duration_seconds,
                    'total_items_processed': s.total_items_processed,
                    'total_cost_usd': s.total_cost_usd
                }
                for name, s in summaries.items()
            }
        }
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self.start_time = datetime.utcnow()


# Context-local metrics collector
_collector_context: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get or create the current metrics collector."""
    global _collector_context
    
    with _collector_lock:
        if _collector_context is None:
            _collector_context = MetricsCollector()
        return _collector_context


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set the current metrics collector."""
    global _collector_context
    
    with _collector_lock:
        _collector_context = collector


def reset_metrics_collector() -> None:
    """Reset the metrics collector."""
    global _collector_context
    
    with _collector_lock:
        _collector_context = None


def collect_stage_metrics(
    stage_name: str,
    metrics: MetricsDict,
    success: bool = True
) -> None:
    """
    Convenience function to collect metrics for current context.
    
    Args:
        stage_name: Name of the stage
        metrics: Metrics to collect
        success: Whether the stage succeeded
    """
    collector = get_metrics_collector()
    collector.add_stage_metrics(stage_name, metrics, success)
