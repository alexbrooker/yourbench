import os
import csv
import atexit
import datetime
import collections
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import tiktoken
from loguru import logger


# Enhanced tracking data structures
_cost_data = collections.defaultdict(lambda: {
    "input_tokens": 0, 
    "output_tokens": 0, 
    "calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "total_duration": 0.0,
    "total_queue_time": 0.0,
    "retry_attempts": 0,
    "timeouts": 0,
    "error_categories": collections.defaultdict(int)
})

_performance_data = collections.defaultdict(lambda: {
    "request_sizes": [],
    "response_sizes": [],
    "durations": [],
    "queue_times": [],
    "concurrency_levels": [],
    "retry_counts": []
})

_individual_log_file = os.path.join("logs", "inference_cost_log_individual.csv")
_aggregate_log_file = os.path.join("logs", "inference_cost_log_aggregate.csv")
_performance_log_file = os.path.join("logs", "inference_performance_log.csv")
_individual_header_written = False
_performance_header_written = False

# Thread-safe locks
_cost_lock = threading.Lock()
_performance_lock = threading.Lock()


@dataclass
class InferenceMetrics:
    """Enhanced metrics for a single inference call."""
    request_id: str
    model_name: str
    stage: str
    input_tokens: int
    output_tokens: int
    duration: float
    queue_time: float = 0.0
    retry_count: int = 0
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    concurrency_level: int = 1
    temperature: Optional[float] = None
    encoding_name: str = "cl100k_base"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Gets a tiktoken encoding, defaulting to cl100k_base with fallback."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding '{encoding_name}'. Falling back to 'cl100k_base'. Error: {e}")
        return tiktoken.get_encoding("cl100k_base")


def _ensure_logs_dir():
    """Ensures the logs directory exists."""
    os.makedirs("logs", exist_ok=True)


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Counts tokens in a single string."""
    if not text:
        return 0
    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0


def _count_message_tokens(messages: List[Dict[str, str]], encoding: tiktoken.Encoding) -> int:
    """Counts tokens in a list of messages, approximating OpenAI's format."""
    num_tokens = 0
    # Approximation based on OpenAI's cookbook
    tokens_per_message = 3
    tokens_per_name = 1

    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value:
                num_tokens += _count_tokens(str(value), encoding)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def _categorize_error(error: Exception) -> str:
    """Categorize errors for better tracking."""
    error_str = str(error).lower()
    if "timeout" in error_str:
        return "timeout"
    elif "rate limit" in error_str or "429" in error_str:
        return "rate_limit"
    elif "connection" in error_str or "network" in error_str:
        return "network"
    elif "authentication" in error_str or "401" in error_str:
        return "auth"
    elif "server" in error_str or "500" in error_str:
        return "server_error"
    elif "invalid" in error_str or "400" in error_str:
        return "invalid_request"
    else:
        return "other"


def _log_individual_call(model_name: str, input_tokens: int, output_tokens: int, tags: List[str], encoding_name: str):
    """Legacy function for backward compatibility."""
    metrics = InferenceMetrics(
        request_id="legacy",
        model_name=model_name,
        stage=";".join(tags) if tags else "unknown",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration=0.0,
        encoding_name=encoding_name
    )
    log_inference_metrics(metrics)


def log_inference_metrics(metrics: InferenceMetrics):
    """Log comprehensive inference metrics."""
    global _individual_header_written, _performance_header_written
    
    try:
        _ensure_logs_dir()
        
        # Log individual call details
        is_new_file = not os.path.exists(_individual_log_file)
        file_has_header = False
        
        if not is_new_file:
            try:
                with open(_individual_log_file, "r", newline="", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    file_has_header = first_line.startswith("timestamp,")
            except Exception:
                file_has_header = False
        
        mode = "a" if not is_new_file else "w"
        
        with open(_individual_log_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            if is_new_file or (not file_has_header and not _individual_header_written):
                writer.writerow([
                    "timestamp", "request_id", "model_name", "stage", "input_tokens", "output_tokens", 
                    "duration", "queue_time", "retry_count", "success", "error_type", "error_message",
                    "concurrency_level", "temperature", "encoding_used"
                ])
                _individual_header_written = True
            
            writer.writerow([
                metrics.timestamp, metrics.request_id, metrics.model_name, metrics.stage,
                metrics.input_tokens, metrics.output_tokens, metrics.duration, metrics.queue_time,
                metrics.retry_count, metrics.success, metrics.error_type or "", metrics.error_message or "",
                metrics.concurrency_level, metrics.temperature, metrics.encoding_name
            ])
        
        # Log performance metrics
        is_perf_new_file = not os.path.exists(_performance_log_file)
        with open(_performance_log_file, "a" if not is_perf_new_file else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            if is_perf_new_file or not _performance_header_written:
                writer.writerow([
                    "timestamp", "model_name", "stage", "tokens_per_second", "queue_efficiency", 
                    "success_rate", "avg_retry_count", "throughput_score"
                ])
                _performance_header_written = True
            
            # Calculate derived metrics
            tokens_per_second = (metrics.input_tokens + metrics.output_tokens) / max(metrics.duration, 0.001)
            queue_efficiency = metrics.duration / max(metrics.duration + metrics.queue_time, 0.001)
            
            writer.writerow([
                metrics.timestamp, metrics.model_name, metrics.stage, tokens_per_second, 
                queue_efficiency, 1.0 if metrics.success else 0.0, metrics.retry_count,
                tokens_per_second * queue_efficiency
            ])
        
    except Exception as e:
        logger.error(f"Failed to write inference metrics: {e}")


def _update_aggregate_cost(model_name: str, input_tokens: int, output_tokens: int):
    """Legacy function for backward compatibility."""
    update_aggregate_metrics(model_name, input_tokens, output_tokens, duration=0.0, success=True)


def update_aggregate_metrics(model_name: str, input_tokens: int, output_tokens: int, 
                           duration: float, success: bool = True, queue_time: float = 0.0,
                           retry_count: int = 0, error: Optional[Exception] = None,
                           concurrency_level: int = 1):
    """Update aggregate metrics with enhanced tracking."""
    with _cost_lock:
        try:
            data = _cost_data[model_name]
            data["input_tokens"] += input_tokens
            data["output_tokens"] += output_tokens
            data["calls"] += 1
            data["total_duration"] += duration
            data["total_queue_time"] += queue_time
            data["retry_attempts"] += retry_count
            
            if success:
                data["successful_calls"] += 1
            else:
                data["failed_calls"] += 1
                if error:
                    error_category = _categorize_error(error)
                    data["error_categories"][error_category] += 1
                    if "timeout" in error_category:
                        data["timeouts"] += 1
        except Exception as e:
            logger.error(f"Failed to update aggregate metrics: {e}")
    
    with _performance_lock:
        try:
            perf_data = _performance_data[model_name]
            perf_data["request_sizes"].append(input_tokens)
            perf_data["response_sizes"].append(output_tokens)
            perf_data["durations"].append(duration)
            perf_data["queue_times"].append(queue_time)
            perf_data["concurrency_levels"].append(concurrency_level)
            perf_data["retry_counts"].append(retry_count)
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")


def get_performance_summary(model_name: str) -> Dict:
    """Get current performance summary for a model."""
    with _cost_lock, _performance_lock:
        if model_name not in _cost_data:
            return {}
        
        cost_data = _cost_data[model_name]
        perf_data = _performance_data[model_name]
        
        total_calls = cost_data["calls"]
        if total_calls == 0:
            return {}
        
        success_rate = cost_data["successful_calls"] / total_calls
        avg_duration = cost_data["total_duration"] / total_calls
        avg_queue_time = cost_data["total_queue_time"] / total_calls
        avg_retry_count = cost_data["retry_attempts"] / total_calls
        
        return {
            "model_name": model_name,
            "total_calls": total_calls,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "avg_queue_time": avg_queue_time,
            "avg_retry_count": avg_retry_count,
            "total_input_tokens": cost_data["input_tokens"],
            "total_output_tokens": cost_data["output_tokens"],
            "timeout_rate": cost_data["timeouts"] / total_calls,
            "error_breakdown": dict(cost_data["error_categories"]),
            "avg_request_size": sum(perf_data["request_sizes"]) / len(perf_data["request_sizes"]) if perf_data["request_sizes"] else 0,
            "avg_response_size": sum(perf_data["response_sizes"]) / len(perf_data["response_sizes"]) if perf_data["response_sizes"] else 0,
            "avg_concurrency": sum(perf_data["concurrency_levels"]) / len(perf_data["concurrency_levels"]) if perf_data["concurrency_levels"] else 0,
        }


def _write_aggregate_log():
    """Write enhanced aggregate log with performance metrics."""
    try:
        if not _cost_data:
            logger.info("No cost data collected, skipping aggregate log.")
            return

        _ensure_logs_dir()
        logger.info(f"Writing enhanced aggregate log to {_aggregate_log_file}")
        
        with open(_aggregate_log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name", "total_input_tokens", "total_output_tokens", "total_calls",
                "successful_calls", "failed_calls", "success_rate", "avg_duration", 
                "avg_queue_time", "avg_retry_count", "timeout_rate", "error_breakdown"
            ])
            
            for model_name in sorted(_cost_data.keys()):
                summary = get_performance_summary(model_name)
                if summary:
                    writer.writerow([
                        model_name, summary["total_input_tokens"], summary["total_output_tokens"],
                        summary["total_calls"], summary["total_calls"] - len(summary["error_breakdown"]),
                        len(summary["error_breakdown"]), summary["success_rate"], summary["avg_duration"],
                        summary["avg_queue_time"], summary["avg_retry_count"], summary["timeout_rate"],
                        str(summary["error_breakdown"])
                    ])
        
        logger.success(f"Enhanced aggregate log written to {_aggregate_log_file}")
    except Exception as e:
        print(f"ERROR: Failed to write aggregate log: {e}", flush=True)


# Register the aggregate log function to run at exit
atexit.register(_write_aggregate_log)
