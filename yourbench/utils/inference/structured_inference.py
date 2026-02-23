"""Structured inference engine using instructor + AsyncOpenAI.

Enforces Pydantic-validated structured output from LLMs, replacing fragile
regex/bracket-matching parsing. Works with any OpenAI-compatible provider
(OpenAI, Groq, etc.) via base_url.
"""

import time
import uuid
import asyncio
from typing import Any, Dict, List, TypeVar

import instructor
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from yourbench.utils.logging_context import log_step
from yourbench.utils.inference.inference_core import Model, InferenceCall, _load_models
from yourbench.utils.inference.inference_tracking import (
    InferenceMetrics,
    _count_tokens,
    _get_encoding,
    _count_message_tokens,
    log_inference_metrics,
    get_performance_summary,
    update_aggregate_metrics,
)


T = TypeVar("T", bound=BaseModel)

GLOBAL_TIMEOUT = 300


async def _get_structured_response(
    model: Model,
    inference_call: InferenceCall,
    response_model: type[T],
    request_id: str | None = None,
    concurrency_level: int = 1,
    queue_start_time: float | None = None,
) -> tuple[T, InferenceMetrics]:
    """Send one inference call and parse the response into a Pydantic model via instructor."""
    start_time = time.time()
    request_id = request_id or str(uuid.uuid4())
    queue_time = (start_time - queue_start_time) if queue_start_time else 0.0

    encoding = _get_encoding(model.encoding_name)
    input_tokens = _count_message_tokens(inference_call.messages, encoding)
    stage = ";".join(inference_call.tags) if inference_call.tags else "unknown"

    metrics = InferenceMetrics(
        request_id=request_id,
        model_name=model.model_name,
        stage=stage,
        input_tokens=input_tokens,
        output_tokens=0,
        duration=0.0,
        queue_time=queue_time,
        retry_count=0,
        success=False,
        concurrency_level=concurrency_level,
        temperature=inference_call.temperature,
        encoding_name=model.encoding_name,
    )

    try:
        raw_client = AsyncOpenAI(
            base_url=model.base_url,
            api_key=model.api_key,
            timeout=GLOBAL_TIMEOUT,
        )
        client = instructor.from_openai(raw_client)

        chat_kwargs: Dict[str, Any] = {
            "model": model.model_name,
            "messages": inference_call.messages,
            "response_model": response_model,
        }
        if inference_call.temperature is not None:
            chat_kwargs["temperature"] = inference_call.temperature

        result = await client.chat.completions.create(**chat_kwargs)

        finish_time = time.time()
        # Estimate output tokens from the serialized response
        output_text = result.model_dump_json() if isinstance(result, BaseModel) else str(result)
        metrics.output_tokens = _count_tokens(output_text, encoding)
        metrics.duration = finish_time - start_time
        metrics.success = True

        logger.debug(
            "Structured response OK: model='{}' request_id='{}' duration={:.2f}s tokens={}/{}",
            model.model_name,
            request_id,
            metrics.duration,
            metrics.input_tokens,
            metrics.output_tokens,
        )

        return result, metrics

    except Exception as e:
        finish_time = time.time()
        metrics.duration = finish_time - start_time
        metrics.success = False
        metrics.error_message = str(e)[:500]

        logger.warning(
            "Structured response FAILED: model='{}' request_id='{}' duration={:.2f}s: {}",
            model.model_name,
            request_id,
            metrics.duration,
            str(e)[:100],
        )
        raise
    finally:
        log_inference_metrics(metrics)
        update_aggregate_metrics(
            model.model_name,
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.duration,
            metrics.success,
            metrics.queue_time,
            metrics.retry_count,
            error=Exception(metrics.error_message) if metrics.error_message else None,
            concurrency_level=concurrency_level,
        )


async def _retry_structured_with_backoff(
    model: Model,
    inference_call: InferenceCall,
    response_model: type[T],
    semaphore: asyncio.Semaphore,
    concurrency_level: int,
) -> T | None:
    """Attempt structured inference with exponential backoff. Returns None on total failure."""
    queue_start_time = time.time()
    request_id = str(uuid.uuid4())

    for attempt in range(inference_call.max_retries):
        async with semaphore:
            try:
                result, metrics = await _get_structured_response(
                    model, inference_call, response_model, request_id, concurrency_level, queue_start_time
                )
                metrics.retry_count = attempt
                return result
            except Exception as e:
                logger.warning(
                    "Structured attempt {} failed for model '{}': {}",
                    attempt + 1,
                    model.model_name,
                    str(e)[:100],
                )

        if attempt < inference_call.max_retries - 1:
            backoff_secs = 2 ** (attempt + 2)
            await asyncio.sleep(backoff_secs)

    logger.critical(
        "Structured inference FAILED after {} attempts for model '{}'",
        inference_call.max_retries,
        model.model_name,
    )
    return None


async def _run_structured_inference_async(
    models: List[Model],
    inference_calls: List[InferenceCall],
    response_model: type[T],
) -> Dict[str, List[T | None]]:
    """Launch structured inference tasks for all (model, call) pairs in parallel."""
    logger.info("Starting structured async inference with instructor.")

    model_semaphores: Dict[str, asyncio.Semaphore] = {}
    for model in models:
        concurrency = max(model.max_concurrent_requests, 1)
        model_semaphores[model.model_name] = asyncio.Semaphore(concurrency)

    tasks = []
    for model in models:
        semaphore = model_semaphores[model.model_name]
        for call in inference_calls:
            task = _retry_structured_with_backoff(
                model, call, response_model, semaphore, model.max_concurrent_requests
            )
            tasks.append(task)

    logger.info(
        "Structured inference: {} tasks (models={} x calls={})",
        len(tasks),
        len(models),
        len(inference_calls),
    )

    results = await tqdm_asyncio.gather(*tasks, desc="Running structured inference")

    # Re-map results to {model_name: [responses]}
    responses: Dict[str, List[T | None]] = {}
    idx = 0
    n_calls = len(inference_calls)
    for model in models:
        responses[model.model_name] = list(results[idx : idx + n_calls])
        idx += n_calls

    for model in models:
        successful = len([r for r in responses[model.model_name] if r is not None])
        logger.info(
            "Structured inference: model='{}' {}/{} successful",
            model.model_name,
            successful,
            len(responses[model.model_name]),
        )

    return responses


def run_structured_inference(
    config,
    step_name: str,
    inference_calls: List[InferenceCall],
    response_model: type[T],
) -> Dict[str, List[T | None]]:
    """Run structured inference for the given step, returning validated Pydantic objects.

    Returns:
        Dict mapping model names to lists of Pydantic model instances (or None for failures).
    """
    with log_step(f"structured_inference_{step_name}", num_calls=len(inference_calls)):
        logger.info(f"Starting structured inference for step '{step_name}' with {len(inference_calls)} calls")

        models = _load_models(config, step_name)
        if not models:
            logger.warning("No models found for step '{}'. Returning empty.", step_name)
            return {}

        # Verify all models have base_url (required for AsyncOpenAI)
        valid_models = []
        for m in models:
            if m.base_url:
                valid_models.append(m)
            else:
                logger.warning(
                    "Model '{}' has no base_url, skipping for structured inference "
                    "(structured inference requires an OpenAI-compatible endpoint).",
                    m.model_name,
                )
        if not valid_models:
            logger.error("No models with base_url available for structured inference.")
            return {}

        for call in inference_calls:
            if step_name not in call.tags:
                call.tags.append(step_name)

        try:
            result = asyncio.run(_run_structured_inference_async(valid_models, inference_calls, response_model))

            # Log performance summaries
            for model in valid_models:
                summary = get_performance_summary(model.model_name)
                if summary:
                    logger.info(
                        "Structured performance for {}: calls={}, tokens_in={}, tokens_out={}",
                        model.model_name,
                        summary.get("total_calls", 0),
                        summary.get("total_input_tokens", 0),
                        summary.get("total_output_tokens", 0),
                    )

            return result

        except Exception as e:
            logger.critical("Error running structured inference for step '{}': {}", step_name, e)
            return {}
