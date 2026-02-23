from __future__ import annotations
import re
from typing import Any

from loguru import logger
from pydantic import BaseModel

from datasets import Dataset
from yourbench.utils.schema_loader import load_schema_from_spec
from yourbench.utils.chunking_utils import get_sampling_cfg
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import (
    shuffle_mcq,
    normalize_open_ended,
    _has_difficulty_field,
    _extract_custom_fields,
    _normalize_pair_fields,
    normalize_multi_choice,
    parse_multi_hop_responses,
    parse_single_hop_responses,
    _remove_duplicate_questions,
)
from yourbench.utils.prompt_builder import build_system_prompt
from yourbench.utils.logging_context import log_step, log_stage
from yourbench.utils.question_models import QuestionRow, validate_list, force_int_in_range
from yourbench.utils.question_schemas import get_question_list_model
from yourbench.utils.cross_document_utils import create_cross_document_dataset
from yourbench.utils.inference.inference_core import _load_models, run_inference
from yourbench.utils.inference.inference_builders import (
    build_multi_hop_inference_calls,
    build_single_hop_inference_calls,
)
from yourbench.utils.inference.structured_inference import run_structured_inference


# Placeholders that instructor replaces with its own schema instructions
_SCHEMA_PLACEHOLDERS = re.compile(r"\{(?:schema_definition|example_output|critical_reminders)\}")


def _get_system_prompt(stage_cfg: Any, mode: str, is_multi: bool = False) -> str:
    """Get system prompt, substituting schema placeholders if custom schema is specified."""
    prefix = "multi_hop_" if is_multi else "single_hop_"
    suffix = "_multi" if mode == "multi-choice" else ""
    template = getattr(stage_cfg, f"{prefix}system_prompt{suffix}")

    schema_spec = getattr(stage_cfg, "question_schema", None)
    if not schema_spec:
        return template

    schema_class = load_schema_from_spec(schema_spec, mode)
    return build_system_prompt(template, schema_class)


def _get_system_prompt_for_structured(stage_cfg: Any, mode: str, is_multi: bool = False) -> str:
    """Get system prompt for structured inference — strip schema/example/reminder placeholders.

    When using instructor, the schema is enforced via the API's response_model,
    so we remove the template placeholders that would otherwise inject schema
    instructions into the prompt.
    """
    prefix = "multi_hop_" if is_multi else "single_hop_"
    suffix = "_multi" if mode == "multi-choice" else ""
    template = getattr(stage_cfg, f"{prefix}system_prompt{suffix}")

    # Remove schema placeholders — instructor handles the schema
    cleaned = _SCHEMA_PLACEHOLDERS.sub("", template)
    # Collapse runs of blank lines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _validate_mode(mode: str) -> str:
    """Ensure question mode is valid."""
    mode = (mode or "open-ended").strip().lower()
    if mode not in {"open-ended", "multi-choice"}:
        logger.warning(f"Invalid question_mode '{mode}', defaulting to 'open-ended'")
        return "open-ended"
    return mode


def _get_mode_from_config(stage_cfg: Any) -> str:
    """Extract and validate question_mode from stage config."""
    raw_mode = getattr(stage_cfg, "question_mode", None) or ""
    return _validate_mode(raw_mode)


def _can_use_structured_inference(config, step_name: str) -> bool:
    """Check if structured inference is available for this step.

    Structured inference requires at least one model with a base_url
    (i.e. an OpenAI-compatible endpoint).
    """
    models = _load_models(config, step_name)
    return any(m.base_url for m in models)


# ---------------------------------------------------------------------------
# Legacy (text-parsing) path
# ---------------------------------------------------------------------------


def _build_and_run_inference(
    dataset: Dataset, system_msg: dict, stage_cfg: Any, builder_func: callable, step_name: str, config
) -> tuple[dict, list]:
    """Common pattern: build calls, run inference, return responses + index map."""
    sampling_cfg = (
        get_sampling_cfg(stage_cfg) if hasattr(builder_func, "__name__") and "single" in builder_func.__name__ else {}
    )

    calls, index_map = (
        builder_func(dataset, system_msg, stage_cfg, sampling_cfg)
        if sampling_cfg
        else builder_func(dataset, system_msg, stage_cfg)
    )

    if not calls:
        logger.warning(f"No valid inference calls for {step_name}")
        return {}, []

    responses = run_inference(config=config, step_name=step_name, inference_calls=calls)
    return responses, index_map


# ---------------------------------------------------------------------------
# Structured (instructor) path
# ---------------------------------------------------------------------------


def _get_response_model(stage_cfg: Any, mode: str) -> type[BaseModel]:
    """Return the Pydantic wrapper model for structured inference.

    Returns a ``QuestionListWrapper`` whose ``questions`` field is
    ``list[SchemaClass]`` — either the default or custom schema.
    """
    schema_spec = getattr(stage_cfg, "question_schema", None)
    custom_schema = load_schema_from_spec(schema_spec, mode) if schema_spec else None
    return get_question_list_model(mode, custom_schema)


def _build_and_run_structured_inference(
    dataset: Dataset, system_msg: dict, stage_cfg: Any, builder_func: callable, step_name: str, config, mode: str
) -> tuple[dict, list]:
    """Build calls and run structured inference, returning validated Pydantic lists."""
    sampling_cfg = (
        get_sampling_cfg(stage_cfg) if hasattr(builder_func, "__name__") and "single" in builder_func.__name__ else {}
    )

    calls, index_map = (
        builder_func(dataset, system_msg, stage_cfg, sampling_cfg)
        if sampling_cfg
        else builder_func(dataset, system_msg, stage_cfg)
    )

    if not calls:
        logger.warning(f"No valid inference calls for {step_name}")
        return {}, []

    response_model = _get_response_model(stage_cfg, mode)
    responses = run_structured_inference(
        config=config, step_name=step_name, inference_calls=calls, response_model=response_model
    )
    return responses, index_map


def _pydantic_to_pair(q: BaseModel, question_mode: str) -> dict:
    """Convert a single Pydantic question model to the dict format expected by the parsing pipeline."""
    pair = q.model_dump()
    pair["question_mode"] = question_mode
    return pair


def _structured_to_rows_single_hop(responses: dict, index_map: list, stage_cfg: Any, mode: str) -> list[dict]:
    """Convert structured inference responses to row dicts for single-hop questions."""
    rows = []

    for model, wrappers in responses.items():
        if len(wrappers) != len(index_map):
            logger.error(f"Mismatch: model '{model}' responses={len(wrappers)}, expected={len(index_map)}")
            continue

        for i, wrapper in enumerate(wrappers):
            if wrapper is None:
                logger.warning(f"Structured response at index {i} was None (model={model})")
                continue

            questions = wrapper.questions if hasattr(wrapper, "questions") else []
            if not questions:
                logger.warning(f"No questions in structured response at index {i}")
                continue

            for q in questions:
                try:
                    pair = _pydantic_to_pair(q, mode)
                    pair = shuffle_mcq(pair)
                    pair = _normalize_pair_fields(pair)

                    if mode == "open-ended":
                        pair = normalize_open_ended(pair)
                        if pair is None:
                            continue
                        choices = []
                    elif mode == "multi-choice":
                        pair = normalize_multi_choice(pair)
                        if pair is None:
                            continue
                        choices = pair["choices"]
                    else:
                        continue

                    citations = validate_list(pair.get("citations", []))
                    raw_json = q.model_dump_json()

                    base_row = QuestionRow(
                        chunk_id=index_map[i][2],
                        source_chunk_ids=None,
                        document_id=index_map[i][1],
                        additional_instructions=stage_cfg.additional_instructions,
                        question=str(pair.get("question", "")).strip(),
                        self_answer=str(pair.get("answer", "")).strip(),
                        choices=choices,
                        estimated_difficulty=force_int_in_range(pair.get("estimated_difficulty", 5), 1, 10),
                        self_assessed_question_type=str(pair.get("question_type", "")).strip(),
                        question_mode=mode,
                        generating_model=model,
                        thought_process=str(pair.get("thought_process", "")),
                        raw_response=raw_json,
                        citations=citations,
                    ).to_dict(format="single-hop")

                    if not _has_difficulty_field(pair):
                        base_row.pop("estimated_difficulty", None)

                    custom_fields = _extract_custom_fields(pair)
                    if custom_fields:
                        base_row.update(custom_fields)
                    rows.append(base_row)

                except Exception as e:
                    logger.error(f"Error converting structured question at index {i}: {e}")
                    continue

    return rows


def _structured_to_rows_multi_hop(responses: dict, index_map: list, stage_cfg: Any, mode: str) -> list[dict]:
    """Convert structured inference responses to row dicts for multi-hop questions."""
    rows = []

    for model, wrappers in responses.items():
        for i, wrapper in enumerate(wrappers):
            if wrapper is None:
                continue

            questions = wrapper.questions if hasattr(wrapper, "questions") else []
            for q in questions:
                try:
                    pair = _pydantic_to_pair(q, mode)
                    pair = shuffle_mcq(pair)
                    pair = _normalize_pair_fields(pair)

                    if mode == "open-ended":
                        pair = normalize_open_ended(pair)
                        if pair is None:
                            continue
                        choices = []
                    elif mode == "multi-choice":
                        pair = normalize_multi_choice(pair)
                        if pair is None:
                            continue
                        choices = pair["choices"]
                    else:
                        continue

                    citations = validate_list(pair.get("citations", []))
                    raw_json = q.model_dump_json()

                    base_row = QuestionRow(
                        chunk_id=None,
                        source_chunk_ids=index_map[i][2],
                        document_id=index_map[i][1],
                        additional_instructions=stage_cfg.additional_instructions,
                        question=str(pair.get("question", "")).strip(),
                        self_answer=str(pair.get("answer", "")).strip(),
                        choices=choices,
                        estimated_difficulty=force_int_in_range(pair.get("estimated_difficulty", 5), 1, 10),
                        self_assessed_question_type=str(pair.get("question_type", "")).strip(),
                        question_mode=mode,
                        generating_model=model,
                        thought_process=str(pair.get("thought_process", "")),
                        raw_response=raw_json,
                        citations=citations,
                    ).to_dict(format="multi-hop")

                    if not _has_difficulty_field(pair):
                        base_row.pop("estimated_difficulty", None)

                    custom_fields = _extract_custom_fields(pair)
                    if custom_fields:
                        base_row.update(custom_fields)
                    rows.append(base_row)

                except Exception as e:
                    logger.warning(f"Error converting structured multi-hop question for doc {index_map[i][1]}: {e}")
                    continue

    return rows


# ---------------------------------------------------------------------------
# Column normalization & saving
# ---------------------------------------------------------------------------


def _normalize_column_types(rows: list[dict]) -> list[dict]:
    """Normalize column types across all rows for PyArrow compatibility.

    Detects columns with mixed types across rows and coerces to strings.
    This prevents PyArrow schema inference failures.
    """
    if not rows:
        return rows

    # Collect all unique keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    # For each key, check if types are consistent across rows
    for key in all_keys:
        values = [row.get(key) for row in rows]
        non_none_values = [v for v in values if v is not None]

        if not non_none_values:
            continue

        # Check for type inconsistency
        types = set()
        for v in non_none_values:
            if isinstance(v, list):
                types.add("list")
            elif isinstance(v, str):
                types.add("str")
            elif isinstance(v, (int, float)):
                types.add("num")
            else:
                types.add(type(v).__name__)

        # If mixed types, coerce all to strings
        if len(types) > 1:
            logger.debug(f"Column '{key}' has mixed types {types}, coercing to strings")
            for row in rows:
                if key in row and row[key] is not None:
                    v = row[key]
                    if isinstance(v, list):
                        row[key] = [str(x) if x is not None else None for x in v]
                    else:
                        row[key] = str(v)

        # Also check list element types within lists
        elif "list" in types:
            # Check if list elements have consistent types across rows
            elem_types = set()
            for v in non_none_values:
                if isinstance(v, list):
                    for elem in v:
                        if elem is not None:
                            elem_types.add(type(elem).__name__)

            if len(elem_types) > 1:
                logger.debug(f"Column '{key}' list elements have mixed types {elem_types}, coercing to strings")
                for row in rows:
                    if key in row and isinstance(row[key], list):
                        row[key] = [str(x) if x is not None else None for x in row[key]]

    return rows


def _save_questions(rows: list[dict], config, subset: str) -> None:
    """Save question rows after deduplication."""
    if not (clean_rows := _remove_duplicate_questions(rows)):
        return

    # Normalize column types across all rows before creating dataset
    clean_rows = _normalize_column_types(clean_rows)

    logger.info(f"Saving {len(clean_rows)} {subset}")
    custom_save_dataset(
        Dataset.from_list(clean_rows), config=config, subset=subset, push_to_hub=config.hf_configuration.push_to_hub
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_single_hop(config) -> None:
    """Generate single-hop questions from individual chunks."""
    with log_stage("single_hop_generation"):
        if not (stage_cfg := config.pipeline.single_hop_question_generation).run:
            logger.info("single_hop_question_generation disabled")
            return

        mode = _get_mode_from_config(stage_cfg)
        step_name = "single_hop_question_generation"
        use_structured = _can_use_structured_inference(config, step_name)

        logger.info(f"Single-hop mode: {mode}, structured={use_structured}")

        if use_structured:
            system_msg = {"role": "system", "content": _get_system_prompt_for_structured(stage_cfg, mode)}
        else:
            system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode)}

        with log_step("loading_dataset"):
            dataset = custom_load_dataset(config=config, subset="chunked")
            logger.debug(f"Loaded {len(dataset) if dataset else 0} documents")

        if use_structured:
            with log_step("generating_questions_structured"):
                responses, index_map = _build_and_run_structured_inference(
                    dataset, system_msg, stage_cfg, build_single_hop_inference_calls, step_name, config, mode
                )

            with log_step("saving_questions"):
                if rows := _structured_to_rows_single_hop(responses, index_map, stage_cfg, mode):
                    _save_questions(rows, config, "single_hop_questions")
                    logger.info(f"Saved {len(rows)} single-hop questions (structured)")
                else:
                    logger.warning("No valid questions from structured inference")
        else:
            with log_step("generating_questions"):
                responses, index_map = _build_and_run_inference(
                    dataset, system_msg, stage_cfg, build_single_hop_inference_calls, step_name, config
                )

            with log_step("saving_questions"):
                if rows := parse_single_hop_responses(responses, index_map, stage_cfg):
                    _save_questions(rows, config, "single_hop_questions")
                    logger.info(f"Saved {len(rows)} single-hop questions (legacy)")


def run_multi_hop(config) -> None:
    """Generate multi-hop questions."""
    stage_cfg = config.pipeline.multi_hop_question_generation
    if not stage_cfg.run:
        logger.info("Multi-hop question generation disabled")
        return

    mode = _get_mode_from_config(stage_cfg)
    step_name = "multi_hop_question_generation"
    use_structured = _can_use_structured_inference(config, step_name)

    if use_structured:
        system_msg = {"role": "system", "content": _get_system_prompt_for_structured(stage_cfg, mode, is_multi=True)}
    else:
        system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode, is_multi=True)}

    chunked_ds = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(chunked_ds)} documents for multi-hop (structured={use_structured})")

    _process_questions(
        chunked_ds, "multi_hop_questions", system_msg, stage_cfg, config, step_name, mode, use_structured
    )


def run_cross_document(config) -> None:
    """Generate cross-document questions."""
    stage_cfg = config.pipeline.cross_document_question_generation
    if not stage_cfg.run:
        logger.info("Cross-document question generation disabled")
        return

    mode = _get_mode_from_config(stage_cfg)
    step_name = "cross_document_question_generation"
    use_structured = _can_use_structured_inference(config, step_name)

    if use_structured:
        system_msg = {"role": "system", "content": _get_system_prompt_for_structured(stage_cfg, mode, is_multi=True)}
    else:
        system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode, is_multi=True)}

    chunked_ds = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(chunked_ds)} documents for cross-document (structured={use_structured})")

    cross_cfg = {
        "enable": True,
        "max_combinations": stage_cfg.max_combinations,
        "chunks_per_document": stage_cfg.chunks_per_document,
        "num_docs_per_combination": stage_cfg.num_docs_per_combination,
        "random_seed": stage_cfg.random_seed,
    }

    logger.info("Starting cross-document generation")
    if cross_ds := create_cross_document_dataset(chunked_ds, cross_cfg):
        logger.info(f"Generated {len(cross_ds)} cross-document combinations")
        _process_questions(
            cross_ds, "cross_document_questions", system_msg, stage_cfg, config, step_name, mode, use_structured
        )


def _process_questions(
    dataset: Dataset,
    label: str,
    system_msg: dict,
    stage_cfg: Any,
    config,
    step_name: str,
    mode: str | None = None,
    use_structured: bool = False,
) -> None:
    """Process and save a set of questions (multi-hop or cross-document)."""
    if not dataset or len(dataset) == 0:
        logger.warning(f"No valid {label} dataset")
        return

    if mode is None:
        mode = _get_mode_from_config(stage_cfg)

    if use_structured:
        responses, index_map = _build_and_run_structured_inference(
            dataset, system_msg, stage_cfg, build_multi_hop_inference_calls, step_name, config, mode
        )
        if rows := _structured_to_rows_multi_hop(responses, index_map, stage_cfg, mode):
            _save_questions(rows, config, label)
        else:
            logger.warning(f"No valid questions from structured inference for {label}")
    else:
        responses, index_map = _build_and_run_inference(
            dataset, system_msg, stage_cfg, build_multi_hop_inference_calls, step_name, config
        )
        if rows := parse_multi_hop_responses(responses, index_map, stage_cfg):
            _save_questions(rows, config, label)
        else:
            logger.warning(f"No valid questions parsed for {label} (check model output format)")
