from __future__ import annotations
from typing import Any

from loguru import logger

from datasets import Dataset
from yourbench.utils.schema_loader import (
    load_schema_from_file,
    create_example_from_schema,
    generate_schema_description,
)
from yourbench.utils.chunking_utils import get_sampling_cfg
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import (
    parse_multi_hop_responses,
    _remove_duplicate_questions,
    parse_single_shot_responses,
    parse_responses_with_custom_schema,
)
from yourbench.utils.logging_context import log_step, log_stage
from yourbench.utils.cross_document_utils import create_cross_document_dataset
from yourbench.utils.inference.inference_core import run_inference
from yourbench.utils.inference.inference_builders import (
    build_multi_hop_inference_calls,
    build_single_shot_inference_calls,
)


def _get_system_prompt(stage_cfg: Any, mode: str, is_multi: bool = False) -> str:
    """Get appropriate system prompt based on mode and stage type."""
    prefix = "multi_hop_" if is_multi else "single_shot_"
    suffix = "_multi" if mode == "multi-choice" else ""
    return getattr(stage_cfg, f"{prefix}system_prompt{suffix}")


def _validate_mode(mode: str) -> str:
    """Ensure question mode is valid."""
    mode = (mode or "open-ended").strip().lower()
    if mode not in {"open-ended", "multi-choice"}:
        logger.warning(f"Invalid question_mode '{mode}', defaulting to 'open-ended'")
        return "open-ended"
    return mode


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


def _save_questions(rows: list[dict], config, subset: str) -> None:
    """Save question rows after deduplication."""
    if not (clean_rows := _remove_duplicate_questions(rows)):
        return

    logger.info(f"Saving {len(clean_rows)} {subset}")
    custom_save_dataset(
        Dataset.from_list(clean_rows), config=config, subset=subset, push_to_hub=config.hf_configuration.push_to_hub
    )


def run_single_shot(config) -> None:
    """Generate single-hop questions from individual chunks."""
    with log_stage("single_shot_generation"):
        if not (stage_cfg := config.pipeline.single_shot_question_generation).run:
            logger.info("single_shot_question_generation disabled")
            return

        mode = stage_cfg.question_mode.strip().lower() if stage_cfg.question_mode else "open-ended"
        if mode not in {"open-ended", "multi-choice"}:
            logger.warning(f"Invalid question_mode '{mode}', defaulting to 'open-ended'")
            mode = "open-ended"
        logger.info(f"Single-shot mode: {mode}")

        system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode)}

        with log_step("loading_dataset"):
            dataset = custom_load_dataset(config=config, subset="chunked")
            logger.debug(f"Loaded {len(dataset) if dataset else 0} documents")

        # If using custom schema, modify the prompt to include schema info
        if hasattr(stage_cfg, "custom_schema_path") and stage_cfg.custom_schema_path:
            try:
                import json

                custom_schema = load_schema_from_file(
                    stage_cfg.custom_schema_path, stage_cfg.custom_schema_class, stage_cfg.custom_schema_auto_batch
                )
                schema_desc = generate_schema_description(custom_schema)
                schema_example = create_example_from_schema(custom_schema)

                # Update system message to include schema info
                original_system = system_msg["content"]
                
                # Get the actual field information from the schema
                schema_fields = []
                if hasattr(custom_schema, "__fields__"):
                    # Pydantic v1
                    for field_name, field_obj in custom_schema.__fields__.items():
                        if field_name != "items":  # Skip batch wrapper
                            field_type = str(field_obj.type_).replace('typing.', '')
                            desc = field_obj.field_info.description or f"The {field_name}"
                            schema_fields.append(f"- {field_name}: {desc}")
                elif hasattr(custom_schema, "model_fields"):
                    # Pydantic v2
                    for field_name, field_obj in custom_schema.model_fields.items():
                        if field_name != "items":  # Skip batch wrapper
                            desc = field_obj.description or f"The {field_name}"
                            schema_fields.append(f"- {field_name}: {desc}")
                
                # If it's a batch schema, get the inner model
                if "Batch" in custom_schema.__name__ and hasattr(custom_schema, "model_fields"):
                    items_field = custom_schema.model_fields.get("items")
                    if items_field and hasattr(items_field.annotation, "__args__"):
                        inner_model = items_field.annotation.__args__[0]
                        schema_fields = []
                        if hasattr(inner_model, "model_fields"):
                            for field_name, field_obj in inner_model.model_fields.items():
                                desc = field_obj.description or f"The {field_name}"
                                schema_fields.append(f"- {field_name}: {desc}")
                
                # Create concrete examples based on the actual Question model
                example_1 = {
                    "question": "What is the main purpose of the document?",
                    "answer": "Based on the context, the document discusses...",
                    "follow_up_question": "How does this relate to the broader context?",
                    "follow_up_answer": "This relates to the broader context by..."
                }
                
                example_2 = {
                    "question": "What are the key points mentioned?",
                    "answer": "The key points mentioned in the document are...",
                    "follow_up_question": "Which point is most significant?",
                    "follow_up_answer": "The most significant point is..."
                }
                
                # Format the response based on whether it's a batch
                if "Batch" in custom_schema.__name__:
                    full_example = {"items": [example_1, example_2]}
                    response_format = '{"items": [<questions>]}'
                else:
                    full_example = [example_1, example_2]
                    response_format = '[<questions>]'
                
                schema_instructions = f"""\n\nCRITICAL: You MUST generate output in EXACTLY this JSON format:\n\nRequired fields (and ONLY these fields):\n- question: The question text\n- answer: The complete answer\n- follow_up_question: A related follow-up question\n- follow_up_answer: The answer to the follow-up\n\nDO NOT include any other fields like thought_process, citations, difficulty, etc.\n\nExamples of CORRECT format:\n```json\n{json.dumps(full_example, indent=2)}\n```\n\nGenerate questions following EXACTLY this structure."""
                
                system_msg["content"] = original_system + schema_instructions
                logger.info(f"Added custom schema instructions to system prompt")
                logger.info(f"Loaded custom schema: {custom_schema.__name__}")

            except Exception as e:
                logger.warning(f"Failed to load custom schema for prompting: {e}")

        with log_step("generating_questions"):
            responses, index_map = _build_and_run_inference(
                dataset,
                system_msg,
                stage_cfg,
                build_single_shot_inference_calls,
                "single_shot_question_generation",
                config,
            )

        with log_step("saving_questions"):
            # Check if custom schema is configured
            if hasattr(stage_cfg, "custom_schema_path") and stage_cfg.custom_schema_path:
                logger.info(f"Using custom schema from {stage_cfg.custom_schema_path}")
                rows = parse_responses_with_custom_schema(
                    responses,
                    index_map,
                    stage_cfg,
                    stage_cfg.custom_schema_path,
                    stage_cfg.custom_schema_class,
                    stage_cfg.custom_schema_auto_batch,
                )
            else:
                rows = parse_single_shot_responses(responses, index_map, stage_cfg)

            if rows:
                _save_questions(rows, config, "single_shot_questions")
                logger.info(f"Saved {len(rows)} single-shot questions")


def run_multi_hop(config) -> None:
    """Generate multi-hop questions."""
    stage_cfg = config.pipeline.multi_hop_question_generation
    if not stage_cfg.run:
        logger.info("Multi-hop question generation disabled")
        return

    mode = stage_cfg.question_mode.strip().lower() if stage_cfg.question_mode else "open-ended"
    if mode not in {"open-ended", "multi-choice"}:
        logger.warning(f"Invalid question_mode '{mode}', defaulting to 'open-ended'")
        mode = "open-ended"
    system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode, is_multi=True)}

    chunked_ds = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(chunked_ds)} documents for multi-hop")

    # Process regular multi-hop
    _process_questions(
        chunked_ds, "multi_hop_questions", system_msg, stage_cfg, config, "multi_hop_question_generation"
    )


def run_cross_document(config) -> None:
    """Generate cross-document questions."""
    stage_cfg = config.pipeline.cross_document_question_generation
    if not stage_cfg.run:
        logger.info("Cross-document question generation disabled")
        return

    mode = stage_cfg.question_mode.strip().lower() if stage_cfg.question_mode else "open-ended"
    if mode not in {"open-ended", "multi-choice"}:
        logger.warning(f"Invalid question_mode '{mode}', defaulting to 'open-ended'")
        mode = "open-ended"
    system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode, is_multi=True)}

    chunked_ds = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(chunked_ds)} documents for cross-document")

    # Create cross-document configuration dict for compatibility
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
            cross_ds, "cross_document_questions", system_msg, stage_cfg, config, "cross_document_question_generation"
        )


def _process_questions(dataset: Dataset, label: str, system_msg: dict, stage_cfg: Any, config, step_name: str) -> None:
    """Process and save a set of questions."""
    if not dataset or len(dataset) == 0:
        logger.warning(f"No valid {label} dataset")
        return

    responses, index_map = _build_and_run_inference(
        dataset, system_msg, stage_cfg, build_multi_hop_inference_calls, step_name, config
    )

    if rows := parse_multi_hop_responses(responses, index_map, stage_cfg):
        _save_questions(rows, config, label)
