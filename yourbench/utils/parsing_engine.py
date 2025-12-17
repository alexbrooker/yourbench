import re
import json
import random
import string
import hashlib
from typing import Any, Optional

from loguru import logger

from yourbench.utils.question_models import QuestionRow, validate_list, force_int_in_range
from yourbench.schemas.question_schemas import (
    MultiChoiceQuestion,
    MultiHopQuestionBatch,
    SingleShotQuestionBatch,
)
from yourbench.utils.structured_generation import (
    StructuredGenerator,
)


# JSON parsing functions


def _attempt_json_parse(json_str: str) -> Any:
    """
    Attempt to parse a JSON string. Return parsed object if success,
    or None if parsing fails.
    """
    try:
        return json.loads(json_str)
    except Exception:
        return None


def _maybe_strip_triple_backticks(text_in: str) -> str:
    """
    Removes triple backticks (``` or ```json) from the beginning
    and end of a string, if present.
    """
    if not text_in or not isinstance(text_in, str):
        return ""
    try:
        pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, text_in)
        if match:
            return match.group(1)
    except Exception as e:
        logger.debug(f"Error stripping backticks: {e}")
    return text_in


def _best_effort_json_extract(full_text: str) -> list[str]:
    """
    Collect bracket-delimited substrings that might be valid JSON.
    Returns a list of candidates (which may be empty).
    """
    if not full_text or not isinstance(full_text, str):
        return []
    candidates = []
    try:
        pattern = r"([\[{].*?[\]}])"
        matches = re.findall(pattern, full_text, flags=re.DOTALL)
        for match_text in matches:
            if (match_text.startswith("[") and match_text.endswith("]")) or (
                match_text.startswith("{") and match_text.endswith("}")
            ):
                candidates.append(match_text.strip())
    except Exception as e:
        logger.debug(f"Error in best-effort JSON extraction: {e}")
    return candidates


def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extract text enclosed in <tag>...</tag> from the given string.
    Returns an empty string if the tag is not found.
    """
    try:
        pattern = rf"<{tag}\s*>([\s\S]*?)</{tag}>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    except Exception as e:
        logger.debug(f"Error extracting tag content for '{tag}': {e}")
    return ""


def extract_content_from_xml_tags(full_content, xml_tag):
    # This function extracts the content between the XML tags
    # It uses regex to find the content and includes error handling

    # Define the regex patterns to match the content
    pattern_with_closing_tag = f"<{xml_tag}>(.*?)</{xml_tag}>"
    pattern_without_closing_tag = f"<{xml_tag}>(.*)"

    try:
        # First, try to find matches with both opening and closing tags
        matches_with_closing = re.findall(pattern_with_closing_tag, full_content, re.DOTALL)
        if matches_with_closing:
            return matches_with_closing[0].strip()

        # If no matches found, try to find content with only opening tag
        matches_without_closing = re.findall(pattern_without_closing_tag, full_content, re.DOTALL)
        if matches_without_closing:
            return matches_without_closing[0].strip()

        # If still no matches found, return an empty string
        return ""

    except Exception as extraction_error:
        logger.error(f"Error extracting content from XML tags: {extraction_error}")
        return ""


def parse_qa_pairs_from_response(raw_response: str) -> list[dict[str, Any]]:
    """
    Attempt to parse question-answer pairs from a raw LLM response.

    The function searches in this priority order:
        1. <output_json>...</output_json> tags.
        2. ```json fenced code blocks.
        3. Best-effort bracket-based extraction.

    If any candidate JSON is found, it attempts to parse it. If parsing
    succeeds and yields a list, it returns that list. Otherwise, it
    returns an empty list.

    Even if this returns an empty list, callers are expected to store
    the raw response (e.g., so the pipeline does not lose data).

    Args:
        raw_response (str): The complete raw response string from the model.

    Returns:
        A list of dict objects, each presumably containing
        question-answer information. If no valid parse is found,
        an empty list is returned.
    """
    if not raw_response or not isinstance(raw_response, str):
        return []

    # 1) Check for <output_json>...</output_json>
    extracted_json_str = _extract_tag_content(raw_response, "output_json")
    if extracted_json_str.strip():
        possible_parsed = _attempt_json_parse(_maybe_strip_triple_backticks(extracted_json_str))
        if isinstance(possible_parsed, list):
            return possible_parsed

    # 2) Check for ```json fenced code block
    fence_pattern = r"```json\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, raw_response)
    if fence_match:
        possible_parsed = _attempt_json_parse(fence_match.group(1).strip())
        if isinstance(possible_parsed, list):
            return possible_parsed

    # 3) Best-effort bracket-based extraction
    bracket_candidates = _best_effort_json_extract(raw_response)
    for candidate in bracket_candidates:
        possible_parsed = _attempt_json_parse(candidate)
        if isinstance(possible_parsed, list):
            return possible_parsed

    # If no valid parse was found, return empty.
    return []


# QA response parsing utils

OPEN_ENDED_TYPES = {
    "analytical",
    "application-based",
    "clarification",
    "counterfactual",
    "conceptual",
    "true-false",
    "factual",
    "open-ended",
    "false-premise",
    "edge-case",
}

MULTI_CHOICE_TYPES = {
    "analytical",
    "application-based",
    "clarification",
    "counterfactual",
    "conceptual",
    "true-false",
    "factual",
    "false-premise",
    "edge-case",
}


def normalize_open_ended(pair: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Ensures open-ended questions are valid.
    Returns None if the entry should be skipped.
    """
    pair = dict(pair)  # defensive copy
    mode = pair.get("question_mode", "").strip().lower()
    q_type = pair.get("question_type", "").strip().lower()

    if mode != "open-ended":
        return pair

    if q_type not in OPEN_ENDED_TYPES:
        logger.warning(f"Inconsistent open-ended question_type: '{q_type}'")
        return pair

    # No choices for open-ended
    pair["choices"] = []

    answer = pair.get("answer", "").strip()
    if len(answer) == 1 and answer.upper() in {"A", "B", "C", "D"}:
        # Misclassified multiple choice
        return None

    return pair


def normalize_multi_choice(pair: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Ensures multiple-choice questions are valid.
    Returns None if the entry should be skipped.
    """
    pair = dict(pair)
    mode = pair.get("question_mode", "").strip().lower()
    q_type = pair.get("question_type", "").strip().lower()

    if mode != "multi-choice":
        return pair

    if q_type not in MULTI_CHOICE_TYPES:
        logger.warning(f"Inconsistent multiple-choice question_type: '{q_type}'")
        return pair

    choices = validate_list(pair.get("choices", []))
    if len(choices) != 4:
        logger.warning("MCQ must have exactly 4 choices.")
        return None

    pair["choices"] = choices
    return pair


def parse_single_shot_responses(responses, index_map, stage_cfg):
    """Parse single-shot responses with optional structured output support.

    If use_structured_outputs is enabled in config, attempts to parse responses
    using Pydantic schemas for validation. Falls back to legacy parsing if needed.
    """
    rows = []
    question_mode = (
        str(
            getattr(stage_cfg, "question_mode", "open-ended")
            if hasattr(stage_cfg, "question_mode")
            else stage_cfg.get("question_mode", "open-ended")
            if isinstance(stage_cfg, dict)
            else "open-ended"
        )
        .strip()
        .lower()
    )

    # Check if structured outputs are enabled
    use_structured = (
        getattr(stage_cfg, "use_structured_outputs", False)
        if hasattr(stage_cfg, "use_structured_outputs")
        else stage_cfg.get("use_structured_outputs", False)
        if isinstance(stage_cfg, dict)
        else False
    )

    if use_structured:
        logger.info("Using structured output parsing for single-shot questions")
        generator = StructuredGenerator()

    for model, replies in responses.items():
        if len(replies) != len(index_map):
            logger.error(f"Mismatch: model '{model}' replies={len(replies)}, expected={len(index_map)}")
            continue

        for i, reply in enumerate(replies):
            parsed_qa_pairs = None

            # Try structured parsing first if enabled
            if use_structured:
                structured_batch = generator.parse_structured_response(reply, SingleShotQuestionBatch, strict=False)

                if structured_batch:
                    # Convert structured response to legacy format
                    parsed_qa_pairs = []
                    for q in structured_batch.qa_pairs:
                        pair_dict = {
                            "question": q.question,
                            "answer": q.answer,
                            "question_type": q.question_type.value,
                            "estimated_difficulty": q.estimated_difficulty,
                            "thought_process": q.thought_process,
                            "citations": [c.text for c in q.citations],
                        }
                        if isinstance(q, MultiChoiceQuestion):
                            pair_dict["choices"] = q.choices
                            # Ensure answer matches correct choice
                            choice_idx = ord(q.correct_choice) - ord("A")
                            if 0 <= choice_idx < len(q.choices):
                                pair_dict["answer"] = q.choices[choice_idx]
                        parsed_qa_pairs.append(pair_dict)
                else:
                    logger.debug(f"Structured parsing failed at index {i}, trying legacy parser")

            # Fallback to legacy parsing if structured parsing wasn't used or failed
            if parsed_qa_pairs is None:
                parsed_qa_pairs = parse_qa_pairs_from_response(reply)

            if not parsed_qa_pairs:
                logger.warning(f"No parseable QA pairs at index {i}.")
                continue

            for pair in parsed_qa_pairs:
                if not isinstance(pair, dict):
                    logger.debug(f"Skipping non-dict item in single-shot response: {type(pair)}")
                    continue
                try:
                    pair = shuffle_mcq(pair)
                    pair["question_mode"] = question_mode

                    if question_mode == "open-ended":
                        pair = normalize_open_ended(pair)
                        if pair is None:
                            continue
                        choices = []
                    elif question_mode == "multi-choice":
                        pair = normalize_multi_choice(pair)
                        if pair is None:
                            continue
                        choices = pair["choices"]
                    else:
                        logger.warning(f"Unsupported question_mode: {question_mode}")
                        continue

                    citations = validate_list(pair.get("citations", []))

                    rows.append(
                        QuestionRow(
                            chunk_id=index_map[i][2],
                            source_chunk_ids=None,
                            document_id=index_map[i][1],
                            additional_instructions=stage_cfg.additional_instructions,
                            question=str(pair.get("question", "")).strip(),
                            self_answer=str(pair.get("answer", "")).strip(),
                            choices=choices,
                            estimated_difficulty=force_int_in_range(pair.get("estimated_difficulty", 5), 1, 10),
                            self_assessed_question_type=str(pair.get("question_type", "")).strip(),
                            question_mode=pair["question_mode"],
                            generating_model=model,
                            thought_process=str(pair.get("thought_process", "")),
                            raw_response=reply,
                            citations=citations,
                        ).to_dict(format="single-hop")
                    )
                except Exception as e:
                    logger.error(f"Error parsing QA pair at index {i}: {e}")
                    continue

    return rows


def parse_multi_hop_responses(responses, index_map, stage_cfg):
    """Parse multi-hop responses with optional structured output support.

    If use_structured_outputs is enabled in config, attempts to parse responses
    using Pydantic schemas for validation. Falls back to legacy parsing if needed.
    """
    rows = []
    question_mode = (
        str(
            getattr(stage_cfg, "question_mode", "open-ended")
            if hasattr(stage_cfg, "question_mode")
            else stage_cfg.get("question_mode", "open-ended")
            if isinstance(stage_cfg, dict)
            else "open-ended"
        )
        .strip()
        .lower()
    )

    # Check if structured outputs are enabled
    use_structured = (
        getattr(stage_cfg, "use_structured_outputs", False)
        if hasattr(stage_cfg, "use_structured_outputs")
        else stage_cfg.get("use_structured_outputs", False)
        if isinstance(stage_cfg, dict)
        else False
    )

    if use_structured:
        logger.info("Using structured output parsing for multi-hop questions")
        generator = StructuredGenerator()

    for model, replies in responses.items():
        for i, raw in enumerate(replies):
            parsed = None

            # Try structured parsing first if enabled
            if use_structured:
                structured_batch = generator.parse_structured_response(raw, MultiHopQuestionBatch, strict=False)

                if structured_batch:
                    # Convert structured response to legacy format
                    parsed = []
                    for q in structured_batch.qa_pairs:
                        pair_dict = {
                            "question": q.question,
                            "answer": q.answer,
                            "question_type": q.question_type.value,
                            "estimated_difficulty": q.estimated_difficulty,
                            "thought_process": q.thought_process,
                            "citations": [c.text for c in q.citations],
                        }
                        if isinstance(q, MultiChoiceQuestion):
                            pair_dict["choices"] = q.choices
                            # Ensure answer matches correct choice
                            choice_idx = ord(q.correct_choice) - ord("A")
                            if 0 <= choice_idx < len(q.choices):
                                pair_dict["answer"] = q.choices[choice_idx]
                        parsed.append(pair_dict)
                else:
                    logger.debug(f"Structured parsing failed for multi-hop at index {i}, trying legacy parser")

            # Fallback to legacy parsing if structured parsing wasn't used or failed
            if parsed is None:
                parsed = parse_qa_pairs_from_response(raw)

            for pair in parsed:
                if not isinstance(pair, dict):
                    logger.debug(f"Skipping non-dict item in multi-hop response: {type(pair)}")
                    continue
                try:
                    pair = shuffle_mcq(pair)
                    pair["question_mode"] = question_mode

                    if question_mode == "open-ended":
                        pair = normalize_open_ended(pair)
                        if pair is None:
                            continue
                        choices = []
                    elif question_mode == "multi-choice":
                        pair = normalize_multi_choice(pair)
                        if pair is None:
                            continue
                        choices = pair["choices"]
                    else:
                        logger.warning(f"Unsupported question_mode: {question_mode}")
                        continue

                    citations = validate_list(pair.get("citations", []))

                    rows.append(
                        QuestionRow(
                            chunk_id=None,
                            source_chunk_ids=index_map[i][2],
                            document_id=index_map[i][1],
                            additional_instructions=stage_cfg.additional_instructions,
                            question=str(pair.get("question", "")).strip(),
                            self_answer=str(pair.get("answer", "")).strip(),
                            choices=choices,
                            estimated_difficulty=force_int_in_range(pair.get("estimated_difficulty", 5), 1, 10),
                            self_assessed_question_type=str(pair.get("question_type", "")).strip(),
                            question_mode=pair["question_mode"],
                            generating_model=model,
                            thought_process=str(pair.get("thought_process", "")),
                            raw_response=raw,
                            citations=citations,
                        ).to_dict(format="multi-hop")
                    )
                except Exception as e:
                    logger.warning(f"Parse error in multi-hop QA for doc {index_map[i][1]}: {e}")
                    continue

    return rows


def shuffle_mcq(question_dict: dict) -> dict:
    """
    Shuffles MCQ choices randomly and ensures the correct answer is placed under a random label A-D.
    The final choices are labeled A., B., C., D. in order, but the correct answer may be under any of them.
    """
    labeled_choices = question_dict.get("choices", [])
    answer_letter = question_dict.get("answer", "").strip().upper()

    if not labeled_choices or not answer_letter:
        return question_dict

    # Extract raw text (removing A., B., etc.)
    raw_choices = [choice[3:].strip() for choice in labeled_choices]
    answer_index = ord(answer_letter) - ord("A")
    answer_choice_text = raw_choices[answer_index]

    # Shuffle the raw choices randomly
    seed_input = repr((raw_choices, answer_letter))
    seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16)

    rng = random.Random(seed)
    rng.shuffle(raw_choices)

    # Find new index of the correct choice
    new_correct_index = raw_choices.index(answer_choice_text)
    new_answer_letter = chr(ord("A") + new_correct_index)

    # Re-label as A., B., C., D.
    labeled_shuffled = [f"({chr(ord('A') + i)}) {text}" for i, text in enumerate(raw_choices)]

    # Update the question dict
    question_dict["choices"] = labeled_shuffled
    question_dict["answer"] = new_answer_letter

    return question_dict


def _remove_duplicate_questions(rows: list[dict]) -> list[dict]:
    """
    Removes duplicate question entries based on an enhanced normalized question text.
    Normalization includes:
        - Lowercasing
        - Removing punctuation
        - Removing digits
        - Stripping and collapsing whitespace
    The original question format is preserved in the output.
    """
    seen_questions = set()
    deduped_rows = []

    for row in rows:
        question = row.get("question")
        if question is None:
            deduped_rows.append(row)
            continue

        # Normalize for deduplication
        norm_question = question.lower()
        norm_question = re.sub(rf"[{re.escape(string.punctuation)}]", "", norm_question)
        norm_question = re.sub(r"\d+", "", norm_question)
        norm_question = " ".join(norm_question.split())

        if norm_question not in seen_questions:
            seen_questions.add(norm_question)
            deduped_rows.append(row)

    removed = len(rows) - len(deduped_rows)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate questions. Final count: {len(deduped_rows)}")
    else:
        logger.info(f"No duplicate questions detected. Final count: {len(deduped_rows)}")

    return deduped_rows


def parse_responses_with_custom_schema(
    responses: dict,
    index_map: list,
    stage_cfg: Any,
    schema_path: str,
    schema_class: Optional[str] = None,
    auto_batch: bool = True,
) -> list[dict]:
    """Parse responses using a custom Pydantic schema.

    Args:
        responses: Model responses dictionary
        index_map: Index mapping for responses
        stage_cfg: Stage configuration
        schema_path: Path to Python file containing Pydantic model
        schema_class: Name of the class to use (optional)
        auto_batch: Whether to auto-wrap in batch if needed

    Returns:
        List of parsed question dictionaries
    """
    import json

    from yourbench.utils.schema_loader import (
        load_schema_from_file,
        create_example_from_schema,
        generate_schema_description,
        validate_schema_for_generation,
    )
    from yourbench.utils.question_models import QuestionRow
    from yourbench.utils.structured_generation import StructuredGenerator

    # Load the custom schema
    try:
        custom_schema = load_schema_from_file(schema_path, schema_class, auto_batch)
        validate_schema_for_generation(custom_schema)
        logger.info(f"Loaded custom schema: {custom_schema.__name__}")
    except Exception as e:
        logger.error(f"Failed to load custom schema from {schema_path}: {e}")
        logger.warning("Falling back to standard parsing")
        return parse_single_shot_responses(responses, index_map, stage_cfg)

    # Generate schema description and example for logging
    schema_desc = generate_schema_description(custom_schema)
    schema_example = create_example_from_schema(custom_schema)
    logger.debug(f"Schema description:\n{schema_desc}")
    logger.debug(f"Schema example: {json.dumps(schema_example, indent=2)}")

    # Initialize structured generator
    generator = StructuredGenerator()
    rows = []

    # Process responses
    for model, replies in responses.items():
        if len(replies) != len(index_map):
            logger.error(f"Mismatch: model '{model}' replies={len(replies)}, expected={len(index_map)}")
            continue

        for i, reply in enumerate(replies):
            try:
                # Parse with custom schema
                parsed = generator.parse_structured_response(reply, custom_schema, strict=False)

                if not parsed:
                    logger.warning(f"Failed to parse response at index {i} with custom schema")
                    continue

                # Convert to question rows
                # Handle both single items and batches
                items = []
                if hasattr(parsed, "items"):
                    # It's a batch
                    items = parsed.items
                elif hasattr(parsed, "__dict__"):
                    # Single item
                    items = [parsed]

                for item in items:
                    # Convert Pydantic model to dict
                    item_dict = item.model_dump() if hasattr(item, "model_dump") else item.dict()

                    # Map custom fields to standard QuestionRow fields
                    # This is flexible - users can have any field names
                    question_text = item_dict.get("question", "")
                    if not question_text:
                        # Try other common field names
                        question_text = item_dict.get("query", "") or item_dict.get("prompt", "")

                    answer_text = item_dict.get("answer", "")
                    if not answer_text:
                        # Try other common field names
                        answer_text = (
                            item_dict.get("response", "")
                            or item_dict.get("solution", "")
                            or item_dict.get("self_answer", "")
                        )

                    # Build QuestionRow with available fields
                    row = QuestionRow(
                        chunk_id=index_map[i][0],
                        source_chunk_ids=[index_map[i][0]],
                        document_id=index_map[i][1],
                        additional_instructions=getattr(stage_cfg, "additional_instructions", ""),
                        question=question_text.strip(),
                        self_answer=answer_text.strip(),
                        choices=[],  # Will be filled if multi-choice
                        estimated_difficulty=item_dict.get("difficulty", 5),
                        self_assessed_question_type=item_dict.get("question_type", "custom"),
                        question_mode="open-ended",  # Will be updated based on schema
                        generating_model=model,
                        thought_process=item_dict.get("thought_process", ""),
                        raw_response=reply,
                        citations=item_dict.get("citations", []),
                    )

                    # Check for multi-choice fields
                    if "choices" in item_dict:
                        row.choices = item_dict["choices"]
                        row.question_mode = "multi-choice"

                    # Check for follow-up questions (from example_pydantic.py)
                    row_dict = row.to_dict()

                    # Add any extra fields from custom schema
                    for key, value in item_dict.items():
                        if key not in row_dict and value:  # Only add non-empty values
                            row_dict[key] = value

                    rows.append(row_dict)

            except Exception as e:
                logger.warning(f"Error parsing response at index {i} with custom schema: {e}")
                continue

    return rows
