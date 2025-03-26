# yourbench/pipeline/answer_generation.py

"""
answer_generation.py

Stage: answer_generation
------------------------
Generates answers to both single-shot and multi-hop questions for one or more 'scenarios'.
Each scenario is appended to the *same* output subsets:
  - single_shot_questions_with_answers
  - multi_hop_questions_with_answers

We distinguish the scenario by the 'answer_fashion' column in the final dataset (e.g. 'zero_shot', 'with_correct_chunk').

Configuration Example (YAML):
  answer_generation:
    run: true
    scenarios:
      - zero_shot
      - with_correct_chunk
"""

from typing import Any, Dict, List

from loguru import logger

from datasets import Dataset
from yourbench.utils.prompts import GOLD_QA_USER_PROMPT, ZEROSHOT_QA_USER_PROMPT
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.inference_engine import InferenceCall, run_inference


def run(config: Dict[str, Any]) -> None:
    """
    Main entry point for the answer_generation stage.

    Looks at config["pipeline"]["answer_generation"]["scenarios"] to see which answer-generation
    modes to run. Currently supported: "zero_shot" and "with_correct_chunk".

    Always concatenates new rows into:
      - single_shot_questions_with_answers
      - multi_hop_questions_with_answers

    A column 'answer_fashion' indicates which scenario was used to generate the answer.
    """
    stage_cfg = config.get("pipeline", {}).get("answer_generation", {})
    if not stage_cfg.get("run", False):
        logger.info("answer_generation stage is disabled. Skipping.")
        return

    # Force dataset concatenation ON for this stage
    logger.info("Forcing 'concat_if_exist = True' so new rows get appended.")
    config.setdefault("hf_configuration", {})
    config["hf_configuration"]["concat_if_exist"] = True

    # Determine which scenarios to run
    scenarios = stage_cfg.get("scenarios", ["zero_shot"])
    logger.info(f"answer_generation scenarios = {scenarios}")

    for scenario in scenarios:
        logger.info(f"--- Starting scenario: {scenario} ---")

        # Single-shot
        _generate_answers_for_questions(
            config=config,
            source_subset="single_shot_questions",
            scenario=scenario,
            output_subset="single_shot_questions_with_answers",
        )

        # Multi-hop
        _generate_answers_for_questions(
            config=config,
            source_subset="multi_hop_questions",
            scenario=scenario,
            output_subset="multi_hop_questions_with_answers",
        )

    logger.success("answer_generation stage complete.")


def _generate_answers_for_questions(
    config: Dict[str, Any], source_subset: str, scenario: str, output_subset: str
) -> None:
    """
    Loads a question subset (single_shot or multi_hop), runs the chosen scenario
    (zero_shot or with_correct_chunk), and appends results to the specified
    output_subset, storing 'scenario' in answer_fashion.

    The final dataset has new columns:
      - 'answering_model'
      - 'answer'
      - 'answer_fashion' = scenario
    """
    logger.info(f"Loading questions from '{source_subset}' for scenario='{scenario}'.")
    try:
        question_ds = custom_load_dataset(config, subset=source_subset)
    except Exception as e:
        logger.warning(f"Could not load {source_subset} dataset. Error: {e}")
        return

    if question_ds is None or len(question_ds) == 0:
        logger.warning(f"No data in {source_subset}; skipping answer generation.")
        return

    # Build inference calls for the chosen scenario
    calls, row_map = _build_inference_calls_scenario(config, question_ds, scenario, source_subset)
    if not calls:
        logger.warning(f"No inference calls for subset={source_subset}, scenario={scenario}.")
        return

    # Run them
    responses = run_inference(
        config=config,
        step_name="answer_generation",  # "model_roles" can link here
        inference_calls=calls,
    )
    if not responses:
        logger.warning(f"No responses returned for subset={source_subset}, scenario={scenario}.")
        return

    # Parse and assemble
    final_ds = _parse_and_assemble(
        question_ds=question_ds, responses_dict=responses, row_map=row_map, scenario=scenario
    )
    if final_ds is None or len(final_ds) == 0:
        logger.warning(f"No answers parsed for subset={source_subset}, scenario={scenario}.")
        return

    # Append to the same subset
    custom_save_dataset(dataset=final_ds, config=config, subset=output_subset)
    logger.info(f"Appended {len(final_ds)} new answers to '{output_subset}' (scenario={scenario}).")


def _build_inference_calls_scenario(config: Dict[str, Any], question_ds: Dataset, scenario: str, source_subset: str):
    """
    For each row in question_ds, build an InferenceCall with the appropriate prompt:
     - "zero_shot" => use ZEROSHOT_QA_USER_PROMPT
     - "with_correct_chunk" => retrieve chunk(s) from 'chunked' dataset and use GOLD_QA_USER_PROMPT

    Returns (calls, row_map) for re-assembly after inference.
    """
    calls = []
    row_map = []

    if "question" not in question_ds.column_names:
        logger.error("Dataset missing 'question' column. Cannot build calls.")
        return calls, row_map

    doc_meta_map = {}
    if scenario == "with_correct_chunk":
        # Load chunked dataset to retrieve correct chunk text
        logger.info("Loading 'chunked' dataset to retrieve chunk text for with_correct_chunk scenario.")
        try:
            chunked_ds = custom_load_dataset(config, subset="chunked")
            doc_meta_map = _build_doc_meta_map(chunked_ds)
        except Exception as e:
            logger.error(f"Failed loading chunked dataset for scenario={scenario}: {e}")
            return calls, row_map

    for i, row in enumerate(question_ds):
        qtext = row.get("question", "")
        if not qtext or not isinstance(qtext, str):
            continue

        if scenario == "zero_shot":
            user_prompt = ZEROSHOT_QA_USER_PROMPT.format(question=qtext)
            user_msg = {"role": "user", "content": user_prompt}
            calls.append(InferenceCall(messages=[user_msg], tags=["zero_shot"]))
            row_map.append(i)

        elif scenario == "with_correct_chunk":
            doc_id = row.get("document_id", "")
            if not doc_id:
                logger.debug(f"Row {i} missing document_id, skipping with_correct_chunk.")
                continue

            # Single-shot uses 'chunk_id', multi-hop uses 'source_chunk_ids'
            if source_subset.startswith("single_shot"):
                chunk_id = row.get("chunk_id", "")
                chunk_text = doc_meta_map.get(doc_id, {}).get("chunks_map", {}).get(chunk_id, "")
                doc_summary = doc_meta_map.get(doc_id, {}).get("document_summary", "")
                user_prompt = GOLD_QA_USER_PROMPT.format(question=qtext, summary=doc_summary, document=chunk_text)
                user_msg = {"role": "user", "content": user_prompt}
                calls.append(InferenceCall(messages=[user_msg], tags=["with_correct_chunk"]))
                row_map.append(i)

            elif source_subset.startswith("multi_hop"):
                chunk_ids = row.get("source_chunk_ids", [])
                if not isinstance(chunk_ids, list):
                    chunk_ids = []
                doc_summary = doc_meta_map.get(doc_id, {}).get("document_summary", "")
                combined_texts = []
                for cid in chunk_ids:
                    text_ = doc_meta_map.get(doc_id, {}).get("chunks_map", {}).get(cid, "")
                    if text_:
                        combined_texts.append(text_)
                chunk_text_joined = "\n\n".join(combined_texts)

                user_prompt = GOLD_QA_USER_PROMPT.format(
                    question=qtext, summary=doc_summary, document=chunk_text_joined
                )
                user_msg = {"role": "user", "content": user_prompt}
                calls.append(InferenceCall(messages=[user_msg], tags=["with_correct_chunk"]))
                row_map.append(i)

        else:
            logger.warning(f"Unrecognized scenario '{scenario}' (row={i}), skipping.")
            continue

    return calls, row_map


def _parse_and_assemble(
    question_ds: Dataset, responses_dict: Dict[str, List[str]], row_map: List[int], scenario: str
) -> Dataset:
    """
    For each model, parse <answer> from each response. We replicate the original row
    and add columns: 'answering_model', 'answer', 'answer_fashion' = scenario.
    """
    final_records = {col: [] for col in question_ds.column_names}
    final_records["answering_model"] = []
    final_records["answer"] = []
    final_records["answer_fashion"] = []

    for model_name, model_responses in responses_dict.items():
        logger.info(f"Processing {len(model_responses)} responses from model={model_name} for scenario={scenario}.")
        n_common = min(len(model_responses), len(row_map))
        for idx in range(n_common):
            resp = model_responses[idx]
            row_idx = row_map[idx]

            parsed_answer = extract_content_from_xml_tags(resp, "answer")
            if not parsed_answer.strip():
                parsed_answer = "No <answer> found."

            # replicate original row
            for col in question_ds.column_names:
                final_records[col].append(question_ds[col][row_idx])

            final_records["answering_model"].append(model_name)
            final_records["answer"].append(parsed_answer)
            final_records["answer_fashion"].append(scenario)

    if not final_records["answer"]:
        return None

    return Dataset.from_dict(final_records)


def _build_doc_meta_map(chunked_ds: Dataset) -> Dict[str, Any]:
    """
    Build a map doc_id -> {
        'document_summary': str,
        'document_text': str,
        'chunks_map': {chunk_id -> chunk_text}
    }
    for quick retrieval of the chunk text used in with_correct_chunk scenario.
    """
    meta_map = {}
    for row in chunked_ds:
        doc_id = row.get("document_id", "")
        doc_text = row.get("document_text", "")
        doc_summary = row.get("document_summary", "")
        chunk_entries = row.get("chunks", [])

        chunk_map = {}
        if isinstance(chunk_entries, list):
            for cdict in chunk_entries:
                cid = cdict.get("chunk_id", "")
                ctext = cdict.get("chunk_text", "")
                chunk_map[cid] = ctext

        meta_map[doc_id] = {"document_text": doc_text, "document_summary": doc_summary, "chunks_map": chunk_map}
    return meta_map
