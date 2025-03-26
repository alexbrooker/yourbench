from typing import Any, Dict, List

from loguru import logger

from datasets import Dataset
from yourbench.utils.prompts import SCORE_ANSWER_BINARY_USER_PROMPT
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.inference_engine import InferenceCall, run_inference


def run(config: Dict[str, Any]) -> None:
    """
    Stage: score_answer_binary
    --------------------------
    Loads single_shot_questions_with_answers and multi_hop_questions_with_answers,
    then uses a model to score each row (0 or 1) indicating correctness of the predicted answer.

    Produces:
      - single_shot_questions_scored
      - multi_hop_questions_scored
    """
    stage_cfg = config.get("pipeline", {}).get("score_answer_binary", {})
    if not stage_cfg.get("run", False):
        logger.info("score_answer_binary stage is disabled. Skipping.")
        return

    # Force dataset concatenation ON for this stage, so new rows get appended if needed.
    logger.info("Forcing 'concat_if_exist = True' so new rows get appended.")
    config.setdefault("hf_configuration", {})
    config["hf_configuration"]["concat_if_exist"] = True

    logger.info("Starting score_answer_binary stage...")

    # 1) Score the single-shot set
    _score_dataset(
        config=config,
        source_subset="single_shot_questions_with_answers",
        output_subset="single_shot_questions_scored",
    )

    # 2) Score the multi-hop set
    _score_dataset(
        config=config,
        source_subset="multi_hop_questions_with_answers",
        output_subset="multi_hop_questions_scored",
    )

    logger.success("score_answer_binary stage complete.")


def _score_dataset(config: Dict[str, Any], source_subset: str, output_subset: str) -> None:
    """
    Loads a 'with_answers' subset, builds scoring calls, parses out 0/1 from the response,
    and saves to output_subset.
    """
    logger.info(f"Loading questions+answers from '{source_subset}'...")
    try:
        ds_in = custom_load_dataset(config, subset=source_subset)
    except Exception as e:
        logger.warning(f"Could not load subset={source_subset}: {e}")
        return

    if ds_in is None or len(ds_in) == 0:
        logger.warning(f"No data in {source_subset}, skipping scoring.")
        return

    # Check columns
    required_cols = {"question", "answer", "answer_fashion", "answering_model", "self_answer"}
    missing_cols = required_cols - set(ds_in.column_names)
    if missing_cols:
        logger.warning(f"{source_subset} is missing required columns: {missing_cols}. Skipping.")
        return

    # 1) Build inference calls
    calls = []
    row_map = []
    for i, row in enumerate(ds_in):
        question = row.get("question", "")
        gold_answer = row.get("self_answer", "")
        predicted_answer = row.get("answer", "")

        if not question or not predicted_answer or not gold_answer:
            continue  # skip rows lacking required fields

        user_content = SCORE_ANSWER_BINARY_USER_PROMPT.format(
            question=question, ground_truth=gold_answer, predicted_answer=predicted_answer
        )
        user_msg = {"role": "user", "content": user_content}

        calls.append(InferenceCall(messages=[user_msg], tags=["score_answer_binary"]))
        row_map.append(i)

    if not calls:
        logger.warning(f"No scoring calls built for {source_subset}.")
        return

    # 2) Run inference
    responses_dict = run_inference(config=config, step_name="score_answer_binary", inference_calls=calls)
    if not responses_dict:
        logger.warning(f"No responses from model for {source_subset}.")
        return

    # 3) Parse and assemble
    final_ds = _parse_score_responses(ds_in, responses_dict, row_map)
    if final_ds is None or len(final_ds) == 0:
        logger.warning(f"No scores parsed for {source_subset}.")
        return

    # 4) Save to output_subset
    custom_save_dataset(dataset=final_ds, config=config, subset=output_subset)
    logger.info(f"Appended {len(final_ds)} new rows to subset='{output_subset}'.")


def _parse_score_responses(
    original_ds: Dataset,
    responses_dict: Dict[str, List[str]],
    row_map: List[int],
) -> Dataset:
    """
    Combine the original dataset with the model's <score> output.
    Produces new rows each with:
      - answering_model
      - answer_fashion
      - question, ground_truth_answer, answer
      - binary_score
      - scoring_model (which model gave the 0/1)
    """
    # Prepare structure to hold final records
    final_records = {col: [] for col in original_ds.column_names}
    final_records["scoring_model"] = []
    final_records["binary_score"] = []
    final_records["judgement"] = []
    final_records["scratchpad"] = []
    final_records["scoring_response"] = []

    # For each model's output, attach the parsed scores
    for model_name, responses in responses_dict.items():
        if len(responses) != len(row_map):
            logger.warning(f"Model={model_name} returned {len(responses)} responses, expected {len(row_map)}.")
            # We'll process only min(len(responses), len(row_map)) to stay safe
        n_common = min(len(responses), len(row_map))

        for idx in range(n_common):
            raw_resp = responses[idx]
            row_idx = row_map[idx]

            # Attempt to parse <score> from raw_resp
            parsed_score = extract_content_from_xml_tags(raw_resp, "score").strip()
            # extract the judgement
            parsed_judgement = extract_content_from_xml_tags(raw_resp, "judgement").strip()
            # extract the scratchpad
            parsed_scratchpad = extract_content_from_xml_tags(raw_resp, "scratchpad").strip()
            if parsed_score not in ("0", "1"):
                # If not found or invalid, default to 0
                parsed_score = "0"

            # replicate original row
            for col in original_ds.column_names:
                final_records[col].append(original_ds[col][row_idx])

            # Add new columns
            final_records["scoring_model"].append(model_name)
            final_records["binary_score"].append(int(parsed_score))
            final_records["judgement"].append(parsed_judgement)
            final_records["scratchpad"].append(parsed_scratchpad)
            # also add the full raw scoring response
            final_records["scoring_response"].append(raw_resp)

    if not final_records["binary_score"]:
        return None

    return Dataset.from_dict(final_records)
