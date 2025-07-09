from typing import List
from dataclasses import dataclass

from loguru import logger

# User prompts are now passed via configuration
from yourbench.utils.chunking_utils import sample_multihop_groups, sample_single_hop_chunks
from yourbench.utils.inference.inference_core import InferenceCall


@dataclass
class InferenceJob:
    inference_calls: List[InferenceCall]


def build_single_shot_inference_calls(dataset, system_msg, stage_cfg, sampling_cfg):
    calls = []
    index_map = []

    for idx, row in enumerate(dataset):
        document_chunks = row.get("chunks") or []
        selected_chunks = sample_single_hop_chunks(document_chunks, sampling_cfg)

        for ch_idx, chunk in enumerate(selected_chunks):
            chunk_id = chunk.get("chunk_id", f"{idx}_{ch_idx}")
            chunk_text = chunk.get("chunk_text", "")
            user_msg = {
                "role": "user",
                "content": stage_cfg.single_shot_user_prompt.format(
                    title=row.get("document_filename", f"doc_{idx}"),
                    document_summary=row.get("document_summary", ""),
                    text_chunk=chunk_text,
                    additional_instructions=getattr(stage_cfg, "additional_instructions", "")
                    if hasattr(stage_cfg, "additional_instructions")
                    else stage_cfg.get("additional_instructions", "")
                    if isinstance(stage_cfg, dict)
                    else "",
                ),
            }
            calls.append(InferenceCall(messages=[system_msg, user_msg], tags=["single_shot_qa"]))
            index_map.append((idx, row.get("document_id", f"doc_{idx}"), chunk_id))

    return calls, index_map


def build_multi_hop_inference_calls(dataset, system_msg, stage_cfg):
    calls = []
    index_map = []

    for idx, row in enumerate(dataset):
        chunk_sampling = (
            getattr(stage_cfg, "chunk_sampling", {})
            if hasattr(stage_cfg, "chunk_sampling")
            else stage_cfg.get("chunk_sampling", {})
            if isinstance(stage_cfg, dict)
            else {}
        )
        groups = sample_multihop_groups(row.get("multihop_chunks") or [], chunk_sampling)
        for group in groups:
            # TODO how it's possible here?
            if not isinstance(group, dict):
                logger.warning("Multihop groups are not a dict, skipping")
                continue
            chunk_ids = group.get("chunk_ids", [])
            texts = group.get("chunks_text", [])
            if not texts:
                logger.warning("Chunks texts are empty, skipping")
                continue
            full_text = "".join([f"<text_chunk_{i}>{t}</text_chunk_{i}>\n" for i, t in enumerate(texts)])
            user_msg = {
                "role": "user",
                "content": stage_cfg.multi_hop_user_prompt.format(
                    title=row.get("document_filename", ""),
                    document_summary=row.get("document_summary", ""),
                    chunks=full_text,
                    additional_instructions=getattr(stage_cfg, "additional_instructions", "")
                    if hasattr(stage_cfg, "additional_instructions")
                    else stage_cfg.get("additional_instructions", "")
                    if isinstance(stage_cfg, dict)
                    else "",
                ),
            }
            calls.append(InferenceCall(messages=[system_msg, user_msg], tags=["multi_hop_qa"]))
            index_map.append((idx, row.get("document_id", f"doc_{idx}"), chunk_ids))

    return calls, index_map
