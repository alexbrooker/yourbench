"""Token estimation utilities using tiktoken."""

from typing import TYPE_CHECKING
from pathlib import Path

import tiktoken
from loguru import logger


if TYPE_CHECKING:
    from yourbench.conf.schema import YourbenchConfig


def get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get tiktoken encoder with fallback to cl100k_base."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    encoder = get_encoder(encoding_name)
    return len(encoder.encode(text))


def count_file_tokens(file_path: Path, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a file."""
    if not file_path.exists():
        return 0
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return count_tokens(text, encoding_name)
    except Exception as e:
        logger.debug(f"Error reading {file_path}: {e}")
        return 0


def estimate_source_tokens(source_dir: str, supported_extensions: list[str] = None) -> dict:
    """Estimate tokens in source documents.

    Returns dict with:
        - total_tokens: Total input tokens
        - file_count: Number of files
        - files: Dict of file -> token count
    """
    if supported_extensions is None:
        supported_extensions = [".md", ".txt", ".pdf"]

    source_path = Path(source_dir)
    if not source_path.exists():
        return {"total_tokens": 0, "file_count": 0, "files": {}}

    files = {}
    total = 0

    for ext in supported_extensions:
        for file_path in source_path.rglob(f"*{ext}"):
            # Skip PDF for now - text extraction needed
            if ext == ".pdf":
                # Rough estimate: ~500 tokens per page, ~2 pages per PDF
                tokens = 1000
            else:
                tokens = count_file_tokens(file_path)
            files[str(file_path)] = tokens
            total += tokens

    return {
        "total_tokens": total,
        "file_count": len(files),
        "files": files,
    }


def estimate_pipeline_tokens(config: "YourbenchConfig") -> dict:
    """Estimate token usage for the full pipeline.

    Returns detailed breakdown of estimated input/output tokens per stage.
    """
    from yourbench.conf.loader import get_enabled_stages

    result = {
        "source_tokens": 0,
        "stages": {},
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
    }

    enabled = get_enabled_stages(config)
    if not enabled:
        return result

    # Estimate source document tokens
    source_dir = config.pipeline.ingestion.source_documents_dir
    exts = config.pipeline.ingestion.supported_file_extensions
    source_info = estimate_source_tokens(source_dir, exts)
    result["source_tokens"] = source_info["total_tokens"]
    result["source_file_count"] = source_info["file_count"]

    source_tokens = result["source_tokens"]
    if source_tokens == 0:
        # Fallback estimate
        source_tokens = 10000

    total_input = 0
    total_output = 0

    # Stage-by-stage estimation
    for stage in enabled:
        stage_est = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        if stage == "ingestion":
            # Ingestion reads files, may use LLM for PDF extraction
            if config.pipeline.ingestion.llm_ingestion:
                stage_est["input_tokens"] = source_tokens
                stage_est["output_tokens"] = source_tokens  # Similar size output
                stage_est["calls"] = source_info.get("file_count", 1)
            else:
                stage_est["note"] = "No LLM calls (text extraction only)"

        elif stage == "summarization":
            # Summarization processes all content
            max_tokens = config.pipeline.summarization.max_tokens
            chunks = max(1, source_tokens // max_tokens)
            stage_est["input_tokens"] = source_tokens + chunks * 500  # prompts
            stage_est["output_tokens"] = chunks * 2000  # summaries
            stage_est["calls"] = chunks

        elif stage == "chunking":
            # Chunking is local, no LLM
            stage_est["note"] = "No LLM calls (local chunking)"

        elif stage == "single_hop_question_generation":
            # Estimate chunks and questions
            chunk_size = config.pipeline.chunking.l_max_tokens
            num_chunks = max(1, source_tokens // chunk_size)
            stage_est["input_tokens"] = num_chunks * (chunk_size + 1000)  # chunk + prompt
            stage_est["output_tokens"] = num_chunks * 1500  # ~3-5 QA pairs per chunk
            stage_est["calls"] = num_chunks

        elif stage == "multi_hop_question_generation":
            # Multi-hop uses chunk combinations
            chunk_size = config.pipeline.chunking.l_max_tokens
            num_chunks = max(1, source_tokens // chunk_size)
            h_min = config.pipeline.chunking.h_min
            h_max = config.pipeline.chunking.h_max
            avg_hops = (h_min + h_max) // 2
            combinations = min(num_chunks, 20)  # Estimate combinations
            stage_est["input_tokens"] = combinations * (chunk_size * avg_hops + 1000)
            stage_est["output_tokens"] = combinations * 1500
            stage_est["calls"] = combinations

        elif stage == "cross_document_question_generation":
            # Cross-doc uses document combinations
            max_combos = config.pipeline.cross_document_question_generation.max_combinations
            chunk_size = config.pipeline.chunking.l_max_tokens
            docs_per_combo = sum(config.pipeline.cross_document_question_generation.num_docs_per_combination) // 2
            stage_est["input_tokens"] = max_combos * (chunk_size * docs_per_combo + 1000)
            stage_est["output_tokens"] = max_combos * 1500
            stage_est["calls"] = max_combos

        elif stage == "question_rewriting":
            # Rewriting processes all generated questions
            estimated_questions = source_tokens // 500  # Rough estimate
            stage_est["input_tokens"] = estimated_questions * 500
            stage_est["output_tokens"] = estimated_questions * 300
            stage_est["calls"] = estimated_questions

        elif stage == "prepare_lighteval":
            stage_est["note"] = "No LLM calls (formatting only)"

        elif stage == "citation_score_filtering":
            stage_est["note"] = "No LLM calls (filtering only)"

        result["stages"][stage] = stage_est
        total_input += stage_est.get("input_tokens", 0)
        total_output += stage_est.get("output_tokens", 0)

    result["total_input_tokens"] = total_input
    result["total_output_tokens"] = total_output
    result["total_tokens"] = total_input + total_output

    return result


def format_token_count(tokens: int) -> str:
    """Format token count with K/M suffix."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)
