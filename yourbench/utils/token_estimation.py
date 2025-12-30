"""Token estimation utilities using tiktoken."""

from typing import TYPE_CHECKING
from pathlib import Path

import tiktoken
from loguru import logger
from markitdown import MarkItDown


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


def _extract_file_content(file_path: Path, processor: MarkItDown) -> str | None:
    """Extract text content from a file using MarkItDown.

    This mirrors the ingestion pipeline logic but without LLM calls.
    """
    file_ext = file_path.suffix.lower()

    try:
        # Handle simple text files directly
        if file_ext == ".md":
            return file_path.read_text(encoding="utf-8")

        if file_ext in {".txt", ".text"}:
            return file_path.read_text(encoding="utf-8")

        if file_ext in {".html", ".htm"}:
            try:
                import trafilatura

                html = file_path.read_text(encoding="utf-8")
                content = trafilatura.extract(
                    html, output_format="markdown", include_comments=False, include_tables=True
                )
                if content:
                    return content
            except Exception:
                pass

        # Use MarkItDown for everything else (PDF, docx, etc.)
        result = processor.convert(str(file_path))
        return result.text_content if result else None

    except Exception as e:
        logger.debug(f"Error extracting content from {file_path}: {e}")
        return None


def run_estimation_ingestion(
    source_dir: str,
    supported_extensions: list[str] | None = None,
    use_llm: bool = False,
) -> dict:
    """Run ingestion without LLM to extract document content for estimation.

    Returns:
        dict with:
            - documents: list of {file_path, content, tokens}
            - total_tokens: sum of all document tokens
            - file_count: number of successfully processed files
            - llm_required: True if LLM ingestion would be needed
    """
    if supported_extensions is None:
        supported_extensions = [".md", ".txt", ".pdf", ".docx", ".html", ".htm"]

    source_path = Path(source_dir)
    if not source_path.exists():
        return {"documents": [], "total_tokens": 0, "file_count": 0, "llm_required": False}

    # Initialize MarkItDown without LLM
    processor = MarkItDown()

    documents = []
    total_tokens = 0
    llm_required = False

    # Collect all matching files
    all_files = []
    for ext in supported_extensions:
        all_files.extend(source_path.rglob(f"*{ext}"))

    for file_path in all_files:
        # Skip files in output directories
        if "output" in str(file_path):
            continue

        # Check if this file type would need LLM ingestion for better quality
        if use_llm and file_path.suffix.lower() == ".pdf":
            llm_required = True

        content = _extract_file_content(file_path, processor)
        if content:
            tokens = count_tokens(content)
            documents.append({
                "file_path": str(file_path),
                "content": content,
                "tokens": tokens,
            })
            total_tokens += tokens

    return {
        "documents": documents,
        "total_tokens": total_tokens,
        "file_count": len(documents),
        "llm_required": llm_required,
    }


def simulate_chunking(documents: list[dict], chunk_max_tokens: int) -> list[dict]:
    """Simulate the chunking process to count actual chunks.

    Returns list of chunks with token counts.
    """
    from yourbench.utils.chunking_utils import split_into_token_chunks

    chunks = []
    for doc in documents:
        content = doc.get("content", "")
        if not content:
            continue

        doc_chunks = split_into_token_chunks(content, chunk_max_tokens, overlap=0)
        for i, chunk_text in enumerate(doc_chunks):
            chunk_tokens = count_tokens(chunk_text)
            chunks.append({
                "doc_path": doc.get("file_path", ""),
                "chunk_index": i,
                "tokens": chunk_tokens,
            })

    return chunks


def estimate_pipeline_tokens(config: "YourbenchConfig") -> dict:
    """Estimate token usage for the full pipeline.

    Runs actual ingestion (non-LLM) and chunking simulation for accurate estimates.
    Returns detailed breakdown of estimated input/output tokens per stage.
    Output tokens are given as a range (25th-75th percentile estimates).
    """
    from yourbench.conf.loader import get_enabled_stages

    result = {
        "source_tokens": 0,
        "source_file_count": 0,
        "num_chunks": 0,
        "stages": {},
        "total_input_tokens": 0,
        "total_output_tokens_low": 0,
        "total_output_tokens_high": 0,
        "total_tokens_low": 0,
        "total_tokens_high": 0,
    }

    enabled = get_enabled_stages(config)
    if not enabled:
        return result

    # Run actual ingestion to get real document content
    source_dir = config.pipeline.ingestion.source_documents_dir
    exts = config.pipeline.ingestion.supported_file_extensions
    use_llm = config.pipeline.ingestion.llm_ingestion

    logger.info(f"Running estimation ingestion on {source_dir}...")
    ingestion_result = run_estimation_ingestion(source_dir, exts, use_llm)

    result["source_tokens"] = ingestion_result["total_tokens"]
    result["source_file_count"] = ingestion_result["file_count"]
    result["llm_ingestion_required"] = ingestion_result["llm_required"]

    source_tokens = result["source_tokens"]
    if source_tokens == 0:
        logger.warning("No content extracted from source documents")
        return result

    # Simulate chunking to get accurate chunk count
    chunk_max_tokens = config.pipeline.chunking.l_max_tokens
    chunks = simulate_chunking(ingestion_result["documents"], chunk_max_tokens)
    num_chunks = len(chunks)
    result["num_chunks"] = num_chunks

    # Calculate multi-hop combinations estimate
    h_min = config.pipeline.chunking.h_min
    h_max = config.pipeline.chunking.h_max
    num_multihops_factor = config.pipeline.chunking.num_multihops_factor
    num_multihop_combos = max(1, num_chunks // max(1, num_multihops_factor))

    total_input = 0
    total_output_low = 0
    total_output_high = 0

    # Stage-by-stage estimation with actual data
    for stage in enabled:
        stage_est = {"input_tokens": 0, "output_tokens_low": 0, "output_tokens_high": 0, "calls": 0}

        if stage == "ingestion":
            if use_llm:
                # LLM ingestion processes each file
                stage_est["input_tokens"] = source_tokens
                # Output is similar to input for ingestion
                stage_est["output_tokens_low"] = int(source_tokens * 0.8)
                stage_est["output_tokens_high"] = int(source_tokens * 1.2)
                stage_est["calls"] = ingestion_result["file_count"]
            else:
                stage_est["note"] = "No LLM calls (text extraction only)"

        elif stage == "summarization":
            max_tokens = config.pipeline.summarization.max_tokens
            summary_chunks = max(1, source_tokens // max_tokens)
            stage_est["input_tokens"] = source_tokens + summary_chunks * 500
            # Summaries are typically 10-30% of input
            stage_est["output_tokens_low"] = int(source_tokens * 0.10)
            stage_est["output_tokens_high"] = int(source_tokens * 0.30)
            stage_est["calls"] = summary_chunks

        elif stage == "chunking":
            stage_est["note"] = "No LLM calls (local chunking)"
            stage_est["chunks_created"] = num_chunks

        elif stage == "single_hop_question_generation":
            avg_chunk_tokens = sum(c["tokens"] for c in chunks) // max(1, num_chunks)
            prompt_overhead = 1000
            stage_est["input_tokens"] = num_chunks * (avg_chunk_tokens + prompt_overhead)
            # Question generation output is 25-75% of chunk content
            base_content = num_chunks * avg_chunk_tokens
            stage_est["output_tokens_low"] = int(base_content * 0.25)
            stage_est["output_tokens_high"] = int(base_content * 0.75)
            stage_est["calls"] = num_chunks

        elif stage == "multi_hop_question_generation":
            avg_hops = (h_min + h_max) // 2
            avg_chunk_tokens = sum(c["tokens"] for c in chunks) // max(1, num_chunks)
            prompt_overhead = 1000
            stage_est["input_tokens"] = num_multihop_combos * (avg_chunk_tokens * avg_hops + prompt_overhead)
            # Multi-hop generates more content: 30-80% of combined chunks
            base_content = num_multihop_combos * avg_chunk_tokens * avg_hops
            stage_est["output_tokens_low"] = int(base_content * 0.30)
            stage_est["output_tokens_high"] = int(base_content * 0.80)
            stage_est["calls"] = num_multihop_combos

        elif stage == "cross_document_question_generation":
            max_combos = config.pipeline.cross_document_question_generation.max_combinations
            docs_range = config.pipeline.cross_document_question_generation.num_docs_per_combination
            docs_per_combo = sum(docs_range) // 2
            avg_chunk_tokens = sum(c["tokens"] for c in chunks) // max(1, num_chunks)
            prompt_overhead = 1000
            stage_est["input_tokens"] = max_combos * (avg_chunk_tokens * docs_per_combo + prompt_overhead)
            # Cross-doc generates 30-80% of combined content
            base_content = max_combos * avg_chunk_tokens * docs_per_combo
            stage_est["output_tokens_low"] = int(base_content * 0.30)
            stage_est["output_tokens_high"] = int(base_content * 0.80)
            stage_est["calls"] = max_combos

        elif stage == "question_rewriting":
            estimated_questions = num_chunks * 3
            stage_est["input_tokens"] = estimated_questions * 500
            # Rewriting output is similar to input: 50-100%
            stage_est["output_tokens_low"] = int(stage_est["input_tokens"] * 0.50)
            stage_est["output_tokens_high"] = int(stage_est["input_tokens"] * 1.00)
            stage_est["calls"] = estimated_questions

        elif stage == "prepare_lighteval":
            stage_est["note"] = "No LLM calls (formatting only)"

        elif stage == "citation_score_filtering":
            stage_est["note"] = "No LLM calls (filtering only)"

        result["stages"][stage] = stage_est
        total_input += stage_est.get("input_tokens", 0)
        total_output_low += stage_est.get("output_tokens_low", 0)
        total_output_high += stage_est.get("output_tokens_high", 0)

    result["total_input_tokens"] = total_input
    result["total_output_tokens_low"] = total_output_low
    result["total_output_tokens_high"] = total_output_high
    result["total_tokens_low"] = total_input + total_output_low
    result["total_tokens_high"] = total_input + total_output_high

    return result


def format_token_count(tokens: int) -> str:
    """Format token count with K/M suffix."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def format_token_range(low: int, high: int) -> str:
    """Format a token range with K/M suffix."""
    return f"{format_token_count(low)} - {format_token_count(high)}"
