import io
import uuid
import base64
from pathlib import Path

import fitz
import trafilatura
from PIL import Image
from loguru import logger
from markitdown import MarkItDown

from datasets import Dataset
from huggingface_hub import InferenceClient
from yourbench.utils.dataset_engine import custom_save_dataset
from yourbench.utils.configuration_engine import YourbenchConfig
from yourbench.utils.inference.inference_core import (
    InferenceCall,
    _load_models,
    run_inference,
)


def run(config: YourbenchConfig) -> None:
    """Convert documents to markdown and optionally upload to Hub."""
    source_dir = config.pipeline_config.ingestion.source_documents_dir
    output_dir = config.pipeline_config.ingestion.output_dir

    # Process files
    processor = _get_processor(config)
    files_processed = 0
    successful_outputs = []

    for file_path in source_dir.rglob("*"):
        if file_path.is_file():
            try:
                if content := _convert_file(file_path, config, processor):
                    # Preserve relative path to avoid filename collisions
                    relative_path = file_path.relative_to(source_dir)
                    output_path = output_dir / f"{relative_path.with_suffix('.md')}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(content, encoding="utf-8")
                    logger.debug(f"Converted {file_path.name} → {output_path.name}")
                    successful_outputs.append(output_path)
                    files_processed += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue

    logger.info(f"Processed {files_processed} files")

    # Upload to hub if configured - only upload successfully converted files
    if config.pipeline_config.ingestion.upload_to_hub and successful_outputs:
        _upload_to_hub(config, successful_outputs)


def _get_processor(config: YourbenchConfig) -> MarkItDown:
    """Initialize markdown processor with optional LLM support."""
    model = config.model_list[0] # choose the first model in the list. TODO: add support for model choice.
    if not config.pipeline_config.ingestion.llm_ingestion or not model:
        return MarkItDown()

    try:
        client = InferenceClient(
            base_url=model.base_url,
            api_key=model.api_key,
        )
        logger.debug(f"Using LLM: {model.model_name}")
        return MarkItDown(llm_client=client, llm_model=model.model_name)
    except Exception as e:
        logger.warning(f"Failed to init LLM processor: {e}")
        return MarkItDown()


def _convert_file(file_path: Path, config: YourbenchConfig, processor: MarkItDown) -> str | None:
    """Convert file to markdown based on type."""
    supported_extensions = {
        ".md",
        ".txt",
        ".html",
        ".htm",
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
        ".rtf",
        ".odt",
    }

    file_ext = file_path.suffix.lower()

    if file_ext not in supported_extensions:
        logger.warning(f"Unsupported file type: {file_ext} for file {file_path.name}")
        return None

    match file_ext:
        case ".md":
            return file_path.read_text(encoding="utf-8")

        case ".html" | ".htm":
            if content := _extract_html(file_path):
                return content
            # Fallback to MarkItDown
            return processor.convert(str(file_path)).text_content

        case ".pdf" if getattr(config.pipeline_config.ingestion, "llm_ingestion", False):
            return _process_pdf_llm(file_path, config)

        case _:
            return processor.convert(str(file_path)).text_content


def _extract_html(path: Path) -> str | None:
    """Extract markdown from HTML using trafilatura."""
    try:
        html = path.read_text(encoding="utf-8")
        return trafilatura.extract(html, output_format="markdown", include_comments=False, include_tables=True)
    except Exception as e:
        logger.debug(f"Trafilatura failed for {path.name}: {e}")
        return None


def _process_pdf_llm(pdf_path: Path, config: YourbenchConfig) -> str:
    """Convert every page of a PDF to Markdown using an LLM."""
    from dataclasses import asdict
    
    # Convert YourbenchConfig to dict for inference functions
    config_dict = asdict(config)
    
    # Handle case where YAML has 'models' instead of 'model_list'
    if not config_dict.get("model_list") and config.model_list:
        config_dict["model_list"] = [asdict(model) for model in config.model_list]
    
    models = _load_models(config_dict, "ingestion")
    
    # If no models found but we have models in config, use the first one
    if not models and config.model_list:
        logger.info(f"No models configured for ingestion role, using first available model: {config.model_list[0].model_name}")
        first_model_dict = asdict(config.model_list[0])
        config_dict["model_list"] = [first_model_dict]
        models = _load_models(config_dict, "ingestion")
    
    if not models:
        logger.warning(f"No LLM models configured for PDF ingestion of {pdf_path.name}, falling back to MarkItDown")
        try:
            return MarkItDown().convert(str(pdf_path)).text_content
        except Exception as exc:
            logger.error(f"Fallback conversion failed for {pdf_path.name}: {exc}")
            return ""

    dpi = getattr(config.pipeline_config.ingestion, "pdf_dpi", 300)
    images = _pdf_to_images(pdf_path, dpi)
    if not images:
        return ""

    calls = [
        InferenceCall(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Convert this document page to clean Markdown. "
                                "Preserve all text, structure, tables, and formatting. "
                                "Output only the content in Markdown."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_img_to_b64(img)}"},
                        },
                    ],
                }
            ],
            tags=["pdf_ingestion", f"page_{idx + 1}", pdf_path.name],
        )
        for idx, img in enumerate(images)
    ]

    pages: list[str] = []
    responses = run_inference(config_dict, "ingestion", calls)
    if responses:
        model_name = next(iter(responses))
        pages.extend(responses[model_name])

    return "\n\n---\n\n".join(filter(None, pages))


def _pdf_to_images(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """Convert PDF pages to images."""
    try:
        with fitz.open(pdf_path) as doc:
            images = []
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                images.append(img)
        return images
    except Exception as e:
        logger.error(f"Failed to convert {pdf_path.name}: {e}")
        return []


def _img_to_b64(image: Image.Image) -> str:
    """Convert PIL image to base64."""
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


def _upload_to_hub(config: YourbenchConfig, md_files: list[Path]):
    """Upload markdown files to Hugging Face Hub."""
    if not md_files:
        logger.warning("No markdown files to upload")
        return

    docs = []
    for path in md_files:
        try:
            if content := path.read_text(encoding="utf-8").strip():
                docs.append({
                    "document_id": str(uuid.uuid4()),
                    "document_text": content,
                    "document_filename": path.name,
                    "document_metadata": {"file_size": path.stat().st_size},
                })
        except Exception as e:
            logger.error(f"Failed to read {path.name} for upload: {e}")
            continue

    if not docs:
        logger.warning("No valid documents to upload")
        return

    dataset = Dataset.from_list(docs)
    custom_save_dataset(dataset, config, subset="ingested")
    logger.info(f"Uploaded {len(docs)} documents to Hub")
