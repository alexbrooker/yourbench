import os
import glob
from typing import Dict, Any

from loguru import logger
from markitdown import MarkItDown

def run(config: Dict[str, Any]) -> None:
    """
    Execute the ingestion stage of the pipeline.

    This function checks the pipeline configuration for the ingestion stage. If the user
    has enabled the ingestion stage (run == True), it proceeds to:
      1. Read the source_documents_dir for all files.
      2. For each file, convert it to Markdown using MarkItDown.
      3. Save the output (.md) to the output_dir.

    Parameters:
        config (Dict[str, Any]): A dictionary containing overall pipeline configuration.
    Returns:
        None
    """
    # Validate and extract ingestion configuration 
    logger.debug(f"Ingestion config: {config.get('pipeline', {}).get('ingestion', {})}")
    ingestion_cfg = config.get("pipeline", {}).get("ingestion", {})
    if not isinstance(ingestion_cfg, dict):
        logger.error("Ingestion config is missing or incorrectly formatted.")
        return

    if not ingestion_cfg.get("run", False):
        logger.info("Ingestion stage disabled. Skipping.")
        return

    source_dir = ingestion_cfg.get("source_documents_dir")
    output_dir = ingestion_cfg.get("output_dir")
    if not source_dir or not output_dir:
        logger.error("source_documents_dir or output_dir not specified. Cannot proceed.")
        return

    # Prepare output directory 
    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Ensured output directory exists: {}", output_dir)

    # (Optional) Resolve a model for advanced image descriptions, etc. 
    md = MarkItDown()

    # Convert each file in the source directory 
    file_paths = glob.glob(os.path.join(source_dir, "**"), recursive=True)
    if not file_paths:
        logger.warning("No files found in source directory: {}", source_dir)
        return

    logger.info(
        "Starting ingestion: converting files from '{}' to '{}'.",
        source_dir,
        output_dir
    )

    for file in file_paths:
        if os.path.isfile(file):
            _convert_file(file, output_dir, md)

    logger.success("Ingestion complete. Processed files from '{}' to '{}'.", source_dir, output_dir)


def _convert_file(file_path: str, output_dir: str, md: MarkItDown) -> None:
    """
    Convert a single file to Markdown and save the output.

    Parameters:
        file_path (str): Path to the source file to be converted.
        output_dir (str): Directory where the resulting .md file should be saved.
        md (MarkItDown): A configured MarkItDown instance.
    """
    logger.debug("Converting file: {}", file_path)
    try:
        # Convert the file to Markdown
        result = md.convert(file_path)

        # Construct an output filename with .md extension
        base_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{file_name_no_ext}.md")

        # Save the converted text content
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(result.text_content)

        logger.info("Successfully converted '{}' -> '{}'.", file_path, output_file)
    except Exception as exc:
        logger.error("Failed to convert '{}'. Error: {}", file_path, exc)
