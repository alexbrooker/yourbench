"""Upload ingested dataset to HuggingFace Hub."""

from loguru import logger

from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.configuration_engine import YourbenchConfig


def run(config: YourbenchConfig) -> None:
    """
    Upload the ingested dataset to HuggingFace Hub or save locally.

    This is a separate stage from ingestion to allow saving the processed
    dataset independently. It loads the ingested dataset and saves it
    using the standard dataset saving mechanism.
    """
    stage_cfg = config.pipeline_config.upload_ingest_to_hub
    if not stage_cfg.run:
        logger.info("upload_ingest_to_hub stage is disabled. Skipping.")
        return

    logger.info("Loading ingested dataset for upload...")

    try:
        # Load the ingested dataset
        dataset = custom_load_dataset(config=config, subset="ingested")

        if not dataset or len(dataset) == 0:
            logger.warning("No ingested dataset found or dataset is empty. Nothing to upload.")
            return

        logger.info(f"Found {len(dataset)} documents to upload")

        # Save the dataset (this handles both local saving and hub upload)
        custom_save_dataset(dataset=dataset, config=config, subset="ingested", save_local=True, push_to_hub=True)

        logger.success("Successfully uploaded ingested dataset")

    except Exception as e:
        logger.error(f"Failed to upload ingested dataset: {e}")
        raise
