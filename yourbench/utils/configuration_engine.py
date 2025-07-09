"""
Module handles everything related to the configuration of the pipeline.
"""

import os
from typing import Any
from pathlib import Path
from dataclasses import field, fields, dataclass

import yaml
from loguru import logger
from randomname import get_name as get_random_name

from huggingface_hub import whoami


def _expand_env(value: Any) -> Any:
    """
    Replace leading '$VARNAME' with its environment value.
    Special case: if $HF_ORGANIZATION is missing we try HF_TOKEN + whoami().
    """
    if not (isinstance(value, str) and value.startswith("$")):
        return value

    var = value[1:]
    if env := os.getenv(var):
        return env

    # == SPECIAL CASES ==
    if var == "HF_ORGANIZATION":
        token = os.getenv("HF_TOKEN")
        if token:
            try:
                return whoami(token)["name"]
            except Exception:
                logger.warning("Failed to get organization name from HF_TOKEN. Push to hub will fail.")
                pass  # fall through and return literal
    return value


def _expand_dataclass(obj: Any) -> None:
    """In-place $ENV expansion for every str field of a dataclass."""
    for f in fields(obj):
        setattr(obj, f.name, _expand_env(getattr(obj, f.name)))


@dataclass
class HuggingFaceConfig:
    """Configuration for the Hugging Face dataset."""

    hf_dataset_name: str = get_random_name()
    hf_organization: str = "$HF_ORGANIZATION"
    hf_token: str = "$HF_TOKEN"
    private: bool = False
    concat_if_exist: bool = False
    local_dataset_dir: Path | None = Path("data/saved_dataset")
    local_saving: bool = True
    upload_card: bool = True

    def __post_init__(self):
        _expand_dataclass(self)


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = "$HF_TOKEN"
    max_concurrent_requests: int = 32
    encoding_name: str = "cl100k_base"

    # You can find the list of available providers here: https://huggingface.co/docs/huggingface_hub/guides/inference#supported-providers-and-tasks
    # huggingface specific
    provider: str | None = None
    bill_to: str | None = None

    def __post_init__(self):
        _expand_dataclass(self)

        # if base_url is not set, and provider is not set, default to "auto"
        if not self.base_url and not self.provider:
            self.provider = "auto"


@dataclass
class IngestionConfig:
    """Configuration for the ingestion stage"""

    run: bool = False
    source_documents_dir: Path | None = Path("data/raw")
    output_dir: Path | None = Path("data/processed")
    upload_to_hub: bool = True
    llm_ingestion: bool = False
    pdf_dpi: int = 300

    def __post_init__(self):
        # convert string directories to Path objects
        self.source_documents_dir = Path(self.source_documents_dir)
        self.output_dir = Path(self.output_dir)

        if not self.source_documents_dir or not self.output_dir:
            logger.error("Missing source or output director. Creating default directories.")
            raise ValueError("Missing source or output directory")


@dataclass
class SummarizationConfig:
    """Configuration for the summarization stage"""

    run: bool = False


@dataclass
class ChunkingConfig:
    """Configuration for the chunking stage"""

    run: bool = False


@dataclass
class QuestionGenerationConfig:
    """Configuration for the question generation stage"""

    run: bool = False


@dataclass
class SingleShotQuestionGenerationConfig:
    """Configuration for the single shot question generation stage"""

    run: bool = False


@dataclass
class MultiHopQuestionGenerationConfig:
    """Configuration for the multi hop question generation stage"""

    run: bool = False


@dataclass
class QuestionRewritingConfig:
    """Configuration for the question rewriting stage"""

    run: bool = False


@dataclass
class LightevalConfig:
    """Configuration for the lighteval stage"""

    run: bool = False


@dataclass
class CitationScoreFilteringConfig:
    """Configuration for the citation score filtering stage"""

    run: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    question_generation: QuestionGenerationConfig = field(default_factory=QuestionGenerationConfig)
    single_shot_question_generation: SingleShotQuestionGenerationConfig = field(
        default_factory=SingleShotQuestionGenerationConfig
    )
    multi_hop_question_generation: MultiHopQuestionGenerationConfig = field(
        default_factory=MultiHopQuestionGenerationConfig
    )
    question_rewriting: QuestionRewritingConfig = field(default_factory=QuestionRewritingConfig)
    lighteval: LightevalConfig = field(default_factory=LightevalConfig)
    citation_score_filtering: CitationScoreFilteringConfig = field(default_factory=CitationScoreFilteringConfig)


@dataclass
class YourbenchConfig:
    """The main configuration class for the YourBench pipeline."""

    hf_configuration: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    model_list: list[ModelConfig] = field(default_factory=list)
    model_roles: dict[str, list[str]] = field(default_factory=dict)
    debug: bool = False

    def __post_init__(self):
        """Assign default model roles for each pipeline stage if not specified."""
        if not self.model_list:
            return

        # Get the first model name as default
        default_model = self.model_list[0].model_name
        if not default_model:
            return

        # All pipeline stages that can use models
        pipeline_stages = [
            "ingestion",
            "summarization",
            "chunking",
            "question_generation",
            "single_shot_question_generation",
            "multi_hop_question_generation",
            "question_rewriting",
            "lighteval",
            "citation_score_filtering",
        ]

        # Assign default model to stages that don't have model roles defined
        for stage in pipeline_stages:
            if stage not in self.model_roles:
                self.model_roles[stage] = [default_model]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "YourbenchConfig":
        """
        Load YAML → dict → dataclass, with env-var expansion
        confined to HuggingFaceConfig.__post_init__.
        """
        with open(Path(path), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        hf_kwargs = data.get("hf_configuration", {})

        # Handle both 'models' and 'model_list' keys for backward compatibility
        model_list = data.get("model_list", data.get("models", []))
        model_roles = data.get("model_roles", {})

        # Handle pipeline configuration with proper nested dataclass instantiation
        pipeline_data = data.get("pipeline", {})
        pipeline_kwargs = {}

        # Map stage names to their corresponding config classes
        stage_config_classes = {
            "ingestion": IngestionConfig,
            "summarization": SummarizationConfig,
            "chunking": ChunkingConfig,
            "question_generation": QuestionGenerationConfig,
            "single_shot_question_generation": SingleShotQuestionGenerationConfig,
            "multi_hop_question_generation": MultiHopQuestionGenerationConfig,
            "question_rewriting": QuestionRewritingConfig,
            "lighteval": LightevalConfig,
            "citation_score_filtering": CitationScoreFilteringConfig,
        }

        # Convert each stage configuration dict to its dataclass instance
        for stage_name, config_data in pipeline_data.items():
            if stage_name in stage_config_classes:
                config_class = stage_config_classes[stage_name]
                pipeline_kwargs[stage_name] = config_class(**config_data)
            else:
                logger.warning(f"Unknown pipeline stage: {stage_name}")

        return cls(
            hf_configuration=HuggingFaceConfig(**hf_kwargs),
            model_list=[ModelConfig(**m) for m in model_list],
            model_roles=model_roles,
            pipeline_config=PipelineConfig(**pipeline_kwargs),
        )


if __name__ == "__main__":
    config = YourbenchConfig.from_yaml("example/configs/simple_example.yaml")
    print(config)
