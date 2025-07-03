"""
Configuration schema definitions for YourBench.

This module centralizes all configuration schema definitions to ensure
consistency between the CLI generator, documentation, and core pipeline code.
"""

from __future__ import annotations
from typing import Any
from dataclasses import field, dataclass


@dataclass
class GlobalSettings:
    """Global settings that apply across the entire application."""

    debug: bool = False


@dataclass
class HuggingFaceConfig:
    """Settings for integration with the Hugging Face Hub."""

    token: str = "$HF_TOKEN"
    hf_organization: str = "$HF_ORGANIZATION"
    private: bool = True
    hf_dataset_name: str = "yourbench_dataset"
    concat_if_exist: bool = False


@dataclass
class ModelConfig:
    """Definition of a model available for use in YourBench."""

    model_name: str
    provider: str | None = None
    api_key: str = "$HF_TOKEN"
    base_url: str | None = None
    max_concurrent_requests: int = 16


@dataclass
class ModelRoles:
    """Assignment of models to specific pipeline stages."""

    ingestion: list[str] = field(default_factory=list)
    summarization: list[str] = field(default_factory=list)
    chunking: list[str] = field(default_factory=list)
    single_shot_question_generation: list[str] = field(default_factory=list)
    multi_hop_question_generation: list[str] = field(default_factory=list)


@dataclass
class IngestionStageConfig:
    """Configuration for the document ingestion stage."""

    run: bool = True
    source_documents_dir: str = "data/raw"
    output_dir: str = "data/processed"


@dataclass
class UploadToHubStageConfig:
    """Configuration for the upload to Hub stage."""

    run: bool = True
    source_documents_dir: str = "data/processed"


@dataclass
class SummarizationStageConfig:
    """Configuration for the summarization stage."""

    run: bool = True
    max_tokens: int = 4096
    token_overlap: int = 100
    encoding_name: str = "cl100k_base"


@dataclass
class ChunkSamplingConfig:
    """Configuration for chunk sampling in question generation stages."""

    mode: str = "all"  # "all", "count", "percentage"
    random_seed: int = 42
    value: int | float | None = None  # Used for "count" and "percentage" modes


@dataclass
class ChunkingConfig:
    """Configuration for the chunking stage."""

    chunking_mode: str = "fast_chunking"  # "fast_chunking" or "semantic_chunking"
    l_max_tokens: int = 128
    token_overlap: int = 0
    encoding_name: str = "cl100k_base"
    l_min_tokens: int = 64  # Only for semantic_chunking
    tau_threshold: float = 0.8  # Only for semantic_chunking
    h_min: int = 2
    h_max: int = 3
    num_multihops_factor: int = 3


@dataclass
class ChunkingStageConfig:
    """Configuration for the chunking stage."""

    run: bool = True
    chunking_configuration: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass
class QuestionGenerationStageConfig:
    """Configuration for question generation stages."""

    run: bool = True
    additional_instructions: str = "Generate questions to test a curious adult"
    chunk_sampling: ChunkSamplingConfig = field(default_factory=ChunkSamplingConfig)


@dataclass
class SimpleStageConfig:
    """Configuration for simple on/off stages."""

    run: bool = True


@dataclass
class PipelineConfig:
    """Configuration for the stages of the YourBench pipeline."""

    ingestion: IngestionStageConfig = field(default_factory=IngestionStageConfig)
    upload_ingest_to_hub: UploadToHubStageConfig = field(default_factory=UploadToHubStageConfig)
    summarization: SummarizationStageConfig = field(default_factory=SummarizationStageConfig)
    chunking: ChunkingStageConfig = field(default_factory=ChunkingStageConfig)
    single_shot_question_generation: QuestionGenerationStageConfig = field(
        default_factory=QuestionGenerationStageConfig
    )
    multi_hop_question_generation: QuestionGenerationStageConfig = field(default_factory=QuestionGenerationStageConfig)
    lighteval: SimpleStageConfig = field(default_factory=SimpleStageConfig)
    citation_score_filtering: SimpleStageConfig = field(default_factory=SimpleStageConfig)


@dataclass
class YourBenchConfig:
    """Complete YourBench configuration schema."""

    settings: GlobalSettings = field(default_factory=GlobalSettings)
    hf_configuration: HuggingFaceConfig | None = None
    local_dataset_dir: str | None = None
    model_list: list[ModelConfig] = field(default_factory=list)
    model_roles: ModelRoles = field(default_factory=ModelRoles)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary format for YAML serialization."""
        result = {}

        # Settings
        result["settings"] = {"debug": self.settings.debug}

        # HF configuration or local directory
        if self.hf_configuration:
            result["hf_configuration"] = {
                "token": self.hf_configuration.token,
                "hf_organization": self.hf_configuration.hf_organization,
                "private": self.hf_configuration.private,
                "hf_dataset_name": self.hf_configuration.hf_dataset_name,
                "concat_if_exist": self.hf_configuration.concat_if_exist,
            }
        elif self.local_dataset_dir:
            result["local_dataset_dir"] = self.local_dataset_dir

        # Model list
        result["model_list"] = []
        for model in self.model_list:
            model_dict = {
                "model_name": model.model_name,
                "provider": model.provider,
                "api_key": model.api_key,
                "max_concurrent_requests": model.max_concurrent_requests,
            }
            if model.base_url:
                model_dict["base_url"] = model.base_url
            result["model_list"].append(model_dict)

        # Model roles
        result["model_roles"] = {
            "ingestion": self.model_roles.ingestion,
            "summarization": self.model_roles.summarization,
            "chunking": self.model_roles.chunking,
            "single_shot_question_generation": self.model_roles.single_shot_question_generation,
            "multi_hop_question_generation": self.model_roles.multi_hop_question_generation,
        }

        # Pipeline configuration
        pipeline_dict = {}

        # Ingestion
        pipeline_dict["ingestion"] = {
            "run": self.pipeline.ingestion.run,
            "source_documents_dir": self.pipeline.ingestion.source_documents_dir,
            "output_dir": self.pipeline.ingestion.output_dir,
        }

        # Upload to Hub
        pipeline_dict["upload_ingest_to_hub"] = {
            "run": self.pipeline.upload_ingest_to_hub.run,
            "source_documents_dir": self.pipeline.upload_ingest_to_hub.source_documents_dir,
        }

        # Summarization
        pipeline_dict["summarization"] = {
            "run": self.pipeline.summarization.run,
            "max_tokens": self.pipeline.summarization.max_tokens,
            "token_overlap": self.pipeline.summarization.token_overlap,
            "encoding_name": self.pipeline.summarization.encoding_name,
        }

        # Chunking
        chunking_config = {
            "chunking_mode": self.pipeline.chunking.chunking_configuration.chunking_mode,
            "l_max_tokens": self.pipeline.chunking.chunking_configuration.l_max_tokens,
            "token_overlap": self.pipeline.chunking.chunking_configuration.token_overlap,
            "encoding_name": self.pipeline.chunking.chunking_configuration.encoding_name,
            "h_min": self.pipeline.chunking.chunking_configuration.h_min,
            "h_max": self.pipeline.chunking.chunking_configuration.h_max,
            "num_multihops_factor": self.pipeline.chunking.chunking_configuration.num_multihops_factor,
        }

        if self.pipeline.chunking.chunking_configuration.chunking_mode == "semantic_chunking":
            chunking_config.update({
                "l_min_tokens": self.pipeline.chunking.chunking_configuration.l_min_tokens,
                "tau_threshold": self.pipeline.chunking.chunking_configuration.tau_threshold,
            })

        pipeline_dict["chunking"] = {
            "run": self.pipeline.chunking.run,
            "chunking_configuration": chunking_config,
        }

        # Question generation stages
        for stage_name in ["single_shot_question_generation", "multi_hop_question_generation"]:
            stage_config = getattr(self.pipeline, stage_name)
            chunk_sampling = {
                "mode": stage_config.chunk_sampling.mode,
                "random_seed": stage_config.chunk_sampling.random_seed,
            }
            if stage_config.chunk_sampling.value is not None:
                chunk_sampling["value"] = stage_config.chunk_sampling.value

            pipeline_dict[stage_name] = {
                "run": stage_config.run,
                "additional_instructions": stage_config.additional_instructions,
                "chunk_sampling": chunk_sampling,
            }

        # Simple stages
        pipeline_dict["lighteval"] = {"run": self.pipeline.lighteval.run}
        pipeline_dict["citation_score_filtering"] = {"run": self.pipeline.citation_score_filtering.run}

        result["pipeline"] = pipeline_dict
        return result


# Schema metadata for CLI generation
SCHEMA_METADATA = {
    "providers": ["null", "hf-inference", "novita", "together"],
    "chunking_modes": ["fast_chunking", "semantic_chunking"],
    "sampling_modes": ["all", "count", "percentage"],
    "encoding_names": ["cl100k_base", "gpt2"],
    "roles": [
        ("ingestion", "Document ingestion (vision-capable model recommended)"),
        ("summarization", "Document summarization"),
        ("chunking", "Semantic chunking (embedding model recommended)"),
        ("single_shot_question_generation", "Single-shot question generation"),
        ("multi_hop_question_generation", "Multi-hop question generation"),
    ],
}
