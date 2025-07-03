"""
Configuration Generator for YourBench

This module provides utilities for generating YourBench configuration files
through an interactive CLI interface, similar to accelerate config.
Uses the centralized schema from config_schema.py to ensure consistency.
"""

import os
from pathlib import Path

import yaml
import typer
from loguru import logger
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.console import Console

from yourbench.utils.config_schema import (
    SCHEMA_METADATA,
    ModelRoles,
    ModelConfig,
    ChunkingConfig,
    GlobalSettings,
    PipelineConfig,
    YourBenchConfig,
    HuggingFaceConfig,
    SimpleStageConfig,
    ChunkingStageConfig,
    ChunkSamplingConfig,
    IngestionStageConfig,
    UploadToHubStageConfig,
    SummarizationStageConfig,
    QuestionGenerationStageConfig,
)


console = Console()


def generate_config() -> YourBenchConfig:
    """
    Generate a YourBench configuration through an interactive CLI.

    Returns:
        YourBenchConfig: The generated configuration object.
    """
    # Welcome message
    console.print(
        Panel.fit(
            "[bold blue]Welcome to the YourBench Configuration Generator![/bold blue]\n\n"
            "This tool will help you create a configuration file for YourBench by asking "
            "a series of questions about your setup and preferences.\n\n"
            "Press Ctrl+C at any time to exit."
        )
    )

    # Global settings
    console.print("\n[bold]Global Settings[/bold]")
    settings = GlobalSettings(debug=Confirm.ask("Enable debug mode?", default=False))

    # Hugging Face configuration
    console.print("\n[bold]Hugging Face Configuration[/bold]")
    use_hf = Confirm.ask("Do you want to use Hugging Face Hub for dataset storage?", default=True)

    hf_config = None
    local_dataset_dir = None

    if use_hf:
        hf_token = Prompt.ask(
            "Enter your Hugging Face token (or leave empty to use $HF_TOKEN env variable)", default="$HF_TOKEN"
        )
        hf_org = Prompt.ask(
            "Enter your Hugging Face organization/username (or leave empty to use $HF_ORGANIZATION env variable)",
            default="$HF_ORGANIZATION",
        )
        hf_dataset_name = Prompt.ask(
            "Enter the name for your dataset on Hugging Face Hub", default="yourbench_dataset"
        )
        is_private = Confirm.ask("Should the dataset be private?", default=True)
        concat_if_exist = Confirm.ask("Concatenate with existing dataset if it exists?", default=False)

        hf_config = HuggingFaceConfig(
            token=hf_token,
            hf_organization=hf_org,
            private=is_private,
            hf_dataset_name=hf_dataset_name,
            concat_if_exist=concat_if_exist,
        )
    else:
        local_dataset_dir = Prompt.ask(
            "Enter the local directory path for dataset storage", default=str(Path.home() / "yourbench_datasets")
        )

    # Model configuration
    console.print("\n[bold]Model Configuration[/bold]")
    model_list = []

    while True:
        console.print("\n[bold cyan]Add a model:[/bold cyan]")
        model_name = Prompt.ask("Model name (e.g., gpt-4o, Qwen/Qwen2.5-72B-Instruct)")

        provider_options = SCHEMA_METADATA["providers"] + ["other"]
        provider_idx = (
            typer.prompt(
                "Select provider (or enter 'other' to specify):\n"
                + "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(provider_options)])
                + "\nEnter number",
                type=int,
                default=1,
            )
            - 1
        )

        provider = provider_options[provider_idx]
        if provider == "other":
            provider = Prompt.ask("Enter custom provider name")
        elif provider == "null":
            provider = None

        api_key = Prompt.ask(
            "API key (or environment variable, e.g., $OPENAI_API_KEY)",
            default="$HF_TOKEN" if provider in ["hf-inference", "novita", "together"] else None,
        )

        base_url = None
        if provider is None:  # OpenAI or compatible
            base_url = Prompt.ask(
                "Base URL (leave empty for default)",
                default="https://api.openai.com/v1" if "gpt" in model_name.lower() else None,
            )

        max_concurrent = int(Prompt.ask("Maximum concurrent requests", default="16"))

        model_config = ModelConfig(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            max_concurrent_requests=max_concurrent,
        )
        model_list.append(model_config)

        if not Confirm.ask("Add another model?", default=False):
            break

    # Model roles
    console.print("\n[bold]Model Roles[/bold]")
    available_models = [model.model_name for model in model_list]
    model_roles_dict = {}

    for role_key, role_desc in SCHEMA_METADATA["roles"]:
        console.print(f"\n[bold cyan]{role_desc}[/bold cyan]")

        for i, model in enumerate(available_models):
            console.print(f"{i + 1}. {model}")

        selection = Prompt.ask(
            "Enter model number(s) to use for this role (comma-separated, or 'all' for all models)", default="1"
        )

        if selection.lower() == "all":
            model_roles_dict[role_key] = available_models.copy()
        else:
            try:
                indices = [int(idx.strip()) - 1 for idx in selection.split(",")]
                selected_models = [available_models[idx] for idx in indices if 0 <= idx < len(available_models)]
                model_roles_dict[role_key] = selected_models
            except (ValueError, IndexError):
                console.print("[bold red]Invalid selection. Using the first model.[/bold red]")
                model_roles_dict[role_key] = [available_models[0]]

    model_roles = ModelRoles(**model_roles_dict)

    # Pipeline configuration
    console.print("\n[bold]Pipeline Configuration[/bold]")

    # Ingestion stage
    console.print("\n[bold cyan]Document Ingestion Stage[/bold cyan]")
    if Confirm.ask("Enable document ingestion stage?", default=True):
        source_dir = Prompt.ask("Source documents directory", default="data/raw")
        output_dir = Prompt.ask("Output directory for processed documents", default="data/processed")
        ingestion_config = IngestionStageConfig(run=True, source_documents_dir=source_dir, output_dir=output_dir)
    else:
        ingestion_config = IngestionStageConfig(run=False)

    # Upload to Hub stage
    console.print("\n[bold cyan]Upload to Hub Stage[/bold cyan]")
    if Confirm.ask("Enable upload to Hub stage?", default=True):
        source_dir = ingestion_config.output_dir if ingestion_config.run else "data/processed"
        source_dir = Prompt.ask("Source documents directory", default=source_dir)
        upload_config = UploadToHubStageConfig(run=True, source_documents_dir=source_dir)
    else:
        upload_config = UploadToHubStageConfig(run=False)

    # Summarization stage
    console.print("\n[bold cyan]Summarization Stage[/bold cyan]")
    if Confirm.ask("Enable summarization stage?", default=True):
        max_tokens = int(Prompt.ask("Maximum tokens per chunk", default="4096"))
        token_overlap = int(Prompt.ask("Token overlap between chunks", default="100"))
        encoding_name = Prompt.ask("Encoding name (cl100k_base for GPT/Qwen, gpt2 for others)", default="cl100k_base")
        summarization_config = SummarizationStageConfig(
            run=True, max_tokens=max_tokens, token_overlap=token_overlap, encoding_name=encoding_name
        )
    else:
        summarization_config = SummarizationStageConfig(run=False)

    # Chunking stage
    console.print("\n[bold cyan]Chunking Stage[/bold cyan]")
    if Confirm.ask("Enable chunking stage?", default=True):
        chunking_mode_options = SCHEMA_METADATA["chunking_modes"]
        chunking_mode_idx = (
            typer.prompt(
                "Select chunking mode:\n1. fast_chunking (token-based)\n2. semantic_chunking (embedding-based)\nEnter number",
                type=int,
                default=1,
            )
            - 1
        )
        chunking_mode = chunking_mode_options[chunking_mode_idx]

        l_max_tokens = int(Prompt.ask("Maximum tokens per chunk", default="128"))
        token_overlap = int(Prompt.ask("Token overlap between chunks", default="0"))
        encoding_name = Prompt.ask("Encoding name", default="cl100k_base")

        chunking_inner_config = ChunkingConfig(
            chunking_mode=chunking_mode,
            l_max_tokens=l_max_tokens,
            token_overlap=token_overlap,
            encoding_name=encoding_name,
        )

        if chunking_mode == "semantic_chunking":
            l_min_tokens = int(Prompt.ask("Minimum tokens per chunk", default="64"))
            tau_threshold = float(Prompt.ask("Similarity threshold (tau)", default="0.8"))
            chunking_inner_config.l_min_tokens = l_min_tokens
            chunking_inner_config.tau_threshold = tau_threshold

        h_min = int(Prompt.ask("Minimum hops (h_min)", default="2"))
        h_max = int(Prompt.ask("Maximum hops (h_max)", default="3"))
        num_multihops_factor = int(Prompt.ask("Multi-hop factor", default="3"))

        chunking_inner_config.h_min = h_min
        chunking_inner_config.h_max = h_max
        chunking_inner_config.num_multihops_factor = num_multihops_factor

        chunking_stage_config = ChunkingStageConfig(run=True, chunking_configuration=chunking_inner_config)
    else:
        chunking_stage_config = ChunkingStageConfig(run=False)

    # Question generation stages
    def create_question_generation_config(stage_name: str) -> QuestionGenerationStageConfig:
        console.print(f"\n[bold cyan]{stage_name.replace('_', ' ').title()} Stage[/bold cyan]")
        if Confirm.ask(f"Enable {stage_name.replace('_', ' ')} stage?", default=True):
            additional_instructions = Prompt.ask(
                "Additional instructions for question generation", default="Generate questions to test a curious adult"
            )

            sampling_mode_options = SCHEMA_METADATA["sampling_modes"]
            sampling_mode_idx = (
                typer.prompt(
                    "Select chunk sampling mode:\n1. all (use all chunks)\n2. count (fixed number of chunks)\n3. percentage (percentage of chunks)\nEnter number",
                    type=int,
                    default=1,
                )
                - 1
            )
            sampling_mode = sampling_mode_options[sampling_mode_idx]
            random_seed = int(Prompt.ask("Random seed for sampling", default="42"))

            chunk_sampling_config = ChunkSamplingConfig(mode=sampling_mode, random_seed=random_seed)
            if sampling_mode == "count":
                sampling_value = int(Prompt.ask("Number of chunks to sample", default="5"))
                chunk_sampling_config.value = sampling_value
            elif sampling_mode == "percentage":
                sampling_value = float(Prompt.ask("Percentage of chunks to sample (0.0-1.0)", default="0.3"))
                chunk_sampling_config.value = sampling_value

            return QuestionGenerationStageConfig(
                run=True, additional_instructions=additional_instructions, chunk_sampling=chunk_sampling_config
            )
        else:
            return QuestionGenerationStageConfig(run=False)

    single_shot_config = create_question_generation_config("single_shot_question_generation")
    multi_hop_config = create_question_generation_config("multi_hop_question_generation")

    # Simple stages
    console.print("\n[bold cyan]LightEval Stage[/bold cyan]")
    lighteval_config = SimpleStageConfig(run=Confirm.ask("Enable LightEval stage?", default=True))

    console.print("\n[bold cyan]Citation Score Filtering Stage[/bold cyan]")
    citation_config = SimpleStageConfig(run=Confirm.ask("Enable citation score filtering stage?", default=True))

    # Build complete configuration
    pipeline_config = PipelineConfig(
        ingestion=ingestion_config,
        upload_ingest_to_hub=upload_config,
        summarization=summarization_config,
        chunking=chunking_stage_config,
        single_shot_question_generation=single_shot_config,
        multi_hop_question_generation=multi_hop_config,
        lighteval=lighteval_config,
        citation_score_filtering=citation_config,
    )

    return YourBenchConfig(
        settings=settings,
        hf_configuration=hf_config,
        local_dataset_dir=local_dataset_dir,
        model_list=model_list,
        model_roles=model_roles,
        pipeline=pipeline_config,
    )


def save_config(config: YourBenchConfig, output_path: str) -> None:
    """
    Save the configuration object to a YAML file.

    Args:
        config (YourBenchConfig): The configuration object.
        output_path (str): The path where to save the configuration file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert to dictionary and save as YAML
    config_dict = config.to_dict()
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.success(f"Configuration saved to {output_path}")


def generate_and_save_config(output_path: str) -> None:
    """
    Generate a configuration through the interactive CLI and save it to a file.

    Args:
        output_path (str): The path where to save the configuration file.
    """
    try:
        config = generate_config()
        save_config(config, output_path)

        console.print(
            Panel.fit(
                f"[bold green]Configuration successfully generated and saved to:[/bold green]\n"
                f"[bold]{output_path}[/bold]\n\n"
                f"You can now run YourBench with this configuration using:\n"
                f"[bold]yourbench run --config {output_path}[/bold]"
            )
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Configuration generation cancelled.[/bold yellow]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]Error generating configuration: {str(e)}[/bold red]")
        logger.exception(f"Configuration generation failed: {e}")
        raise typer.Exit(code=1)
