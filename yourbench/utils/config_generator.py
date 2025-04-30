"""
Configuration Generator for YourBench

This module provides utilities for generating YourBench configuration files
through an interactive CLI interface, similar to accelerate config.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from loguru import logger
import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich import print as rich_print


console = Console()


def generate_config() -> Dict[str, Any]:
    """
    Generate a YourBench configuration through an interactive CLI.
    
    Returns:
        Dict[str, Any]: The generated configuration dictionary.
    """
    config: Dict[str, Any] = {}
    
    # Welcome message
    console.print(Panel.fit(
        "[bold blue]Welcome to the YourBench Configuration Generator![/bold blue]\n\n"
        "This tool will help you create a configuration file for YourBench by asking "
        "a series of questions about your setup and preferences.\n\n"
        "Press Ctrl+C at any time to exit."
    ))
    
    # Global settings
    console.print("\n[bold]Global Settings[/bold]")
    config["settings"] = {
        "debug": Confirm.ask("Enable debug mode?", default=False)
    }
    
    # Hugging Face configuration
    console.print("\n[bold]Hugging Face Configuration[/bold]")
    use_hf = Confirm.ask("Do you want to use Hugging Face Hub for dataset storage?", default=True)
    
    if use_hf:
        hf_token = Prompt.ask(
            "Enter your Hugging Face token (or leave empty to use $HF_TOKEN env variable)",
            default="$HF_TOKEN"
        )
        
        hf_org = Prompt.ask(
            "Enter your Hugging Face organization/username (or leave empty to use $HF_ORGANIZATION env variable)",
            default="$HF_ORGANIZATION"
        )
        
        hf_dataset_name = Prompt.ask(
            "Enter the name for your dataset on Hugging Face Hub",
            default="yourbench_dataset"
        )
        
        is_private = Confirm.ask("Should the dataset be private?", default=True)
        
        concat_if_exist = Confirm.ask(
            "Concatenate with existing dataset if it exists?", 
            default=False
        )
        
        config["hf_configuration"] = {
            "token": hf_token,
            "hf_organization": hf_org,
            "private": is_private,
            "hf_dataset_name": hf_dataset_name,
            "concat_if_exist": concat_if_exist
        }
    else:
        local_dir = Prompt.ask(
            "Enter the local directory path for dataset storage",
            default=str(Path.home() / "yourbench_datasets")
        )
        config["local_dataset_dir"] = local_dir
    
    # Model configuration
    console.print("\n[bold]Model Configuration[/bold]")
    
    model_list = []
    while True:
        console.print("\n[bold cyan]Add a model:[/bold cyan]")
        
        model_name = Prompt.ask("Model name (e.g., gpt-4o, Qwen/Qwen2.5-72B-Instruct)")
        
        provider_options = ["null", "hf-inference", "novita", "together", "other"]
        provider_idx = typer.prompt(
            "Select provider (or enter 'other' to specify):\n" +
            "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(provider_options)]) +
            "\nEnter number",
            type=int,
            default=1
        ) - 1
        
        provider = provider_options[provider_idx]
        if provider == "other":
            provider = Prompt.ask("Enter custom provider name")
        elif provider == "null":
            provider = None
        
        api_key = Prompt.ask(
            "API key (or environment variable, e.g., $OPENAI_API_KEY)",
            default="$HF_TOKEN" if provider in ["hf-inference", "novita", "together"] else None
        )
        
        base_url = None
        if provider is None:  # OpenAI or compatible
            base_url = Prompt.ask(
                "Base URL (leave empty for default)",
                default="https://api.openai.com/v1" if "gpt" in model_name.lower() else None
            )
        
        max_concurrent = Prompt.ask(
            "Maximum concurrent requests",
            default="16"
        )
        
        model_entry = {
            "model_name": model_name,
            "provider": provider,
            "api_key": api_key,
            "max_concurrent_requests": int(max_concurrent)
        }
        
        if base_url:
            model_entry["base_url"] = base_url
            
        model_list.append(model_entry)
        
        if not Confirm.ask("Add another model?", default=False):
            break
    
    config["model_list"] = model_list
    
    # Model roles
    console.print("\n[bold]Model Roles[/bold]")
    
    available_models = [model["model_name"] for model in model_list]
    
    model_roles = {}
    
    # For each role, ask which models to use
    roles = [
        ("ingestion", "Document ingestion (vision-capable model recommended)"),
        ("summarization", "Document summarization"),
        ("chunking", "Semantic chunking (embedding model recommended)"),
        ("single_shot_question_generation", "Single-shot question generation"),
        ("multi_hop_question_generation", "Multi-hop question generation")
    ]
    
    for role_key, role_desc in roles:
        console.print(f"\n[bold cyan]{role_desc}[/bold cyan]")
        
        # Display available models with numbers
        for i, model in enumerate(available_models):
            console.print(f"{i+1}. {model}")
        
        # Get user selection
        selection = Prompt.ask(
            "Enter model number(s) to use for this role (comma-separated, or 'all' for all models)",
            default="1"
        )
        
        if selection.lower() == "all":
            model_roles[role_key] = available_models
        else:
            try:
                indices = [int(idx.strip()) - 1 for idx in selection.split(",")]
                selected_models = [available_models[idx] for idx in indices if 0 <= idx < len(available_models)]
                model_roles[role_key] = selected_models
            except (ValueError, IndexError):
                console.print("[bold red]Invalid selection. Using the first model.[/bold red]")
                model_roles[role_key] = [available_models[0]]
    
    config["model_roles"] = model_roles
    
    # Pipeline configuration
    console.print("\n[bold]Pipeline Configuration[/bold]")
    
    pipeline_config = {}
    
    # Ingestion stage
    console.print("\n[bold cyan]Document Ingestion Stage[/bold cyan]")
    if Confirm.ask("Enable document ingestion stage?", default=True):
        source_dir = Prompt.ask(
            "Source documents directory",
            default="data/raw"
        )
        output_dir = Prompt.ask(
            "Output directory for processed documents",
            default="data/processed"
        )
        
        pipeline_config["ingestion"] = {
            "run": True,
            "source_documents_dir": source_dir,
            "output_dir": output_dir
        }
    else:
        pipeline_config["ingestion"] = {"run": False}
    
    # Upload to Hub stage
    console.print("\n[bold cyan]Upload to Hub Stage[/bold cyan]")
    if Confirm.ask("Enable upload to Hub stage?", default=True):
        source_dir = pipeline_config.get("ingestion", {}).get("output_dir", "data/processed")
        source_dir = Prompt.ask(
            "Source documents directory",
            default=source_dir
        )
        
        pipeline_config["upload_ingest_to_hub"] = {
            "run": True,
            "source_documents_dir": source_dir
        }
    else:
        pipeline_config["upload_ingest_to_hub"] = {"run": False}
    
    # Summarization stage
    console.print("\n[bold cyan]Summarization Stage[/bold cyan]")
    if Confirm.ask("Enable summarization stage?", default=True):
        max_tokens = Prompt.ask(
            "Maximum tokens per chunk",
            default="4096"
        )
        token_overlap = Prompt.ask(
            "Token overlap between chunks",
            default="100"
        )
        encoding_name = Prompt.ask(
            "Encoding name (cl100k_base for GPT/Qwen, gpt2 for others)",
            default="cl100k_base"
        )
        
        pipeline_config["summarization"] = {
            "run": True,
            "max_tokens": int(max_tokens),
            "token_overlap": int(token_overlap),
            "encoding_name": encoding_name
        }
    else:
        pipeline_config["summarization"] = {"run": False}
    
    # Chunking stage
    console.print("\n[bold cyan]Chunking Stage[/bold cyan]")
    if Confirm.ask("Enable chunking stage?", default=True):
        chunking_mode_options = ["fast_chunking", "semantic_chunking"]
        chunking_mode_idx = typer.prompt(
            "Select chunking mode:\n1. fast_chunking (token-based)\n2. semantic_chunking (embedding-based)\nEnter number",
            type=int,
            default=1
        ) - 1
        chunking_mode = chunking_mode_options[chunking_mode_idx]
        
        l_max_tokens = Prompt.ask(
            "Maximum tokens per chunk",
            default="128"
        )
        token_overlap = Prompt.ask(
            "Token overlap between chunks",
            default="0"
        )
        encoding_name = Prompt.ask(
            "Encoding name",
            default="cl100k_base"
        )
        
        chunking_config = {
            "chunking_mode": chunking_mode,
            "l_max_tokens": int(l_max_tokens),
            "token_overlap": int(token_overlap),
            "encoding_name": encoding_name,
        }
        
        if chunking_mode == "semantic_chunking":
            l_min_tokens = Prompt.ask(
                "Minimum tokens per chunk",
                default="64"
            )
            tau_threshold = Prompt.ask(
                "Similarity threshold (tau)",
                default="0.8"
            )
            chunking_config.update({
                "l_min_tokens": int(l_min_tokens),
                "tau_threshold": float(tau_threshold)
            })
        
        # Multi-hop settings
        h_min = Prompt.ask(
            "Minimum hops (h_min)",
            default="2"
        )
        h_max = Prompt.ask(
            "Maximum hops (h_max)",
            default="3"
        )
        num_multihops_factor = Prompt.ask(
            "Multi-hop factor",
            default="3"
        )
        
        chunking_config.update({
            "h_min": int(h_min),
            "h_max": int(h_max),
            "num_multihops_factor": int(num_multihops_factor)
        })
        
        pipeline_config["chunking"] = {
            "run": True,
            "chunking_configuration": chunking_config
        }
    else:
        pipeline_config["chunking"] = {"run": False}
    
    # Single-shot question generation
    console.print("\n[bold cyan]Single-Shot Question Generation Stage[/bold cyan]")
    if Confirm.ask("Enable single-shot question generation stage?", default=True):
        additional_instructions = Prompt.ask(
            "Additional instructions for question generation",
            default="Generate questions to test a curious adult"
        )
        
        sampling_mode_options = ["all", "count", "percentage"]
        sampling_mode_idx = typer.prompt(
            "Select chunk sampling mode:\n1. all (use all chunks)\n2. count (fixed number of chunks)\n3. percentage (percentage of chunks)\nEnter number",
            type=int,
            default=1
        ) - 1
        sampling_mode = sampling_mode_options[sampling_mode_idx]
        
        random_seed = Prompt.ask(
            "Random seed for sampling",
            default="42"
        )
        
        # Construct chunk_sampling config conditionally
        chunk_sampling_config = {
            "mode": sampling_mode,
            "random_seed": int(random_seed)
        }
        if sampling_mode == "count":
            sampling_value = Prompt.ask(
                "Number of chunks to sample",
                default="5"
            )
            chunk_sampling_config["value"] = int(sampling_value)
        elif sampling_mode == "percentage":
            sampling_value = Prompt.ask(
                "Percentage of chunks to sample (0.0-1.0)",
                default="0.3"
            )
            chunk_sampling_config["value"] = float(sampling_value)
            
        pipeline_config["single_shot_question_generation"] = {
            "run": True,
            "additional_instructions": additional_instructions,
            "chunk_sampling": chunk_sampling_config
        }
    else:
        pipeline_config["single_shot_question_generation"] = {"run": False}
    
    # Multi-hop question generation
    console.print("\n[bold cyan]Multi-Hop Question Generation Stage[/bold cyan]")
    if Confirm.ask("Enable multi-hop question generation stage?", default=True):
        additional_instructions = Prompt.ask(
            "Additional instructions for question generation",
            default="Generate questions to test a curious adult"
        )
        
        sampling_mode_options = ["all", "count", "percentage"]
        sampling_mode_idx = typer.prompt(
            "Select chunk sampling mode:\n1. all (use all chunks)\n2. count (fixed number of chunks)\n3. percentage (percentage of chunks)\nEnter number",
            type=int,
            default=1
        ) - 1
        sampling_mode = sampling_mode_options[sampling_mode_idx]
        
        random_seed = Prompt.ask(
            "Random seed for sampling",
            default="42"
        )
        
        # Construct chunk_sampling config conditionally
        chunk_sampling_config = {
            "mode": sampling_mode,
            "random_seed": int(random_seed)
        }
        if sampling_mode == "count":
            sampling_value = Prompt.ask(
                "Number of chunks to sample",
                default="5"
            )
            chunk_sampling_config["value"] = int(sampling_value)
        elif sampling_mode == "percentage":
            sampling_value = Prompt.ask(
                "Percentage of chunks to sample (0.0-1.0)",
                default="0.3"
            )
            chunk_sampling_config["value"] = float(sampling_value)
            
        pipeline_config["multi_hop_question_generation"] = {
            "run": True,
            "additional_instructions": additional_instructions,
            "chunk_sampling": chunk_sampling_config
        }
    else:
        pipeline_config["multi_hop_question_generation"] = {"run": False}
    
    # LightEval stage
    console.print("\n[bold cyan]LightEval Stage[/bold cyan]")
    pipeline_config["lighteval"] = {
        "run": Confirm.ask("Enable LightEval stage?", default=True)
    }
    
    # Citation score filtering stage
    console.print("\n[bold cyan]Citation Score Filtering Stage[/bold cyan]")
    pipeline_config["citation_score_filtering"] = {
        "run": Confirm.ask("Enable citation score filtering stage?", default=True)
    }
    
    config["pipeline"] = pipeline_config
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save the configuration dictionary to a YAML file.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
        output_path (str): The path where to save the configuration file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the configuration to a YAML file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
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
        
        console.print(Panel.fit(
            f"[bold green]Configuration successfully generated and saved to:[/bold green]\n"
            f"[bold]{output_path}[/bold]\n\n"
            f"You can now run YourBench with this configuration using:\n"
            f"[bold]yourbench run --config {output_path}[/bold]"
        ))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Configuration generation cancelled.[/bold yellow]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]Error generating configuration: {str(e)}[/bold red]")
        logger.exception(f"Configuration generation failed: {e}")
        raise typer.Exit(code=1)