"""Adapter to integrate Hydra with existing YourBench configuration system."""

import os
from pathlib import Path
from typing import Any, Dict

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger


def load_hydra_config(config_path: str = None, overrides: list = None) -> DictConfig:
    """Load configuration using Hydra composition.
    
    Args:
        config_path: Path to config.yaml or None to use default
        overrides: List of Hydra overrides (e.g., ['model=openai', 'pipeline=minimal'])
    
    Returns:
        Composed Hydra configuration
    """
    if config_path is None:
        config_dir = Path(__file__).parent.resolve()
        config_name = "config"
    else:
        config_path = Path(config_path).resolve()
        config_dir = config_path.parent.resolve()
        config_name = config_path.stem
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with absolute config directory
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose configuration with overrides
        cfg = compose(config_name=config_name, overrides=overrides or [])
    
    # Resolve environment variables and interpolations
    OmegaConf.resolve(cfg)
    
    return cfg


def is_hydra_config(config_path: str) -> bool:
    """Check if a config file is a Hydra config (has defaults section)."""
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line.startswith('defaults:')
        return False
    except Exception:
        return False


def hydra_to_legacy_format(cfg: DictConfig) -> Dict[str, Any]:
    """Convert Hydra config to legacy YourBench format.
    
    This allows existing code to work with Hydra configs without modification.
    """
    # Convert to plain dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # The Hydra config already matches the expected structure
    # Just ensure all required fields are present
    if 'model_list' not in config_dict and 'models' in config_dict:
        config_dict['model_list'] = config_dict.pop('models')
    
    if 'pipeline_config' not in config_dict and 'pipeline' in config_dict:
        # Legacy format expects pipeline_config instead of pipeline
        pass  # Keep as 'pipeline' since loader handles both
    
    return config_dict


def create_hydra_config_from_legacy(legacy_config: Dict[str, Any]) -> str:
    """Convert a legacy config to Hydra format and save it.
    
    This is a migration helper to convert existing configs to Hydra format.
    """
    # Create a Hydra config structure
    hydra_config = {
        'defaults': [
            'model: custom',
            'pipeline: default',
            'hf: default',
            '_self_'
        ]
    }
    
    # Add any overrides from the legacy config
    if 'model_list' in legacy_config:
        hydra_config['model_list'] = legacy_config['model_list']
    
    if 'hf_configuration' in legacy_config:
        hydra_config['hf_configuration'] = legacy_config['hf_configuration']
    
    if 'pipeline' in legacy_config or 'pipeline_config' in legacy_config:
        hydra_config['pipeline'] = legacy_config.get('pipeline', legacy_config.get('pipeline_config', {}))
    
    return OmegaConf.to_yaml(hydra_config)
