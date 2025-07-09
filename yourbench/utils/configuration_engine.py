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


@dataclass
class YourbenchConfig:
    """The main configuration class for the YourBench pipeline."""

    hf_configuration: HuggingFaceConfig
    model_list: list[ModelConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "YourbenchConfig":
        """
        Load YAML → dict → dataclass, with env-var expansion
        confined to HuggingFaceConfig.__post_init__.
        """
        with open(Path(path), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        hf_kwargs = data.get("hf_configuration", {})
        model_list = data.get("model_list", [])
        return cls(hf_configuration=HuggingFaceConfig(**hf_kwargs), model_list=[ModelConfig(**m) for m in model_list])


if __name__ == "__main__":
    config = YourbenchConfig.from_yaml("example/configs/simple_example.yaml")
    print(config)
