"""
YourBench Dataset Engine – Offline‑Friendly Rewrite
--------------------------------------------------
This module replaces the original ``yourbench.utils.dataset_engine`` to add **first‑class
offline support** while remaining 100 % backward‑compatible with existing configs.
Core additions:
• Local‑only mode: honoured automatically when ``HF_HUB_OFFLINE=1`` **or** one of the
    config flags ``offline: true`` / ``hf_configuration.offline: true`` is set.
• Seamless local caching: every save operation writes to ``config["local_dataset_dir"]``
    (subset‑aware). Load operations transparently look there first when offline or when
    remote fetch fails.
• Graceful Hub push failure handling: any exception during ``push_to_hub`` is caught, a
    warning is logged, and the pipeline continues without data loss.
• Opt‑in remote disabled flag: ``push_to_hub`` can be disabled per‑call or permanently
    via ``hf_configuration.push_to_hub: false`` in the user config.

Usage remains identical:
    from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,  # for local path loading
)
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HFValidationError, OfflineModeIsEnabled
from loguru import logger

__all__ = [
    "ConfigurationError",
    "custom_load_dataset",
    "custom_save_dataset",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigurationError(Exception):
    """Raised when the user configuration is incomplete or invalid."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _is_offline(config: Dict[str, Any]) -> bool:
    """Return ``True`` when we must not make network calls to the HF Hub."""

    env_offline = os.getenv("HF_HUB_OFFLINE", "0") in {"1", "true", "True"}
    cfg_offline = bool(config.get("offline", False) or config.get("hf_configuration", {}).get("offline", False))
    return env_offline or cfg_offline


def _get_full_dataset_repo_name(config: Dict[str, Any]) -> str:
    """Return a *syntactically* valid repo id without performing a network call when offline.

    The logic is identical to the previous implementation, except that we **skip** the
    ``HfApi.repo_info`` validation when offline to avoid raising ``OfflineModeIsEnabled``.
    """

    if "hf_configuration" not in config:
        raise ConfigurationError("Missing 'hf_configuration' section in config file.")

    hf_cfg = config["hf_configuration"]
    dataset_name = hf_cfg.get("hf_dataset_name")
    if not dataset_name:
        raise ConfigurationError("'hf_dataset_name' must be specified under 'hf_configuration'.")

    organization = hf_cfg.get("hf_organization")
    token = hf_cfg.get("token")
    offline = _is_offline(config)

    # Expand env vars, if any ------------------------------------------------
    def _expand(value: Optional[str]) -> Optional[str]:
        if value and isinstance(value, str) and value.startswith("$"):
            expanded = os.getenv(value[1:])
            if not expanded:
                logger.warning(f"Environment variable '{value[1:]}' referenced but not set.")
                return value  # return the original – will trigger later validation
            return expanded
        return value

    dataset_name = _expand(dataset_name)
    organization = _expand(organization)

    # Validate required parts ----------------------------------------------
    if dataset_name.startswith("$"):
        raise ConfigurationError(
            f"Environment variable for hf_dataset_name ('{dataset_name}') is not available.")

    if (not organization or organization.startswith("$")) and token and not offline:
        try:
            organization = whoami(token=token).get("name")
        except Exception:  # network or auth issues; best effort only
            pass

    full_repo = dataset_name if ("/" in dataset_name or not organization) else f"{organization}/{dataset_name}"

    # When online, quickly sanity‑check the id format with lightweight call ---
    if not offline:
        try:
            HfApi().repo_info(repo_id=full_repo, repo_type="dataset", token=token)
        except OfflineModeIsEnabled:
            # Should not happen (offline==False) but catch just in case
            offline = True
        except HFValidationError as ve:
            raise ConfigurationError(f"Invalid HF repo id '{full_repo}': {ve}") from ve
        except Exception as e:
            # Non‑fatal issues (e.g. 404). We only care about *syntax* here.
            logger.debug(f"Repo validation skipped due to: {e}")

    return full_repo


# ---------------------------------------------------------------------------
# Public API – load / save helpers
# ---------------------------------------------------------------------------


def custom_load_dataset(config: Dict[str, Any], subset: Optional[str] = None) -> Dataset:
    """Load the requested subset, preferring local cache when offline.

    Load order:
    1. If offline & ``local_dataset_dir`` exists → ``load_from_disk``.
    2. If online → try Hub ``load_dataset``.
    3. Fallback: attempt local cache (even online) then return empty dataset.
    """

    local_dir = config.get("local_dataset_dir")
    offline = _is_offline(config)

    def _try_local(path: str) -> Optional[Dataset]:
        if path and os.path.exists(path):
            try:
                return load_from_disk(path)
            except Exception as e:
                logger.error(f"Failed loading local dataset at '{path}': {e}")
        return None

    # 1. Offline or explicit local first
    if offline:
        logger.info("Offline mode detected – loading dataset exclusively from disk.")
        path = os.path.join(local_dir, subset) if (local_dir and subset) else local_dir
        ds = _try_local(path)
        if ds is not None:
            return ds
        logger.warning(f"Local dataset not found at '{path}'. Returning empty dataset.")
        return Dataset.from_dict({})

    # 2. Online – try HF Hub --------------------------------------------------
    repo_id = _get_full_dataset_repo_name(config)
    try:
        return load_dataset(repo_id, name=subset, split="train")
    except Exception as e:
        logger.error(f"Remote load failed: {e}. Attempting local cache (if any).")
        # 3. Local fallback
        path = os.path.join(local_dir, subset) if (local_dir and subset) else local_dir
        ds = _try_local(path)
        if ds is not None:
            return ds
        logger.warning("Neither remote nor local data available. Returning empty dataset.")
        return Dataset.from_dict({})


def custom_save_dataset(
    dataset: Dataset,
    config: Dict[str, Any],
    subset: Optional[str] = None,
    *,
    save_local: bool = True,
    push_to_hub: bool | None = None,
) -> None:
    """Save *dataset* locally and/or push it to the Hub.

    Parameters
    ----------
    dataset
        A HF ``Dataset`` instance.
    config
        Pipeline configuration dict.
    subset
        Optional subset name (used both for HF config_name and local dir nesting).
    save_local
        Force saving via ``save_to_disk`` (default: *True*).
    push_to_hub
        Override global decision. ``None`` → default determined from:
            • offline mode (never push when offline)
            • config['hf_configuration'].get('push_to_hub', True)
    """

    offline = _is_offline(config)
    cfg_push_default = bool(config.get("hf_configuration", {}).get("push_to_hub", True))
    do_push = cfg_push_default if push_to_hub is None else push_to_hub
    do_push = do_push and not offline  # never push while offline

    # 1. Always attempt to save locally when requested -----------------------
    local_dir = config.get("local_dataset_dir")
    if save_local and local_dir:
        local_target = os.path.join(local_dir, subset) if subset else local_dir
        try:
            os.makedirs(local_target, exist_ok=True)
            if subset:
                DatasetDict({subset: dataset}).save_to_disk(local_target)
            else:
                dataset.save_to_disk(local_target)
            logger.success(f"Dataset saved locally to '{local_target}'")
        except Exception as e:
            logger.error(f"Failed to save dataset locally at '{local_target}': {e}")

    # 2. Concatenate with existing (remote OR local) if requested ------------
    if config.get("hf_configuration", {}).get("concat_if_exist", False):
        try:
            existing = custom_load_dataset(config, subset=subset)
            dataset = concatenate_datasets([existing, dataset])
            logger.info(f"Concatenated current dataset with existing data ({len(existing)} → {len(dataset)} rows).")
        except Exception as e:
            logger.warning(f"concat_if_exist requested but failed to load existing data: {e}")

    # 3. Push to Hub (optional) ----------------------------------------------
    if do_push:
        repo_id = _get_full_dataset_repo_name(config)
        cfg_name = subset or "default"
        try:
            logger.info(f"Pushing dataset to HuggingFace Hub ({repo_id}, config_name='{cfg_name}').")
            dataset.push_to_hub(
                repo_id=repo_id,
                private=config["hf_configuration"].get("private", True),
                config_name=cfg_name,
            )
            logger.success(f"Dataset successfully pushed to {repo_id}")
        except Exception as e:
            logger.error(f"Push to Hub failed ({e}). Dataset remains safe on disk.")
    else:
        reason = "offline mode" if offline else "push_to_hub disabled by config"
        logger.info(f"Skipping push to Hub – {reason}.")
