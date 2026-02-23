# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YourBench is a dynamic benchmark generation framework (by Hugging Face) that creates domain-specific evaluation datasets from source documents (PDFs, HTML, text). It generates question-answer pairs to evaluate LLMs against fresh content they haven't memorized. Python 3.12+ required.

## Common Commands

```bash
# Run the pipeline
yourbench run example/default_example/config.yaml --debug

# Or via module
python -m yourbench run --config example/default_example/config.yaml

# Validate config without running
yourbench validate config.yaml

# Estimate token usage
yourbench estimate config.yaml

# Lint and format
ruff check .
ruff format .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/unit/test_schema_validation.py

# Run a single test
pytest tests/unit/test_schema_validation.py::test_name -v
```

## Architecture

### Pipeline Execution Flow

The pipeline is orchestrated by `yourbench/pipeline/handler.py`, which dynamically imports and runs stages using `importlib`. Stage execution order is defined in `yourbench/conf/loader.py::STAGE_ORDER`:

1. **ingestion** — Converts source documents to markdown (`pipeline/ingestion.py`)
2. **summarization** — Generates document summaries (`pipeline/summarization.py`)
3. **chunking** — Splits documents into chunks (`pipeline/chunking.py`)
4. **single_hop_question_generation** — Questions from single chunks (`pipeline/question_generation/single_hop.py`)
5. **multi_hop_question_generation** — Questions requiring multiple chunks (`pipeline/question_generation/multi_hop.py`)
6. **cross_document_question_generation** — Questions spanning documents (`pipeline/question_generation/cross_document.py`)
7. **question_rewriting** — Rewrites questions for clarity (`pipeline/question_rewriting.py`)
8. **prepare_lighteval** — Formats for LightEval evaluation (`pipeline/prepare_lighteval.py`)
9. **citation_score_filtering** — Filters by citation quality (`pipeline/citation_score_filtering.py`)

Each stage module must export a `run(config)` function. Question generation stages live in `pipeline/question_generation/` and share core logic in `_core.py`.

### Configuration System

- **Schema**: Pydantic models in `yourbench/conf/schema.py` define all config options with validation and defaults. Root config is `YourbenchConfig`.
- **Loader**: `yourbench/conf/loader.py` handles YAML parsing, `$VAR` env expansion, legacy field migration, and stage auto-enablement (presence of a stage key in `pipeline:` implies `run: True`).
- **Model roles**: `model_roles` in config maps stage names to model names. If not specified, the first model in `model_list` is used for all stages.

### Inference Engine

`yourbench/utils/inference/inference_core.py` is the central inference system:
- `run_inference(config, step_name, inference_calls)` is the main entry point
- Uses `AsyncInferenceClient` from `huggingface_hub` for all LLM calls
- Manages per-model concurrency via `asyncio.Semaphore` (controlled by `max_concurrent_requests`)
- Implements exponential backoff retry (up to 12 attempts by default)
- Returns `Dict[str, List[str]]` mapping model names to response lists

### Dataset Engine

`yourbench/utils/dataset_engine.py` manages all data persistence:
- `custom_load_dataset(config, subset)` — loads from local disk or HF Hub
- `custom_save_dataset(dataset, config, subset)` — saves locally and/or pushes to Hub
- Supports `concat_if_exist` for incremental runs and JSONL export

### Prompts

Default prompt templates live in `yourbench/prompts/` as markdown files, organized by stage. Loaded automatically by `yourbench/conf/prompts.py`. Users can override prompts via config or by pointing to custom file paths.

### Custom Question Schemas

Users can define custom Pydantic output schemas (must export a `DataFormat` class) and point to them via the `question_schema` config field. Schema loading is handled by `yourbench/utils/schema_loader.py`.

## Code Conventions

- **Formatter/Linter**: Ruff with `line-length = 119`. Ignores: E501, C901, F841. Import sorting uses `length-sort` with `lines-after-imports = 2`.
- **Logging**: Use `loguru` throughout. Use `logger.bind(stage=name)` for stage context.
- **Async**: Pipeline stages that do LLM inference use `asyncio.run()` at the boundary, with async internals.
- **Testing**: Tests are in `tests/unit/` and `tests/integration/`. Use pytest.