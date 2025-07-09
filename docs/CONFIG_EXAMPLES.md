# YourBench Configuration Examples

This document provides practical examples of YourBench configurations for different use cases.

## Basic Configuration

The simplest configuration for getting started:

```yaml
hf_configuration:
  hf_dataset_name: my_first_dataset
  private: true

model_list:
  - model_name: Qwen/Qwen2.5-72B-Instruct
    provider: fireworks-ai
    api_key: $HF_TOKEN

pipeline:
  ingestion:
    run: true
    source_documents_dir: data/raw
    output_dir: data/processed
  summarization:
    run: true
  chunking:
    run: true
  single_shot_question_generation:
    run: true
  multi_hop_question_generation:
    run: true
  prepare_lighteval:
    run: true
```

## Multi-Model Configuration

Using different models for different pipeline stages:

```yaml
hf_configuration:
  hf_dataset_name: multi_model_dataset
  hf_organization: $HF_ORGANIZATION
  private: false

model_list:
  - model_name: Qwen/Qwen2.5-VL-72B-Instruct  # Vision model for ingestion
    provider: fireworks-ai
    api_key: $HF_TOKEN
    max_concurrent_requests: 16
  
  - model_name: gpt-4o  # OpenAI for question generation
    base_url: https://api.openai.com/v1
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 8
  
  - model_name: claude-3-opus-20240229  # Anthropic for summarization
    base_url: https://api.anthropic.com/v1
    api_key: $ANTHROPIC_API_KEY
    max_concurrent_requests: 4

model_roles:
  ingestion:
    - Qwen/Qwen2.5-VL-72B-Instruct
  summarization:
    - claude-3-opus-20240229
  single_shot_question_generation:
    - gpt-4o
  multi_hop_question_generation:
    - gpt-4o
  cross_document_question_generation:
    - claude-3-opus-20240229

pipeline:
  ingestion:
    run: true
    llm_ingestion: true  # Use LLM for complex PDFs
    pdf_dpi: 300
  summarization:
    run: true
  chunking:
    run: true
  single_shot_question_generation:
    run: true
  multi_hop_question_generation:
    run: true
  cross_document_question_generation:
    run: true
    max_combinations: 50
    chunks_per_document: 2
    num_docs_per_combination: [2, 3]
  prepare_lighteval:
    run: true
```

## Local Development Configuration

For local development with self-hosted models:

```yaml
hf_configuration:
  hf_dataset_name: local_dev_dataset
  local_dataset_dir: data/local_dataset
  local_saving: true
  private: true

model_list:
  - model_name: llama-3-8b-instruct  # Local vLLM server
    base_url: http://localhost:8000/v1
    api_key: $VLLM_API_KEY
    max_concurrent_requests: 4
  
  - model_name: mistral-7b-instruct  # Another local model
    base_url: http://localhost:8001/v1
    api_key: $VLLM_API_KEY
    max_concurrent_requests: 2

pipeline:
  ingestion:
    run: true
    source_documents_dir: example/data/raw
    output_dir: example/data/processed
    llm_ingestion: false  # Use standard parser for speed
  summarization:
    run: true
  chunking:
    run: true
    chunking_configuration:
      l_max_tokens: 256
      h_min: 2
      h_max: 3
      num_multihops_factor: 1
  single_shot_question_generation:
    run: true
  multi_hop_question_generation:
    run: true
  prepare_lighteval:
    run: true
```

## Production Configuration

A robust production setup with error handling and monitoring:

```yaml
hf_configuration:
  hf_dataset_name: production_dataset_v2
  hf_organization: $HF_ORGANIZATION
  private: false
  concat_if_exist: true  # Append to existing dataset
  local_dataset_dir: /data/yourbench_backups
  local_saving: true  # Always backup locally

model_list:
  - model_name: Qwen/Qwen2.5-VL-72B-Instruct
    provider: fireworks-ai
    api_key: $HF_TOKEN
    max_concurrent_requests: 24
    encoding_name: cl100k_base
  
  - model_name: gpt-4o
    base_url: https://api.openai.com/v1
    api_key: $OPENAI_API_KEY
    max_concurrent_requests: 12
  
  - model_name: claude-3-opus-20240229
    base_url: https://api.anthropic.com/v1
    api_key: $ANTHROPIC_API_KEY
    max_concurrent_requests: 6

model_roles:
  ingestion:
    - Qwen/Qwen2.5-VL-72B-Instruct
  summarization:
    - claude-3-opus-20240229
    - gpt-4o  # Fallback model
  single_shot_question_generation:
    - gpt-4o
  multi_hop_question_generation:
    - claude-3-opus-20240229
  cross_document_question_generation:
    - claude-3-opus-20240229
  question_rewriting:
    - gpt-4o

pipeline:
  ingestion:
    run: true
    source_documents_dir: /data/raw_documents
    output_dir: /data/processed_documents
    llm_ingestion: true
    pdf_dpi: 300
  summarization:
    run: true
    max_tokens: 32768
    token_overlap: 512
  chunking:
    run: true
    chunking_configuration:
      l_max_tokens: 512
      token_overlap: 64
      h_min: 2
      h_max: 5
      num_multihops_factor: 3
  single_shot_question_generation:
    run: true
  multi_hop_question_generation:
    run: true
  cross_document_question_generation:
    run: true
    max_combinations: 200
    chunks_per_document: 3
    num_docs_per_combination: [2, 3, 5]
    random_seed: 42
  question_rewriting:
    run: true
    additional_instructions: "Rewrite questions to be more conversational and engaging"
  prepare_lighteval:
    run: true
  citation_score_filtering:
    run: true
```

## Environment Variables

For all configurations above, ensure these environment variables are set:

```bash
# .env file
HF_TOKEN=hf_your_token_here
HF_ORGANIZATION=your_organization
OPENAI_API_KEY=sk-your_openai_key
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key
VLLM_API_KEY=your_vllm_server_key
```

## Creating Configurations

Use the interactive configuration builder for the best experience:

```bash
# Simple configuration with minimal questions
yourbench create --simple my_config.yaml

# Full configuration with all options
yourbench create my_config.yaml

# Use the configuration
yourbench run my_config.yaml
```

The configuration builder will:
- Guide you through all options
- Validate your choices
- Create the `.env` file automatically
- Provide helpful defaults
- Show you the next steps

## Configuration Validation

The configuration system automatically validates:
- Required fields are present
- Environment variables are accessible
- File paths exist and are readable
- Model configurations are valid
- Pipeline stage dependencies are met

If validation fails, you'll get clear error messages explaining what needs to be fixed.