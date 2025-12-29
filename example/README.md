# YourBench Examples

Pre-configured examples demonstrating different YourBench features.

## Quick Start

```bash
# Simplest example - works with just a HuggingFace token
yourbench run example/default_example/config.yaml
```

## Examples Overview

| Example | Key Feature | Model Provider | Data Included |
|---------|-------------|----------------|---------------|
| [`default_example`](default_example/) | **Quickstart** - Minimal config | HuggingFace (free) | ✅ PDF |
| [`harry_potter_quizz`](harry_potter_quizz/) | **Tutorial** - Comprehensive walkthrough | OpenRouter | ✅ PDF |
| [`custom_prompts_demo`](custom_prompts_demo/) | **Custom Prompts** - Domain-specific questions | OpenRouter | ✅ PDFs |
| [`local_vllm_private_data`](local_vllm_private_data/) | **Self-Hosted** - Local vLLM models | Local vLLM | ✅ HTMLs |
| [`rich_pdf_extraction_with_gemini`](rich_pdf_extraction_with_gemini/) | **LLM Ingestion** - Charts/figures extraction | OpenRouter/Gemini | ✅ PDF |
| [`custom_schema_demo`](custom_schema_demo/) | **Custom Schemas** - Pydantic output control | Any OpenAI-compatible | ✅ PDF |

## Shared Resources

| Resource | Description |
|----------|-------------|
| [`prompts/`](prompts/) | Reusable prompt templates for different domains |

## Which Example Should I Use?

- **Just getting started?** → `default_example`
- **Want a detailed tutorial?** → `harry_potter_quizz`
- **Need custom question styles?** → `custom_prompts_demo`
- **Running your own models?** → `local_vllm_private_data`
- **Need structured output fields?** → `custom_schema_demo`
- **Have complex PDFs with charts?** → `rich_pdf_extraction_with_gemini`

## Environment Variables

Most examples need API keys. Create a `.env` file:

```bash
# For HuggingFace models (default_example)
HF_TOKEN=hf_xxxxx

# For OpenRouter examples
OPENROUTER_API_KEY=sk-xxxxx

# For OpenAI
OPENAI_API_KEY=sk-xxxxx
```
