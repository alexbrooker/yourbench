# Shared Prompt Templates

This directory contains reusable prompt templates for question generation.

## Available Templates

| Template | Use Case | Description |
|----------|----------|-------------|
| `single_hop_default.md` | General | Balanced prompt for most document types |
| `single_hop_technical.md` | Technical docs | API docs, tutorials, specifications |
| `single_hop_business.md` | Business reports | Strategy reports, market analysis |
| `multi_hop_default.md` | General | Multi-hop reasoning across chunks |

## Usage

Reference these in your config:

```yaml
pipeline:
  single_hop_question_generation:
    single_hop_system_prompt: example/prompts/single_hop_technical.md
  multi_hop_question_generation:
    multi_hop_system_prompt: example/prompts/multi_hop_default.md
```

## Customization

Copy a template and modify the:
- **Role description** - Who the question generator is
- **Core objectives** - What makes a good question for your domain
- **Quality standards** - Domain-specific quality criteria
