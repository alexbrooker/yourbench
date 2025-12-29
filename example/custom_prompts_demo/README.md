# Custom Prompts Demo

This example demonstrates how to customize question generation using your own system prompts.

## Key Feature

**Custom System Prompts** - Override the default question generation behavior with domain-specific prompts.

## What's Included

- `custom_prompts/single_hop_system_prompt.md` - Kid-friendly question generator
- `custom_prompts/multi_hop_system_prompt.md` - Multi-hop reasoning for children
- `data/` - Sample children's book PDFs

## How to Run

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY=sk-xxxxx

# Run the pipeline
yourbench run example/custom_prompts_demo/config.yaml
```

## Customization

To create your own custom prompts:

1. Copy a prompt template from `example/prompts/`
2. Modify the role, objectives, and quality criteria
3. Reference your prompt in the config:

```yaml
pipeline:
  single_hop_question_generation:
    single_hop_system_prompt: path/to/your/prompt.md
```

## See Also

- `example/prompts/` - Reusable prompt templates for different domains
