# Custom Schema Demo

This example demonstrates how to use **custom Pydantic schemas** to control the output format of generated questions.

## What This Example Shows

1. **Custom field definitions** - Add fields like `difficulty`, `prerequisites`, `key_concepts`
2. **Type constraints** - Use `Literal` types to restrict values (e.g., difficulty levels)
3. **Structured metadata** - Get consistent, parseable output from the LLM

## Files

```
custom_schema_demo/
├── config.yaml                    # Pipeline configuration
├── schemas/
│   ├── technical_qa.py           # Technical Q&A schema with prerequisites
│   └── educational_assessment.py  # Bloom's taxonomy schema
├── data/
│   └── yourbench_arxiv_paper.pdf  # Sample document
└── README.md
```

## Schema Examples

### Technical Q&A Schema (`schemas/technical_qa.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal

class DataFormat(BaseModel):
    question: str = Field(description="A technical question")
    answer: str = Field(description="Complete technical answer")
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(...)
    prerequisites: list[str] = Field(description="Required prior knowledge")
    key_concepts: list[str] = Field(description="Main concepts covered")
    citations: list[str] = Field(description="Source quotes")
```

### Educational Assessment Schema (`schemas/educational_assessment.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal

class DataFormat(BaseModel):
    question: str = Field(description="Comprehension question")
    answer: str = Field(description="Expected answer")
    bloom_level: Literal["remember", "understand", "apply", "analyze", "evaluate", "create"] = Field(...)
    learning_objective: str = Field(description="What students should learn")
    common_mistakes: list[str] = Field(description="Typical student errors")
    citations: list[str] = Field(description="Source material")
```

## Running This Example

```bash
# Set up your environment
export OPENAI_BASE_URL="your-api-url"
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="your-model"
export HF_TOKEN="your-hf-token"
export HF_ORGANIZATION="your-org"

# Run the pipeline
yourbench run example/custom_schema_demo/config.yaml --debug
```

## Expected Output

With the technical Q&A schema, each generated question will include:

```json
{
  "question": "How does YourBench ensure question grounding?",
  "answer": "YourBench uses citation verification to ensure...",
  "difficulty": "intermediate",
  "prerequisites": ["LLM basics", "Evaluation concepts"],
  "key_concepts": ["grounding", "citation", "benchmark generation"],
  "citations": ["The document states that..."]
}
```

## Switching Schemas

To use a different schema, update your `config.yaml`:

```yaml
pipeline:
  single_hop_question_generation:
    # Switch to educational assessment format
    question_schema: example/custom_schema_demo/schemas/educational_assessment.py
```

## Creating Your Own Schema

1. Create a new `.py` file
2. Define a class named `DataFormat` that inherits from `pydantic.BaseModel`
3. Add fields with `Field(description="...")` to guide the LLM
4. Reference the file path in your config

See [docs/CUSTOM_SCHEMAS.md](../../docs/CUSTOM_SCHEMAS.md) for full documentation.
