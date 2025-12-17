"""Structured output generation for LLMs with Pydantic schemas."""

import re
import json
from typing import Any, List, Type, TypeVar, Optional
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel, ValidationError


# Helper functions copied to avoid circular import
def _maybe_strip_triple_backticks(text_in: str) -> str:
    """
    Removes triple backticks (``` or ```json) from the beginning
    and end of a string, if present.
    """
    if not text_in or not isinstance(text_in, str):
        return ""
    try:
        pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, text_in)
        if match:
            return match.group(1)
    except Exception as e:
        logger.debug(f"Error stripping backticks: {e}")
    return text_in


def _best_effort_json_extract(full_text: str) -> list[str]:
    """
    Collect bracket-delimited substrings that might be valid JSON.
    Returns a list of candidates (which may be empty).
    """
    if not full_text or not isinstance(full_text, str):
        return []
    candidates = []
    try:
        pattern = r"([\[{].*?[\]}])"
        matches = re.findall(pattern, full_text, flags=re.DOTALL)
        for match_text in matches:
            if (match_text.startswith("[") and match_text.endswith("]")) or (
                match_text.startswith("{") and match_text.endswith("}")
            ):
                candidates.append(match_text.strip())
    except Exception as e:
        logger.debug(f"Error in best-effort JSON extraction: {e}")
    return candidates


def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extract text enclosed in <tag>...</tag> from the given string.
    Returns an empty string if the tag is not found.
    """
    try:
        pattern = rf"<{tag}\s*>([\s\S]*?)</{tag}>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    except Exception as e:
        logger.debug(f"Error extracting tag content for '{tag}': {e}")
    return ""


T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredGenerationConfig:
    """Configuration for structured generation."""

    use_structured_outputs: bool = False
    fallback_to_parsing: bool = True
    validation_strict: bool = True
    max_retries: int = 3
    include_schema_in_prompt: bool = True


class StructuredGenerator:
    """Handle structured output generation across different LLM providers."""

    def __init__(self, config: Optional[StructuredGenerationConfig] = None):
        self.config = config or StructuredGenerationConfig()

    def _detect_provider(self, model_name: str, base_url: Optional[str]) -> str:
        """Detect if model supports native structured outputs."""
        model_lower = model_name.lower()
        base_url_lower = (base_url or "").lower()

        # OpenAI models with structured output support
        if any(x in model_lower for x in ["gpt-4", "gpt-3.5"]):
            if any(x in base_url_lower for x in ["openai", "azure"]):
                return "openai"

        # Anthropic Claude models
        if "claude" in model_lower:
            return "anthropic"

        # Default to generic JSON prompting
        return "generic"

    def generate_prompt_with_schema(self, schema: Type[T], original_prompt: str, examples: bool = True) -> str:
        """Add schema information to the prompt for models without native support."""

        schema_json = schema.model_json_schema()

        # Simplify schema for readability
        simplified_schema = self._simplify_schema(schema_json)

        schema_section = f"""IMPORTANT: You must respond with valid JSON that matches this exact structure:

{json.dumps(simplified_schema, indent=2)}

Key requirements:
- Response must be valid JSON only
- Follow the exact field names and types
- Include all required fields
- Ensure proper JSON formatting (quotes, commas, brackets)
"""

        if examples:
            # Create a minimal valid example
            example = self._create_example(schema)
            schema_section += f"""\nExample of a valid response:\n{json.dumps(example, indent=2)}\n"""

        return original_prompt + "\n\n" + schema_section

    def _simplify_schema(self, schema: dict) -> dict:
        """Simplify JSON schema for better LLM understanding."""

        def simplify_property(prop: dict) -> Any:
            if "$ref" in prop:
                # Handle references
                return "object"

            prop_type = prop.get("type")
            if prop_type == "array":
                items = prop.get("items", {})
                return f"array of {simplify_property(items)}"
            elif prop_type == "object":
                if "properties" in prop:
                    return {k: simplify_property(v) for k, v in prop["properties"].items()}
                return "object"
            elif "enum" in prop:
                return f"one of: {prop['enum']}"
            elif "anyOf" in prop:
                types = [simplify_property(p) for p in prop["anyOf"]]
                return f"one of: {types}"
            else:
                return prop_type or "string"

        if "properties" in schema:
            return {k: simplify_property(v) for k, v in schema["properties"].items()}
        return schema

    def _create_example(self, schema: Type[T]) -> dict:
        """Create a minimal valid example for the schema."""

        # Import the question schemas
        from yourbench.schemas.question_schemas import (
            MultiHopQuestionBatch,
            SingleShotQuestionBatch,
        )

        # Create appropriate example based on schema type
        if schema == SingleShotQuestionBatch:
            return {
                "qa_pairs": [
                    {
                        "question": "What is the main topic discussed in this text?",
                        "answer": "The text discusses the importance of structured data validation in modern software systems, particularly focusing on type safety and error handling.",
                        "question_type": "conceptual",
                        "estimated_difficulty": 3,
                        "thought_process": "This question targets the overall understanding of the text's main theme, requiring the reader to synthesize the key points.",
                        "citations": [
                            {"text": "structured data validation is crucial", "source_location": "paragraph 1"}
                        ],
                    }
                ]
            }
        elif schema == MultiHopQuestionBatch:
            return {
                "qa_pairs": [
                    {
                        "question": "How do the concepts in chunk 1 relate to the implementation details in chunk 3?",
                        "answer": "The theoretical framework presented in chunk 1 provides the foundation for the practical implementation described in chunk 3, demonstrating how abstract concepts translate to concrete code.",
                        "question_type": "synthesis",
                        "estimated_difficulty": 7,
                        "thought_process": "This question requires understanding and connecting information from multiple sections to form a cohesive answer.",
                        "citations": [],
                    }
                ],
                "chunks_used": [1, 3],
            }
        else:
            # Generic example
            return {"example": "data"}

    def parse_structured_response(self, raw_response: str, schema: Type[T], strict: bool = True) -> Optional[T]:
        """Parse and validate a response against a Pydantic schema.

        Args:
            raw_response: The raw text response from the LLM
            schema: The Pydantic model to validate against
            strict: Whether to raise on validation errors

        Returns:
            Validated Pydantic model instance or None if parsing fails
        """

        # Try multiple extraction strategies
        json_candidates = []

        # Strategy 1: Direct JSON parse
        try:
            json_candidates.append(json.loads(raw_response))
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from XML tags
        for tag in ["output_json", "json", "response"]:
            extracted = _extract_tag_content(raw_response, tag)
            if extracted:
                try:
                    json_candidates.append(json.loads(extracted))
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Extract from code blocks
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        for match in re.finditer(code_block_pattern, raw_response):
            try:
                json_candidates.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Best-effort extraction
        for candidate_str in _best_effort_json_extract(raw_response):
            try:
                json_candidates.append(json.loads(candidate_str))
            except json.JSONDecodeError:
                pass

        # Try to validate each candidate
        for candidate in json_candidates:
            try:
                return schema.model_validate(candidate)
            except ValidationError as e:
                logger.debug(f"Validation error for candidate: {e}")
                continue

        # If strict mode and no valid parse found
        if strict:
            logger.error(f"Failed to parse structured response for schema {schema.__name__}")
            logger.debug(f"Raw response: {raw_response[:500]}...")

        return None

    def adapt_legacy_response(
        self, parsed_qa_pairs: List[dict], schema: Type[T], question_mode: str = "open-ended"
    ) -> Optional[T]:
        """Adapt legacy parsed QA pairs to structured schema.

        This provides backward compatibility with existing parsing.
        """

        from yourbench.schemas.question_schemas import (
            OpenEndedQuestion,
            MultiChoiceQuestion,
            SingleShotQuestionBatch,
        )

        if not parsed_qa_pairs:
            return None

        try:
            adapted_questions = []

            for pair in parsed_qa_pairs:
                # Map legacy fields to new schema
                question_data = {
                    "question": str(pair.get("question", "")).strip(),
                    "answer": str(pair.get("answer", "")).strip(),
                    "question_type": self._map_question_type(pair.get("question_type", "")),
                    "estimated_difficulty": max(1, min(10, int(pair.get("estimated_difficulty", 5)))),
                    "thought_process": str(pair.get("thought_process", "Generated from document content")),
                    "citations": self._adapt_citations(pair.get("citations", [])),
                }

                # Add choices for multi-choice questions
                if question_mode == "multi-choice" and "choices" in pair:
                    question_data["choices"] = pair["choices"][:4]  # Ensure max 4
                    question_data["correct_choice"] = "A"  # Default if not specified
                    question_cls = MultiChoiceQuestion
                else:
                    question_cls = OpenEndedQuestion

                # Validate and add
                question = question_cls(**question_data)
                adapted_questions.append(question)

            # Return appropriate batch type
            if schema == SingleShotQuestionBatch:
                return SingleShotQuestionBatch(qa_pairs=adapted_questions)
            else:
                # Generic batch
                return schema(qa_pairs=adapted_questions)

        except Exception as e:
            logger.error(f"Failed to adapt legacy response: {e}")
            return None

    def _map_question_type(self, legacy_type: str) -> str:
        """Map legacy question types to schema enum values."""

        type_map = {
            "factual": "factual",
            "analytical": "analytical",
            "conceptual": "conceptual",
            "inferential": "inferential",
            "evaluative": "evaluative",
            "synthesis": "synthesis",
            "application": "application",
        }

        legacy_lower = str(legacy_type).lower().strip()
        return type_map.get(legacy_lower, "conceptual")  # Default to conceptual

    def _adapt_citations(self, legacy_citations: Any) -> List[dict]:
        """Adapt legacy citations to new format."""

        if not legacy_citations:
            return []

        if not isinstance(legacy_citations, list):
            return []

        adapted = []
        for cit in legacy_citations[:5]:  # Max 5 citations
            if isinstance(cit, str):
                adapted.append({"text": cit, "source_location": None})
            elif isinstance(cit, dict) and "text" in cit:
                adapted.append({"text": str(cit["text"]), "source_location": cit.get("source_location")})

        return adapted
