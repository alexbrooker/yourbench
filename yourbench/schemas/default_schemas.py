"""Default Pydantic schemas for YourBench question generation.

These schemas match the existing QuestionRow structure and can be used
as defaults when no custom schema is provided.
"""

from typing import List, Optional

from pydantic import Field, BaseModel, field_validator


class BaseQuestion(BaseModel):
    """Base question schema with common fields."""

    question: str = Field(
        ..., description="The question text. Should be clear, specific, and answerable based on the provided context."
    )
    self_answer: str = Field(
        ...,
        description="A comprehensive answer to the question based on the document context.",
        alias="answer",  # Allow both 'answer' and 'self_answer'
    )
    estimated_difficulty: int = Field(
        default=5, ge=1, le=10, description="Difficulty level from 1 (easiest) to 10 (hardest)."
    )
    self_assessed_question_type: str = Field(
        default="factual",
        description="Type of question: factual, analytical, comparison, causal, procedural, or conceptual.",
        alias="question_type",  # Allow both forms
    )
    thought_process: str = Field(
        default="", description="Reasoning behind the question generation and why it tests understanding."
    )
    citations: List[str] = Field(
        default_factory=list, description="Direct quotes from the source text that support the answer."
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Ensure question ends with a question mark."""
        v = v.strip()
        if v and not v.endswith("?"):
            v = v + "?"
        return v

    @field_validator("self_assessed_question_type")
    @classmethod
    def validate_question_type(cls, v: str) -> str:
        """Normalize question type."""
        valid_types = {
            "factual",
            "analytical",
            "comparison",
            "causal",
            "procedural",
            "conceptual",
            "evaluative",
            "hypothetical",
        }
        v_lower = v.lower().strip()
        if v_lower not in valid_types:
            # Default to factual if unknown type
            return "factual"
        return v_lower

    class Config:
        populate_by_name = True  # Allow population by alias


class SingleHopQuestion(BaseQuestion):
    """Schema for single-hop questions (from one document chunk)."""

    chunk_context: Optional[str] = Field(
        default=None, description="Optional: The specific chunk context this question is based on."
    )


class MultiHopQuestion(BaseQuestion):
    """Schema for multi-hop questions (requiring multiple document chunks)."""

    reasoning_steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning required to answer this question across multiple chunks.",
    )
    source_chunks: List[str] = Field(
        default_factory=list, description="List of chunk IDs or contexts used to formulate this question."
    )
    integration_type: str = Field(
        default="synthesis", description="How information is integrated: synthesis, comparison, sequential, or causal."
    )


class MultiChoiceQuestion(BaseQuestion):
    """Schema for multiple-choice questions."""

    choices: List[str] = Field(..., min_length=4, max_length=4, description="Exactly 4 answer choices (A, B, C, D).")
    correct_answer: str = Field(..., description="The correct answer letter (A, B, C, or D).")
    distractor_reasoning: Optional[str] = Field(
        default=None, description="Explanation of why the distractors are plausible but incorrect."
    )

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v: List[str]) -> List[str]:
        """Ensure exactly 4 choices."""
        if len(v) != 4:
            raise ValueError(f"Must have exactly 4 choices, got {len(v)}")
        return [choice.strip() for choice in v]

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, v: str) -> str:
        """Ensure correct answer is A, B, C, or D."""
        v = v.upper().strip()
        if v not in ["A", "B", "C", "D"]:
            raise ValueError(f"Correct answer must be A, B, C, or D, got {v}")
        return v


# Batch wrappers for generating multiple questions at once
class SingleHopQuestionBatch(BaseModel):
    """Batch of single-hop questions."""

    questions: List[SingleHopQuestion] = Field(
        ..., description="List of single-hop questions generated from the document chunk."
    )


class MultiHopQuestionBatch(BaseModel):
    """Batch of multi-hop questions."""

    questions: List[MultiHopQuestion] = Field(
        ..., description="List of multi-hop questions requiring multiple document chunks."
    )


class MultiChoiceQuestionBatch(BaseModel):
    """Batch of multiple-choice questions."""

    questions: List[MultiChoiceQuestion] = Field(
        ..., description="List of multiple-choice questions with 4 options each."
    )
