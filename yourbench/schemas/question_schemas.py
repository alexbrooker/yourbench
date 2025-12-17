"""Pydantic schemas for question generation structured outputs."""

from enum import Enum
from typing import List, Union, Literal, Optional

from pydantic import Field, BaseModel, ConfigDict, field_validator


class QuestionType(str, Enum):
    """Types of questions that can be generated."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CONCEPTUAL = "conceptual"
    INFERENTIAL = "inferential"
    EVALUATIVE = "evaluative"
    SYNTHESIS = "synthesis"
    APPLICATION = "application"


class Citation(BaseModel):
    """A citation from the source text."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, max_length=500, description="Direct quote from the source text")
    source_location: Optional[str] = Field(None, description="Optional location reference in the source")


class BaseQuestion(BaseModel):
    """Base class for all question types."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=10, max_length=500, description="The question text")
    answer: str = Field(..., min_length=20, max_length=2000, description="The complete answer to the question")
    question_type: QuestionType = Field(..., description="Category of the question")
    estimated_difficulty: int = Field(
        ..., ge=1, le=10, description="Difficulty level from 1 (easiest) to 10 (hardest)"
    )
    thought_process: str = Field(
        ..., min_length=30, max_length=1000, description="Reasoning behind the question and answer"
    )
    citations: List[Citation] = Field(
        default_factory=list, max_length=5, description="Supporting citations from the text"
    )

    @field_validator("question")
    @classmethod
    def question_format(cls, v: str) -> str:
        """Ensure question ends with a question mark."""
        v = v.strip()
        if not v.endswith("?"):
            v = v + "?"
        return v

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        """Ensure answer is meaningful."""
        if v.strip().lower() in ["n/a", "na", "none", "unknown"]:
            raise ValueError("Answer must provide meaningful content")
        return v


class OpenEndedQuestion(BaseQuestion):
    """An open-ended question requiring a free-form answer."""

    pass


class MultiChoiceQuestion(BaseQuestion):
    """A multiple-choice question with exactly 4 options."""

    choices: List[str] = Field(..., min_length=4, max_length=4, description="Exactly 4 answer choices")
    correct_choice: Literal["A", "B", "C", "D"] = Field(..., description="The letter of the correct choice")

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v: List[str]) -> List[str]:
        """Ensure choices are unique and non-empty."""
        if len(v) != 4:
            raise ValueError("Must have exactly 4 choices")

        # Strip whitespace from choices
        v = [choice.strip() for choice in v]

        # Check all choices are non-empty
        if any(not choice for choice in v):
            raise ValueError("All choices must be non-empty")

        # Check choices are unique
        if len(set(v)) != 4:
            raise ValueError("All choices must be unique")

        return v

    @field_validator("answer")
    @classmethod
    def answer_matches_choice(cls, v: str, info) -> str:
        """Ensure answer matches one of the choices."""
        if "choices" in info.data and "correct_choice" in info.data:
            choices = info.data["choices"]
            correct_choice = info.data["correct_choice"]
            choice_index = ord(correct_choice) - ord("A")

            if 0 <= choice_index < len(choices):
                expected_answer = choices[choice_index]
                # Allow some flexibility in answer format
                if not (
                    v == expected_answer or v.startswith(f"{correct_choice})") or v.startswith(f"{correct_choice}.")
                ):
                    # Override with the correct format
                    return expected_answer
        return v


class QuestionBatch(BaseModel):
    """A batch of generated questions."""

    model_config = ConfigDict(str_strip_whitespace=True)

    questions: List[Union[OpenEndedQuestion, MultiChoiceQuestion]] = Field(
        ..., min_length=1, description="List of generated questions"
    )

    metadata: Optional[dict] = Field(None, description="Optional metadata about the generation")


class SingleShotQuestionBatch(BaseModel):
    """Response format for single-shot question generation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    qa_pairs: List[Union[OpenEndedQuestion, MultiChoiceQuestion]] = Field(
        ..., min_length=1, max_length=10, description="Generated question-answer pairs from a single chunk"
    )


class MultiHopQuestionBatch(BaseModel):
    """Response format for multi-hop question generation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    qa_pairs: List[Union[OpenEndedQuestion, MultiChoiceQuestion]] = Field(
        ..., min_length=1, max_length=5, description="Generated questions requiring synthesis across multiple chunks"
    )

    chunks_used: List[int] = Field(default_factory=list, description="Indices of chunks used to generate questions")
