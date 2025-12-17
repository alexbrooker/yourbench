"""Default Pydantic schemas for YourBench question generation."""

from yourbench.schemas.default_schemas import (
    MultiHopQuestion,
    SingleHopQuestion,
    MultiChoiceQuestion,
    MultiHopQuestionBatch,
    SingleHopQuestionBatch,
    MultiChoiceQuestionBatch,
)


__all__ = [
    "SingleHopQuestion",
    "SingleHopQuestionBatch",
    "MultiHopQuestion",
    "MultiHopQuestionBatch",
    "MultiChoiceQuestion",
    "MultiChoiceQuestionBatch",
]
