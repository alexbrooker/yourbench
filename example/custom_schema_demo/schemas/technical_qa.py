from typing import Literal

from pydantic import Field, BaseModel


class DataFormat(BaseModel):
    """Technical documentation Q&A format with detailed metadata."""

    question: str = Field(description="A technical question about the document content")
    answer: str = Field(description="Complete, accurate answer with technical details")
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Technical difficulty level required to answer"
    )
    prerequisites: list[str] = Field(description="List of concepts the reader should understand first")
    key_concepts: list[str] = Field(description="Main technical concepts covered in this Q&A")
    citations: list[str] = Field(description="Direct quotes from the source document that support the answer")
