from pydantic import BaseModel, Field
from typing import Literal


class DataFormat(BaseModel):
    """Educational assessment format with Bloom's taxonomy."""
    
    question: str = Field(
        description="A question testing comprehension of the material"
    )
    answer: str = Field(
        description="The expected correct answer"
    )
    bloom_level: Literal[
        "remember", "understand", "apply", "analyze", "evaluate", "create"
    ] = Field(
        description="Bloom's taxonomy cognitive level this question tests"
    )
    learning_objective: str = Field(
        description="What the student should learn from this question"
    )
    common_mistakes: list[str] = Field(
        description="Typical errors students make when answering"
    )
    citations: list[str] = Field(
        description="Source quotes from the document"
    )
