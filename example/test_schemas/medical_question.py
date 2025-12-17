"""Example custom schema for medical/scientific questions."""

from pydantic import BaseModel, Field
from typing import List, Optional


class MedicalQuestion(BaseModel):
    """Schema for medical/scientific questions with evidence levels."""

    question: str = Field(..., description="The clinical or scientific question")
    answer: str = Field(..., description="Evidence-based answer")

    # Medical-specific fields
    evidence_level: str = Field(
        default="moderate", description="Level of evidence: high, moderate, low, or expert_opinion"
    )
    clinical_relevance: str = Field(
        default="theoretical", description="Clinical relevance: direct, indirect, theoretical"
    )

    # Supporting information
    key_findings: List[str] = Field(default_factory=list, description="Key findings or facts supporting the answer")
    contraindications: Optional[str] = Field(
        default=None, description="Important contraindications or warnings if applicable"
    )
    references: List[str] = Field(default_factory=list, description="Scientific references or citations")

    # Metadata
    specialty: Optional[str] = Field(default=None, description="Medical specialty area (e.g., cardiology, oncology)")
    difficulty: int = Field(default=5, ge=1, le=10, description="Difficulty for medical students (1-10)")
