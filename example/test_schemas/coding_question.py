"""Example custom schema for coding/programming questions."""

from pydantic import BaseModel, Field
from typing import List, Optional


class CodingQuestion(BaseModel):
    """Schema for coding/programming evaluation questions."""

    problem_statement: str = Field(..., description="Clear problem statement or task description")

    # Code-specific fields
    expected_approach: str = Field(..., description="Expected approach or algorithm to solve the problem")
    sample_solution: str = Field(..., description="Sample solution or pseudocode")

    # Constraints and requirements
    time_complexity: str = Field(default="O(n)", description="Expected time complexity (e.g., O(n), O(log n), O(n^2))")
    space_complexity: str = Field(default="O(1)", description="Expected space complexity")

    # Test cases
    test_cases: List[dict] = Field(
        default_factory=list, description="List of test cases with input and expected output"
    )
    edge_cases: List[str] = Field(default_factory=list, description="Important edge cases to consider")

    # Metadata
    difficulty: str = Field(default="medium", description="Difficulty: easy, medium, hard")
    topics: List[str] = Field(
        default_factory=list, description="Topics covered (e.g., arrays, strings, dynamic programming)"
    )
    programming_languages: List[str] = Field(
        default_factory=lambda: ["python", "java", "javascript"],
        description="Suitable programming languages for this problem",
    )

    # Learning objectives
    learning_objectives: List[str] = Field(default_factory=list, description="What concepts this problem teaches")
    common_mistakes: Optional[str] = Field(default=None, description="Common mistakes to watch out for")
