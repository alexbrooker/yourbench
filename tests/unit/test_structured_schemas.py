"""Unit tests for structured question schemas."""

import pytest
from pydantic import ValidationError

from yourbench.schemas.question_schemas import (
    Citation,
    QuestionType,
    OpenEndedQuestion,
    MultiChoiceQuestion,
    MultiHopQuestionBatch,
    SingleShotQuestionBatch,
)


class TestCitation:
    """Test Citation schema."""

    def test_valid_citation(self):
        citation = Citation(text="This is a direct quote from the source", source_location="paragraph 2")
        assert citation.text == "This is a direct quote from the source"
        assert citation.source_location == "paragraph 2"

    def test_citation_strips_whitespace(self):
        citation = Citation(text="  Text with spaces  ", source_location=None)
        assert citation.text == "Text with spaces"

    def test_citation_validation(self):
        # Empty text should fail
        with pytest.raises(ValidationError):
            Citation(text="")

        # Too long text should fail
        with pytest.raises(ValidationError):
            Citation(text="x" * 501)


class TestOpenEndedQuestion:
    """Test OpenEndedQuestion schema."""

    def test_valid_open_ended_question(self):
        question = OpenEndedQuestion(
            question="What is the main topic of this text",
            answer="The text discusses machine learning concepts including neural networks and deep learning algorithms.",
            question_type=QuestionType.CONCEPTUAL,
            estimated_difficulty=5,
            thought_process="This question tests understanding of the overall theme of the text.",
            citations=[],
        )
        # Should automatically add question mark
        assert question.question == "What is the main topic of this text?"
        assert question.estimated_difficulty == 5

    def test_question_validation(self):
        # Too short question
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                question="What?",
                answer="A valid answer that is long enough to pass validation.",
                question_type=QuestionType.FACTUAL,
                estimated_difficulty=3,
                thought_process="Some thought process that explains the reasoning.",
            )

    def test_answer_validation(self):
        # Too short answer
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                question="What is the meaning of life?",
                answer="42",
                question_type=QuestionType.CONCEPTUAL,
                estimated_difficulty=10,
                thought_process="Deep philosophical reasoning about existence.",
            )

        # Invalid answer content
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                question="What is the main topic?",
                answer="N/A",
                question_type=QuestionType.CONCEPTUAL,
                estimated_difficulty=5,
                thought_process="The question asks about the main topic.",
            )

    def test_difficulty_validation(self):
        # Difficulty out of range
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                question="What is the main concept discussed?",
                answer="The main concept is artificial intelligence and its applications in modern technology.",
                question_type=QuestionType.CONCEPTUAL,
                estimated_difficulty=11,  # Out of range
                thought_process="Testing difficulty validation.",
            )


class TestMultiChoiceQuestion:
    """Test MultiChoiceQuestion schema."""

    def test_valid_multi_choice_question(self):
        question = MultiChoiceQuestion(
            question="Which of the following best describes the main topic?",
            answer="Machine Learning - the study of algorithms that improve through experience",
            question_type=QuestionType.CONCEPTUAL,
            estimated_difficulty=3,
            thought_process="This tests comprehension of the main theme.",
            citations=[],
            choices=[
                "Machine Learning - the study of algorithms that improve through experience",
                "Web Development",
                "Database Design",
                "Network Security",
            ],
            correct_choice="A",
        )
        assert len(question.choices) == 4
        assert question.correct_choice == "A"
        assert question.answer == "Machine Learning - the study of algorithms that improve through experience"

    def test_choices_validation(self):
        # Not enough choices
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                question="Which option is correct?",
                answer="Option A",
                question_type=QuestionType.FACTUAL,
                estimated_difficulty=2,
                thought_process="Testing choice validation.",
                choices=["Option A", "Option B"],  # Only 2 choices
                correct_choice="A",
            )

        # Duplicate choices
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                question="Which option is correct?",
                answer="Option A",
                question_type=QuestionType.FACTUAL,
                estimated_difficulty=2,
                thought_process="Testing choice validation.",
                choices=["Option A", "Option A", "Option C", "Option D"],
                correct_choice="A",
            )

        # Empty choice
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                question="Which option is correct?",
                answer="Option A",
                question_type=QuestionType.FACTUAL,
                estimated_difficulty=2,
                thought_process="Testing choice validation.",
                choices=["Option A", "", "Option C", "Option D"],
                correct_choice="A",
            )

    def test_correct_choice_validation(self):
        # Invalid correct choice letter
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                question="Which option is correct?",
                answer="Option A",
                question_type=QuestionType.FACTUAL,
                estimated_difficulty=2,
                thought_process="Testing choice validation.",
                choices=["Option A", "Option B", "Option C", "Option D"],
                correct_choice="E",  # Invalid letter
            )


class TestQuestionBatches:
    """Test question batch schemas."""

    def test_single_shot_batch(self):
        batch = SingleShotQuestionBatch(
            qa_pairs=[
                OpenEndedQuestion(
                    question="What is the main topic?",
                    answer="The main topic is artificial intelligence and its applications in healthcare.",
                    question_type=QuestionType.CONCEPTUAL,
                    estimated_difficulty=3,
                    thought_process="This question tests understanding of the overall theme.",
                    citations=[],
                ),
                OpenEndedQuestion(
                    question="What specific technique is mentioned?",
                    answer="The text mentions deep learning as a specific technique used for image analysis in medical diagnostics.",
                    question_type=QuestionType.FACTUAL,
                    estimated_difficulty=2,
                    thought_process="This question tests recall of specific information.",
                    citations=[Citation(text="deep learning for image analysis", source_location="section 2")],
                ),
            ]
        )
        assert len(batch.qa_pairs) == 2
        assert batch.qa_pairs[0].question_type == QuestionType.CONCEPTUAL

    def test_multi_hop_batch(self):
        batch = MultiHopQuestionBatch(
            qa_pairs=[
                OpenEndedQuestion(
                    question="How does the concept in section 1 relate to the implementation in section 3?",
                    answer="The theoretical framework introduced in section 1 provides the mathematical foundation for the practical implementation described in section 3.",
                    question_type=QuestionType.SYNTHESIS,
                    estimated_difficulty=7,
                    thought_process="This question requires connecting information from multiple sections.",
                    citations=[],
                )
            ],
            chunks_used=[1, 3],
        )
        assert len(batch.qa_pairs) == 1
        assert batch.chunks_used == [1, 3]

    def test_empty_batch_validation(self):
        # Empty qa_pairs should fail
        with pytest.raises(ValidationError):
            SingleShotQuestionBatch(qa_pairs=[])

    def test_mixed_question_types_in_batch(self):
        """Test that batches can contain both open-ended and multi-choice questions."""
        batch = SingleShotQuestionBatch(
            qa_pairs=[
                OpenEndedQuestion(
                    question="What is the main concept?",
                    answer="The main concept is natural language processing and its applications.",
                    question_type=QuestionType.CONCEPTUAL,
                    estimated_difficulty=4,
                    thought_process="Testing conceptual understanding.",
                    citations=[],
                ),
                MultiChoiceQuestion(
                    question="Which technique is used for tokenization?",
                    answer="Byte-Pair Encoding is the technique used for tokenization",
                    question_type=QuestionType.FACTUAL,
                    estimated_difficulty=3,
                    thought_process="Testing knowledge of specific techniques.",
                    citations=[],
                    choices=[
                        "Byte-Pair Encoding is the technique used for tokenization",
                        "Word2Vec",
                        "TF-IDF",
                        "One-Hot Encoding",
                    ],
                    correct_choice="A",
                ),
            ]
        )
        assert len(batch.qa_pairs) == 2
        assert isinstance(batch.qa_pairs[0], OpenEndedQuestion)
        assert isinstance(batch.qa_pairs[1], MultiChoiceQuestion)
