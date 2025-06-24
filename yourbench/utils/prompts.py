"""
This module contains the prompts for the pipeline stages.
"""

SUMMARIZATION_USER_PROMPT = """You are an AI assistant tasked with analyzing and summarizing technical documents, specifically C++ tutorials, lectures, documentation, blogposts, and code examples. Your goal is to generate a concise yet comprehensive summary of the given content. Follow these steps carefully:

1. You will be provided with a document. This document may be very long and/or split into multiple contiguous sections. It may contain unnecessary artifacts such as links, HTML tags, or other web-related elements, as well as C++ code snippets.

2. Here is the document to be summarized:
<document>
{document}
</document>

3. Before generating the summary, use a mental scratchpad to take notes as you read through the document. Enclose your notes within <scratchpad> tags. For example:

<scratchpad>
- Main topic: [Note the main subject of the document]
- Key concepts: [List important C++ concepts, techniques, and explanations]
- Code insights: [Note any critical code examples, syntax patterns, or programming constructs]
- Structure: [Note how the document is organized or chunked]
- Potential artifacts to ignore: [List any web-related elements that should be disregarded]
</scratchpad>

4. As you analyze the document:
   - Focus solely on the technical content, ignoring any unnecessary web-related elements.
   - Treat all sections or chunks as part of a single, continuous document.
   - Identify the main topic, key concepts, and important code-related insights.
   - Ensure technical accuracy, especially when referencing C++ code or terminology.

5. After your analysis, generate a final summary that:
   - Captures the essence of the document in a concise manner.
   - Includes the main topic, key C++ concepts, and code insights.
   - Presents information in a logical and coherent order.
   - Is comprehensive yet concise, typically ranging from 3–5 sentences (unless the document is particularly long or complex).

6. Enclose your final summary within <final_summary> tags. For example:

<final_summary>
[Your concise and comprehensive summary of the document goes here.]
</final_summary>

Remember, your task is to provide a clear, accurate, and concise summary of the technical content, including relevant C++ code insights, while disregarding web artifacts or unrelated elements."""

# Shared blocks

BASE_PROMPT_RULES = """
General Requirements:
- All questions must focus on C++ concepts, behavior, syntax, or idioms. Discard questions that are not directly related to C++ programming.
- Assume the reader has no access to the original document — all context must be included in the question.
- Do not use vague phrases like "the code above", "this function", or "the snippet below".
- Each question is a stand-alone one.

Symbol References:
- Any class, function, method, variable, or flag mentioned in a question must:
  (a) appear explicitly in the <text_chunk>, and
  (b) have its full definition or relevant behavior included in the question as a fenced C++ code block.
- If the symbol or behavior is missing from <text_chunk>, or cannot be shown inline, the question must be discarded.

Code Usage:
- All C++ code included in questions or answers must be wrapped in fenced code blocks with language specified as `cpp`. Example:
  ```cpp
  // Your C++ example here
"""

DOCUMENT_ANALYSIS_REQUIREMENT = """
Begin your output with a <document_analysis> block.

This block should briefly explain:
- The central C++ concepts and themes covered across the text
- Important syntax, behaviors, or idioms (e.g., lifetime rules, pointer semantics, template patterns)
- Common misconceptions or subtle implementation details that may confuse learners
- Any code examples, patterns, or inline comments that provide useful context for constructing meaningful questions

Focus on insights that justify the question choices, highlight reasoning challenges, or reveal conceptual depth.
"""

# Base models

OPEN_ENDED_MODEL = """
```python
class QuestionRow(BaseModel):
   thought_process: str  # Explain why this question was selected, what concept it targets, and how the <text_chunk> supports it.
                        # Confirm that any referenced class, function, method, variable, or flag appears explicitly in the <text_chunk>.
                        # Confirm that any referenced symbol has its relevant code included in the question as a fenced C++ block.
                        # If the question is about behavior, state changes, or return values, include the exact code that shows this behavior.
                        # Discard any question that depends on context not present in the <text_chunk>, such as project goals or external knowledge.
   question_type: Literal["analytical", "application-based", "clarification",
                          "counterfactual", "conceptual", "true-false",
                          "factual", "open-ended", "false-premise", "edge-case", "troubleshooting"]
   question: str  # A fully self-contained question. If referencing any code, include the full code snippet directly in the question.
   answer: str  # A complete, self-contained answer that depends solely on the <text_chunk>.
   estimated_difficulty: int  # Integer from 1 (easy) to 10 (very difficult), targeting an advanced C++ learner. Ultra-hard questions should target difficulty 9–10 and require multiple layers of reasoning, non-obvious behavior, or deep standard knowledge.
   citations: List[str]  # Exact quotes from <text_chunk> that justify the answer. Must be factual and specific.
```
"""

MULTIPLE_CHOICE_MODEL = """
```python
class QuestionRow(BaseModel):
   thought_process: str  # Explain why this question was selected, what concept it targets, and how the <text_chunk> supports it.
                        # Confirm that any referenced class, function, method, variable, or flag appears explicitly in the <text_chunk>.
                        # Confirm that any referenced symbol has its relevant code included in the question as a fenced C++ block.
                        # If the question is about behavior, state changes, or return values, include the exact code that shows this behavior.
                        # Discard any question that depends on context not present in the <text_chunk>, such as project goals or external knowledge.
   question_type: Literal["analytical", "application-based", "clarification",
                          "counterfactual", "conceptual", "true-false",
                          "factual", "false-premise", "edge-case", "troubleshooting"]
   question: str  # Fully self-contained question. If it mentions a code snippet or behavior, the relevant code must be included inline.
   answer: str  # One of "A", "B", "C", or "D"
   choices: List[str]  # Exactly 4 options (A–D), clearly distinct. Only one is correct. Distractors must reflect realistic misunderstandings, not arbitrary edits or vague errors.
   estimated_difficulty: int  # Integer from 1 (easy) to 10 (very difficult), targeting an advanced C++ learner. Ultra-hard questions should target difficulty 9–10 and require multiple layers of reasoning, non-obvious behavior, or deep standard knowledge.
   citations: List[str]  # Verbatim quotes or phrases from the <text_chunk> that directly support the correct answer
```
"""

EXAMPLE_OPEN_ENDED_OUTPUT = """
## Output Format

{DOCUMENT_ANALYSIS_REQUIREMENT}

Wrap your output in <output_json> tags. Output a list of QuestionRow objects using this model:

{OPEN_ENDED_MODEL}

## Example:
<document_analysis>
Key concept: Virtual functions and polymorphism in C++
Facts: Virtual functions allow derived classes to override behavior
Reasoning cues: Importance of virtual destructors and dynamic dispatch in memory-safe design
</document_analysis>

<output_json>
[
  {
    "thought_process": "This question checks the learner’s understanding of how virtual functions support polymorphism in C++. It’s open-ended to encourage explanation of runtime behavior and design rationale.",
    "question_type": "open-ended",
    "question": "Why are virtual functions important in C++ polymorphism, and when should you use a virtual destructor?",
    "answer": "Virtual functions allow derived classes to override base class behavior at runtime using dynamic dispatch. A virtual destructor ensures that destructors of derived classes are correctly called when deleting through a base pointer, preventing resource leaks.",
    "estimated_difficulty": 7,
    "citations": [
      "Virtual functions allow overriding behavior in derived classes.",
      "Virtual destructors ensure proper cleanup during polymorphic deletion."
    ]
  }
]
</output_json>
"""

EXAMPLE_MC_OUTPUT = """
## Output Format

{DOCUMENT_ANALYSIS_REQUIREMENT}

Wrap your output in <output_json> tags. Output a list of QuestionRow objects using this model:

{MULTIPLE_CHOICE_MODEL}

## Example:
<document_analysis>
Key concept: Virtual functions and inheritance in C++
Facts: Virtual functions enable runtime polymorphism; destructors must be virtual to avoid undefined behavior
Reasoning cues: Common misuse of non-virtual destructors in base classes
</document_analysis>

<output_json>
[
  {
    "thought_process": "This question assesses conceptual understanding of C++ polymorphism and destructor behavior. The distractors include common misconceptions like thinking virtual functions are needed for all class methods.",
    "question_type": "conceptual",
    "question": "Why should a base class in C++ have a virtual destructor?",
    "answer": "C",
    "choices": [
      "(A) To make all methods in the class virtual.",
      "(B) To allow constructors to be overridden in derived classes.",
      "(C) To ensure derived class destructors are correctly called during deletion via a base pointer.",
      "(D) To allow function overloading at runtime."
    ],
    "estimated_difficulty": 7,
    "citations": ["A virtual destructor ensures proper cleanup when deleting derived objects through base class pointers."]
  }
]
</output_json>
"""

# Single-hop prompt templates

QUESTION_GENERATION_SYSTEM_PROMPT_HEADER = """## Your Role
You are an expert C++ content creator tasked with designing exceptionally challenging, technically precise questions based on C++ code, lectures, and documentation.

You specialize in exposing edge cases, deep language mechanics, and subtle design trade-offs. Your questions should require expert-level reasoning — including knowledge of undefined behavior, template instantiation quirks, object lifetimes, ABI constraints, and non-obvious runtime effects.

Your goal is to generate **ultra-difficult, self-contained** questions that challenge even seasoned C++ developers and compiler engineers.

## Critical Rule
Only reference functions, methods, classes, or variables if they are explicitly named in the <text_chunks>. Do not infer any symbols.

## Input Structure
<title>
[Document title]
</title>
<document_summary>
[Contextual overview]
</document_summary>
<text_chunk>
[Main content block with C++ examples and explanations]
</text_chunk>
<additional_instructions>
[Constraints or guidance for question creation]
</additional_instructions>

## Objective
Generate a set of meaningful question–answer pairs based only on <text_chunk>. The reader will not see the original document, so each question must be fully self-contained and understandable on its own.
- If referencing a method, function, class, or variable, its entire definition must appear verbatim in <text_chunk> or be fully included inline in the question.
- Do not reference symbols that are only implied or summarized – discard such questions.
- All code examples must be wrapped in proper fenced C++ code blocks using:
   ```cpp
   <code>
   ```
This ensures clarity, consistency, and standalone readability.
"""

QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT = (
    DOCUMENT_ANALYSIS_REQUIREMENT
    + "\n\nWrap your output in <output_json> tags. Output a list of QuestionRow objects using this model:\n"
    + OPEN_ENDED_MODEL
    + "\n\nSpecial requirements:\n"
    + BASE_PROMPT_RULES
    + "\n\n"
    + EXAMPLE_OPEN_ENDED_OUTPUT
)

QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI = (
    DOCUMENT_ANALYSIS_REQUIREMENT
    + "\n\nWrap your output in <output_json> tags. Output a list of QuestionRow objects using this model:\n"
    + MULTIPLE_CHOICE_MODEL
    + "\n\nSpecial requirements:\n"
    + BASE_PROMPT_RULES
    + "\n\n"
    + EXAMPLE_MC_OUTPUT
)

QUESTION_GENERATION_SYSTEM_PROMPT = (
    QUESTION_GENERATION_SYSTEM_PROMPT_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT
)
QUESTION_GENERATION_SYSTEM_PROMPT_MULTI = (
    QUESTION_GENERATION_SYSTEM_PROMPT_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI
)

QUESTION_GENERATION_USER_PROMPT = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunk>
{text_chunk}
</text_chunk>

<additional_instructions>
{additional_instructions}
</additional_instructions>"""

# Multi-hop prompt templates

MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER = """## Your Role
You are an expert C++ educator and evaluator focused on generating extremely challenging, high-complexity multi-hop questions.

Each question must synthesize information across multiple parts of the input and test deep understanding of C++ internals, such as type deduction, copy/move semantics, pointer aliasing, subtle ordering rules, and undefined or implementation-defined behavior.

Your questions are designed to expose reasoning gaps in even advanced C++ programmers. All questions must be self-contained, precise, and deeply technical.

## Critical Rule
Only reference functions, methods, classes, or variables if they are explicitly named in the <text_chunks>. Do not infer any symbols.

## Input Structure
<title>
[Document title]
</title>
<document_summary>
[Contextual summary]
</document_summary>
<text_chunks>
<text_chunk_0>
[First content block]
</text_chunk_0>
<text_chunk_1>
[Second block]
</text_chunk_1>
[Additional blocks allowed]
</text_chunks>
<additional_instructions>
[Instructions on complexity or focus]
</additional_instructions>

## Objective
Generate a set of meaningful multi-hop questions based only on <text_chunk>. The reader will not see the original document, so each question must be fully self-contained and understandable on its own.
- If referencing a method, function, class, or variable, its entire definition must appear verbatim in <text_chunk> or be fully included inline in the question.
- Do not reference symbols that are only implied or summarized – discard such questions.
- All code examples must be wrapped in proper fenced C++ code blocks using:
   ```cpp
   <code>
   ```
This ensures clarity, consistency, and standalone readability.
"""

MULTI_HOP_QUESTION_GENERATION_USER_PROMPT = """<title>
{title}
</title>

<document_summary>
{document_summary}
</document_summary>

<text_chunks>
{chunks}
</text_chunks>

<additional_instructions>
{additional_instructions}
</additional_instructions>"""

MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT = (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT
    + BASE_PROMPT_RULES
)
MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT_MULTI = (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI
    + BASE_PROMPT_RULES
)

COMBINE_SUMMARIES_USER_PROMPT = """\
You will receive a list of chunk-level summaries from the *same* \
document.  Combine them into a single, well-structured paragraph that reads \
naturally and eliminates redundancy.

<chunk_summaries>
{chunk_summaries}
</chunk_summaries>

Return ONLY the final text inside <final_summary> tags."""
