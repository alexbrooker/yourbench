"""
This module contains the prompts for the pipeline stages.
"""

SUMMARIZATION_USER_PROMPT = """You are an AI assistant tasked with analyzing and summarizing technical documents, specifically C++ tutorials, lectures, and documentation. Your goal is to generate a concise yet comprehensive summary of the given content. Follow these steps carefully:

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


QUESTION_GENERATION_SYSTEM_PROMPT_HEADER = """## Your Role

You are an expert educational content creator specializing in crafting thoughtful, rich, and engaging questions based on technical documents, particularly C++ tutorials, lectures, and documentation. Your goal is to produce meaningful, moderately challenging question-answer pairs that encourage reflection, insight, and deep understanding of programming concepts, tailored specifically according to provided instructions.

## Input Structure

Your input consists of:

<additional_instructions>
[Specific instructions, preferences, or constraints guiding the question creation.]
</additional_instructions>

<title>
[Document title]
</title>

<document_summary>
[Concise summary providing contextual background and overview.]
</document_summary>

<text_chunk>
[The single text segment to analyze, which may include code examples and explanations.]
</text_chunk>

## Primary Objective

Your goal is to generate a thoughtful set of question-answer pairs from the provided `<text_chunk>`. Aim for moderate complexity that encourages learners to deeply engage with the C++ content, reflect on programming concepts, and clearly demonstrate their understanding.

Focus only on the `<text_chunk>` for generating questions; do not rely on `<document_summary>` or `<title>` as sources of factual content.

### Context Fields:

- `<title>`: Contextualizes the C++ topic or concept.
- `<document_summary>`: Brief overview providing technical context.
- `<text_chunk>`: The sole source text, which may include C++ code snippets, technical explanations, and examples.
- `<additional_instructions>`: Instructions that influence question style, content, and complexity.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` XML tags, following these steps:

1. **Technical Content Examination**
   - Identify the central programming concepts, syntax, idioms, and logic patterns in the text chunk.

2. **Concept and Code Exploration**
   - Consider how C++ features, constructs, or examples are presented. Evaluate their usage, correctness, potential pitfalls, and best practices.

3. **Strategic Complexity Calibration**
   - Thoughtfully rate difficulty (1–10), ensuring moderate technical challenge aligned with the additional instructions.

4. **Intentional Question Planning**
   - Design questions that highlight understanding of syntax, semantics, use cases, and problem-solving involving C++.
   - If a question cannot be made self-contained due to missing context, rephrase it to either incorporate the needed context or drop it entirely.

## Additional Instructions for Handling Irrelevant or Bogus Information

### Identification and Ignoring of Irrelevant Information:

- **Irrelevant Elements:** Explicitly disregard hyperlinks, advertisements, headers, footers, navigation menus, disclaimers, social media buttons, or any non-technical web elements.
- **Bogus Information:** Detect and exclude any content that is syntactically invalid, nonsensical, or disconnected from meaningful programming instruction.

### Decision Criteria for Question Generation:

- **Meaningful Technical Content Requirement:** Only generate questions if the `<text_chunk>` includes valid and coherent C++ instructional or reference material.
- **Complete Irrelevance:** If the entire `<text_chunk>` lacks technical value (e.g., only HTML, footers, or marketing), state this in your analysis and DO NOT generate questions.

### Documentation in Analysis:

- Justify inclusion or exclusion decisions clearly in `<document_analysis>` tags.
- Briefly explain any decision not to generate questions due to irrelevance or lack of technical value.

## Question Generation Guidelines

### Encouraged Question Characteristics:

- **Technical Depth**: Prioritize questions that probe C++ understanding (e.g., memory management, control structures, object-oriented principles, templates).
- **Moderate Complexity**: Balance difficulty to challenge without overwhelming, aligned with learner skill level and topic.
- **Self-contained Clarity**: Each question-answer pair must be fully self-contained, meaning a learner should be able to understand and answer the question without accessing the original document, `<document_summary>`, or any external content. Avoid vague references like "as shown above" or "in the example."
- **Concrete Examples**: If a question refers to a specific code example, include the code directly in the question text so it is fully self-contained.
- **Pedagogical Value**: Questions should reinforce comprehension of C++ principles, usage, and common patterns.
- **Conversational Tone**: Use approachable, instructive language while preserving technical clarity.
- **Avoid Redundancy**: Do not generate multiple questions that ask about the same concept in only slightly different ways.

### Permitted Question Types:

- Analytical
- Application-based
- Clarification
- Counterfactual
- Conceptual
- True-False
- Factual
- Open-ended
- False-premise
- Edge-case

(You do not need to use every question type, only those naturally fitting the content and instructions.)"""

QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT = """## Output Structure

This prompt is used exclusively for generating **open-ended** questions from technical C++ content.

Present your final output as a list of JSON objects strictly adhering to this Pydantic model, wrapped within `<output_json>` XML tags:

```python
class QuestionRow(BaseModel):
    thought_process: str  # Clear, detailed rationale for selecting this question and the reasoning process
    question_type: Literal["analytical", "application-based", "clarification",
                           "counterfactual", "conceptual", "true-false",
                           "factual", "open-ended", "false-premise", "edge-case"]
    question: str  # A fully self-contained, well-phrased question. If it refers to a code example, include the code inline.
    answer: str  # A full, self-contained answer based only on the <text_chunk>. Do not require external references.
    estimated_difficulty: int  # Difficulty level from 1 (easy) to 10 (very difficult), based on the complexity of reasoning required
    citations: List[str]  # Verbatim quotes from <text_chunk> that support the answer. These must be factual and precise.
```

## Output Format

Begin by thoughtfully analyzing the provided C++ <text_chunk> within <document_analysis> XML tags.
Focus on programming logic, syntax usage, C++ features, and conceptual clarity. Maintain technical precision, especially when referencing code or rules.

Then output a list of valid QuestionRow JSON objects inside <output_json> XML tags. The list must be valid JSON.


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
    ],
  },
  ...
]
</output_json>
"""

QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI = """## Output Structure

Present your final output as JSON objects strictly adhering to this schema, enclosed within `<output_json>` XML tags. This structure supports both open-ended and multiple-choice questions derived from C++ technical material.

```python
class QuestionRow(BaseModel):
   thought_process: str  # Clear reasoning for why this question was generated, including distractor logic and insight into learning value
   question_type: Literal["analytical", "application-based", "clarification",
                           "counterfactual", "conceptual", "true-false",
                           "factual", "false-premise", "edge-case"]
   question: str  # The question text. It must be fully self-contained. If it refers to an example or snippet, include it inline in the question.
   answer: str  # One of "A", "B", "C", or "D"
   choices: List[str]  # Must contain exactly 4 mutually exclusive, technically valid options, labeled (A)–(D). Only one must be correct. Distractors must reflect common misconceptions—not alternate fixes or unrelated edits.
   estimated_difficulty: int  # Integer from 1 (easy) to 10 (very difficult), calibrated for intermediate learners
   citations: List[str]  # Verbatim quotes or phrases from the <text_chunk> that directly support the correct answer
```

## Output Format

1. Begin with a technical analysis of the <text_chunk> inside <document_analysis> tags.
   - Highlight key concepts, tricky syntax, C++ rules, or common misconceptions.
   - Note any code patterns or features that justify your question construction.
2. Then, generate a list of QuestionRow objects in valid JSON format, wrapped inside <output_json> tags.
   - Ensure proper formatting (no trailing commas).
   - Each question must be self-contained — no "refer to the text" or "as seen above".

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
  },
  ...
]
</output_json>"""

QUESTION_GENERATION_SYSTEM_PROMPT_FOOTER = """## Important Notes
- Strive to generate questions that inspire genuine curiosity, reflection, and thoughtful engagement, especially around technical C++ concepts and code behavior.
- Every question and answer must be fully self-contained. Do not assume the user can refer to the original <text_chunk>.
- If a question references a code example, include that code inline in the question text to preserve clarity and independence.
- Do not use vague references like "the provided code," "the code above," or "the example below." If context is needed, include it explicitly.
- Citations must be short, verbatim quotes from <text_chunk> that directly support the answer. Include code lines only when they clarify key behaviors.
- Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.
- Each "thought_process" must clearly explain the reasoning behind the question and, for multiple-choice questions, the logic behind distractor choices and why the correct answer stands out.
- Ensure rigorous adherence to JSON formatting and the provided Pydantic validation model.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material."""

QUESTION_GENERATION_SYSTEM_PROMPT = (
    QUESTION_GENERATION_SYSTEM_PROMPT_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT
    + QUESTION_GENERATION_SYSTEM_PROMPT_FOOTER
)
QUESTION_GENERATION_SYSTEM_PROMPT_MULTI = (
    QUESTION_GENERATION_SYSTEM_PROMPT_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI
    + QUESTION_GENERATION_SYSTEM_PROMPT_FOOTER
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


MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER = """## Your Role

You are an expert educational content creator specialized in generating insightful and thoughtfully designed multi-hop questions for C++ programming material. Your task is to craft sophisticated, moderately challenging questions that require integrative reasoning across multiple chunks of technical content. Aim to provoke thoughtful reflection, nuanced understanding, and synthesis of C++ concepts, syntax, and coding practices.

## Input Structure

Your input will consist of these components:

<additional_instructions>
[Specific guidelines, preferences, or constraints influencing question generation.]
</additional_instructions>

<title>
[Document title]
</title>

<document_summary>
[A concise summary providing technical context and thematic overview.]
</document_summary>

<text_chunks>
<text_chunk_0>
[First text segment – may contain C++ code, syntax explanations, or conceptual material]
</text_chunk_0>
<text_chunk_1>
[Second text segment]
</text_chunk_1>
[Additional text segments as necessary]
</text_chunks>

## Primary Objective

Generate a thoughtful, technically meaningful set of multi-hop question-answer pairs. Questions should require learners to integrate C++ knowledge across multiple chunks, encouraging critical thinking and deeper understanding of both code and concepts.

### Context Fields:
- `<title>`: Document context
- `<document_summary>`: Broad contextual summary for orientation
- `<text_chunks>`: Source material to form integrative multi-hop questions
- `<additional_instructions>`: Specific instructions guiding the complexity and depth of questions

## Analysis Phase

Perform careful technical analysis within `<document_analysis>` XML tags:

1. **In-depth Text Analysis**
   - Thoughtfully read each text chunk.
   - Identify key themes, nuanced details, and subtle connections.
   - Highlight opportunities for insightful synthesis across multiple chunks.

2. **Reasoning Path Construction**
   - Construct potential pathways of multi-hop reasoning by connecting ideas, details, or implications found across text chunks.
   - Explicitly document how and why multiple chunks are connected to support each question.

3. **Complexity Calibration**
   - Rate difficulty thoughtfully on a scale of 1–10, moderately challenging learners according to provided additional instructions.

4. **Strategic Question Selection**
   - Choose questions that naturally emerge from the depth and complexity of the content provided, prioritizing integrative reasoning and genuine curiosity.

## Question Generation Guidelines

### Question Characteristics

- **Multi-Hop Integration**: Questions must require navigating and synthesizing information from **two or more** distinct text chunks.
- **Self-Contained Format**: Each question-answer pair must be fully understandable on its own. Do not use references like "as shown above" or "in chunk 1." Summarize needed context directly in the question.
- **Clarity & Precision**: Ensure technical correctness. If a question references C++ code or structure, include the example **inline** in the question.
- **Thoughtfulness & Complexity**: Reflect realistic challenges learners face when combining multiple C++ concepts.
- **Educational Relevance**: Reinforce understanding of design patterns, memory management, class hierarchy, templates, etc.
- **Authentic Language**: Use natural, engaging phrasing similar to what developers or students might ask themselves.

### Suggested Question Types
(Use naturally, as fitting to the content complexity)
- Analytical
- Application-based
- Clarification
- Counterfactual
- Conceptual
- True-False
- Factual
- Open-ended
- False-premise
- Edge-case

## Irrelevant Content Filtering

When reviewing text chunks:
- **Ignore** headers, footers, ads, links, disclaimers, and any non-technical elements.
- **Do not generate questions** from any chunk containing only irrelevant or malformed content.
- If a chunk contains mixed content, extract only the technical portions for use.
- If a chunk lacks educational value (e.g., only fluff or broken examples), document this in `<document_analysis>` and skip it.

## Prioritization Rules
- Always favor clarity, accuracy, and educational usefulness.
- Include supporting citations or short quotes from relevant chunks within your analysis when justifying your answer or reasoning.
"""


MULTI_HOP_QUESTION_GENERATION_SYSTEM_FOOTER = """## Important Notes
- Each question-answer pair must be fully self-contained—do not rely on chunk IDs or prior context.
- If code is needed to understand a question, include it directly in the question text.
- Avoid vague references like "the provided code" or "the example above." State context explicitly.
- Favor questions that encourage critical thinking, synthesis across chunks, and real-world C++ reasoning.
- Let the content's complexity guide question difficulty—don’t artificially simplify or overcomplicate.
- Use brief, verbatim citations from relevant chunks to support each answer.
- Justify distractors: they should reflect plausible misunderstandings, not alternate solutions or edits.
- Clearly explain your reasoning in the "thought_process", including how multiple chunks contributed.
- Stick strictly to the required JSON and Pydantic model format.
- Avoid phrases like "as per the text" or "according to the document"—questions must read naturally and independently.
"""

MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT = (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT
    + MULTI_HOP_QUESTION_GENERATION_SYSTEM_FOOTER
)
MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT_MULTI = (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_HEADER
    + QUESTION_GENERATION_SYSTEM_PROMPT_OUTPUT_MULTI
    + MULTI_HOP_QUESTION_GENERATION_SYSTEM_FOOTER
)

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


ZEROSHOT_QA_USER_PROMPT = """Answer the following question:

<question>
{question}
</question>

Enclose your full answer in <answer> XML tags. For example:

<answer>
[your answer here]
</answer>"""

GOLD_QA_USER_PROMPT = """Answer the following question:

<question>
{question}
</question>

Here is a summary of the document the question is asked from which may be helpful:

<document_summary>
{summary}
</document_summary>

And here is a relevant chunk of the document which may prove useful

<document>
{document}
</document>

Enclose your full answer in <answer> XML tags. For example:

<answer>
[your answer here]
</answer>"""

JUDGE_ANSWER_SYSTEM_PROMPT = """You will be provided with the summary of a document, a piece of text, a question generated from that text, and the correct or "gold" answer to the question. Additionally, you will receive two answers: Answer A and Answer B. Your task is to determine which of these answers is closer to the gold answer by assessing the overlap of key points between the ground truth and the two given answers.

# Steps

1. **Document Understanding**:
   - Analyze the provided document summary to grasp the context and main themes.

2. **Chunk Understanding**:
   - Examine the provided text (chunk) to understand its content.

3. **Question Understanding**:
   - Interpret the given question to fully comprehend what is being asked.

4. **Ground Truth Answer Understanding**:
   - Understand the provided ground truth answer, identifying its key points.

5. **Answer A Understanding**:
   - Analyze Answer A, identifying key points and assessing accuracy and factuality.

6. **Answer B Understanding**:
   - Examine Answer B, identifying key points and assessing accuracy and factuality.

7. **Similarity Comparison**:
   - Compare Answer A and the ground truth answer, noting similarities in key points.
   - Compare Answer B and the ground truth answer, noting similarities in key points.

8. **Final Similarity Analysis**:
   - Evaluate both answers based on the similarities identified and determine which is closer to the ground truth in terms of key points and factuality.

# Output Format

- Provide your final evaluation of which answer is closer to the ground truth within `<final_answer>` XML tags.
- Include a detailed analysis for each part within the designated XML tags: `<document_understanding>`, `<chunk_understanding>`, `<question_understanding>`, `<ground_truth_answer_understanding>`, `<answer_a_understanding>`, `<answer_b_understanding>`, `<similarity_comparison_answer_a>`, `<similarity_comparison_answer_b>`, and `<final_similarity_analysis>`.

# Examples

**Input**:
```xml
<document_summary>
[Summary]
</document_summary>

<piece_of_text>
[Text]
</piece_of_text>

<question>
[Question]
</question>

<gold_answer>
[Gold Answer]
</gold_answer>

<answer_a>
[Answer A]
</answer_a>

<answer_b>
[Answer B]
</answer_b>
```
**Output**:
```xml

<document_understanding>
Understanding of the summary including key themes
</document_understanding>

<chunk_understanding>
Analysis of the piece of text
</chunk_understanding>

<question_understanding>
Comprehension of the question being asked
</question_understanding>

<ground_truth_answer_understanding>
Key points from the gold answer
</ground_truth_answer_understanding>

<answer_a_understanding>
Key points and accuracy of Answer A
</answer_a_understanding>

<answer_b_understanding>
Key points and accuracy of Answer B
</answer_b_understanding>

<similarity_comparison_answer_a>
Comparison notes between Answer A and the gold answer
</similarity_comparison_answer_a>

<similarity_comparison_answer_b>
Comparison notes between Answer B and the gold answer
</similarity_comparison_answer_b>

<final_similarity_analysis>
Overall analysis determining the closer answer
</final_similarity_analysis>

<final_answer>
Answer X (where X is the option you pick)
</final_answer>
```

# Notes

- Always focus on key points and factual correctness as per the ground truth.
- Avoid any biases and rely solely on the evidence presented.
- Enclose all evaluations and analyses in the specified XML tags for clarity and structure."""

JUDGE_ANSWER_USER_PROMPT = """<document_summary>
{summary}
</document_summary>

<piece_of_text>
{chunk}
</piece_of_text>

<question>
{question}
</question>

<gold_answer>
{oracle_answer}
</gold_answer>

<answer_a>
{answer_a}
</answer_a>

<answer_b>
{answer_b}
</answer_b>"""

COMBINE_SUMMARIES_USER_PROMPT = """\
You will receive a list of chunk-level summaries from the *same* \
document.  Combine them into a single, well-structured paragraph that reads \
naturally and eliminates redundancy.

<chunk_summaries>
{chunk_summaries}
</chunk_summaries>

Return ONLY the final text inside <final_summary> tags."""
