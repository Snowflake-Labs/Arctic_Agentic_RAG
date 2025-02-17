# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Prompts for ambiguity detection"""

PROMPT_V1 = """
You are an agent tasked with determining whether a given query and its associated passage texts are ambiguous or not. Your analysis should be thorough and your output should be a clean, parsable JSON object.

Here is the query you will analyze:
<query>
{query}
</query>

And here are the associated passage texts:
<passage_texts>
{passage_text}
</passage_texts>

Carefully read and analyze the query and passage texts. Consider the following aspects:

1. Ambiguity: Determine if the query or passages have multiple possible interpretations or lack clarity.
2. Key interpretations: Identify the main ways the query could be understood, if any.
3. Intent clarity: Assess how clear the intent of the query is.
4. Reasoning depth: Evaluate how much reasoning is required to understand and answer the query.

Format your output as a JSON object with the following structure:
{{
    "ambiguity_status": "Ambiguous" or "Not Ambiguous",
    "reasoning_depth": 0 or 1,
    "query": "{query}",
    "key_interpretations": ["interpretation1", "interpretation2", ...],
    "intent_clarity": 0 or 1,
    "reasoning_summary": "<concise explanation>",
    "referenced_passages": ["<passage 1>", "<passage 2>", ...]
}}"""

PROMPT_V2 = """
You are an AI language model tasked with analyzing a query and its associated passages to determine if the query is ambiguous. Your goal is to improve query understanding and response accuracy in an information retrieval system.

Here are the associated passages:
<passages>
{passage_text}
</passages>

Here is the query you need to analyze:
<query>
{query}
</query>

Format your output as a JSON object with the following structure:
{{
    "ambiguity_status": "Ambiguous" or "Not Ambiguous",
    "reasoning_depth": 0 or 1,
    "query": "{query}",
    "key_interpretations": ["interpretation1", "interpretation2", ...],
    "intent_clarity": 0 or 1,
    "reasoning_summary": "<concise explanation>",
    "referenced_passages": ["<passage 1>", "<passage 2>", ...]
}}"""

PROMPT_V3 = """
You are an AI language model tasked with analyzing a query and its associated passages to determine if the query is ambiguous.
**Strictly output only a valid JSON object. Do not include any text, explanations, or code fences (```), before or after the JSON.**

Here is the query you need to analyze:
<query>
{query}
</query>

Here are the associated passages:
<passages>
{passage_text}
</passages>

Analyze the following aspects:
1. Inherent Ambiguity: Is the query itself unclear or open to multiple interpretations?
2. Contextual Clues: Do passages clarify or confuse the query's meaning?
3. Query Specificity: Is the query too broad or too narrow?
4. User Intent: Is the information being sought clear?

Output Format:
{{
    "ambiguity_status": "Ambiguous" or "Not Ambiguous",
    "reasoning_depth": 0 or 1,
    "query": "{query}",
    "key_interpretations": ["interpretation1", "interpretation2", ...],
    "intent_clarity": 0 or 1,
    "reasoning_summary": "<concise explanation>",
    "referenced_passages": ["<passage 1>", "<passage 2>", ...]
}}"""

PROMPT_V4 = """
You are an AI language model tasked with analyzing a query and its associated passages to determine if the query is ambiguous. Your goal is to improve query understanding and response accuracy.

Here are the associated passages:
<passages>
{passage_text}
</passages>

Here is the query you need to analyze:
<query>
{query}
</query>

Analysis Process:
1. Consider if the query itself is valid and clear
2. List all possible interpretations, including edge cases
3. Evaluate interpretations against passages
4. Quote relevant passage parts for each interpretation
5. Assess query specificity
6. Consider user intent
7. Determine if multiple plausible interpretations exist
8. Consider arguments for/against ambiguity
9. Make final decision
10. Err on marking queries as ambiguous if in doubt

Output Format:
{{
    "ambiguity_status": "Ambiguous" or "Not Ambiguous",
    "reasoning_depth": 0 or 1,
    "query": "{query}",
    "key_interpretations": ["interpretation1", "interpretation2", ...],
    "intent_clarity": 0 or 1,
    "reasoning_summary": "<concise explanation>",
    "referenced_passages": ["<passage 1>", "<passage 2>", ...]
}}
"""
