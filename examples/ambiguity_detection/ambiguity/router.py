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


import json
import logging
from typing import Any

from arctic_agentic_rag.llm import BaseLLM
from arctic_agentic_rag.logging import Logger
from prompts import PROMPT_V1, PROMPT_V2, PROMPT_V3, PROMPT_V4


logger = logging.getLogger(__name__)


class AmbiguityRouter:
    """Router for detecting query ambiguity"""

    PASSAGE_SEPARATOR = " [NEW_PASSAGE] "

    def __init__(
        self, backbone: BaseLLM, logger: Logger, prompt_version: str = "v4"
    ) -> None:
        self.backbone = backbone
        self.logger = logger
        self.prompt_version = self._get_prompt_version(prompt_version)

    def _get_prompt_version(self, version: str) -> str:
        """
        Get the appropriate prompt template.

        Args:
            version: The version of the prompt template to use.

        Returns:
            str: The selected prompt template.
        """
        prompts = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3, "v4": PROMPT_V4}
        return prompts.get(version, PROMPT_V4)

    def _build_prompt(self, query: str, passages: list[str]) -> str:
        """
        Build the prompt for ambiguity detection.

        Args:
            query: The query to analyze.
            passages: List of relevant passages to consider.

        Returns:
            str: The formatted prompt string.
        """
        combined_passages = self.PASSAGE_SEPARATOR.join(passages)
        return self.prompt_version.format(query=query, passage_text=combined_passages)

    def detect_ambiguity(self, query: str, passages: list[str]) -> dict[str, Any]:
        """
        Detect if a query is ambiguous given the context passages.

        Args:
            query: The query to analyze.
            passages: List of relevant passages.

        Returns:
            dict[str, Any]: Dictionary containing ambiguity analysis results.
        """
        try:
            prompt = self._build_prompt(query, passages)
            self.logger.log_debug("Built prompt for ambiguity detection")  # type: ignore

            response = self.backbone.generate_response_from_messages(
                [
                    {
                        "role": "system",
                        "content": "You are an ambiguity detection expert.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            self.logger.log_debug("Got response from LLM")  # type: ignore

            try:
                result = json.loads(response)
                self.logger.log_info("Successfully parsed ambiguity detection result")  # type: ignore
                return (
                    result
                    if isinstance(result, dict)
                    else {"error": "Invalid response format"}
                )
            except json.JSONDecodeError as e:
                self.logger.log_error(f"Failed to parse LLM response as JSON: {str(e)}")  # type: ignore
                return {"error": "Failed to parse response"}

        except Exception as e:
            self.logger.log_error(f"Error in ambiguity detection: {str(e)}")  # type: ignore
            return {"error": f"Detection failed: {str(e)}"}
