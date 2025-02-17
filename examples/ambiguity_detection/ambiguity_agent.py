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


from typing import Any

from ambiguity.router import AmbiguityRouter

from arctic_agentic_rag.agent import PromptAgent


class AmbiguityDetectionAgent(PromptAgent):
    """Agent for detecting query ambiguity"""

    def __init__(self, prompt_version: str = "v4", **kwargs: Any) -> None:
        """
        Initialize the ambiguity detection agent.

        Args:
            prompt_version: Version of the prompt template to use.
            **kwargs: Additional arguments passed to parent class.
        """
        # Define system prompt
        system_prompt = """You are an expert system designed to detect ambiguity in queries.
        Your task is to analyze queries and determine if they have multiple possible interpretations
        or lack clarity. Provide detailed analysis and clear reasoning for your decisions."""

        # Initialize parent class with system prompt
        kwargs["system_prompt"] = system_prompt  # Add system_prompt to kwargs
        super().__init__(**kwargs)

        # Initialize router
        self.router = AmbiguityRouter(
            backbone=self.backbone, logger=self.logger, prompt_version=prompt_version
        )

    @property
    def required_keys(self) -> list[str]:
        """Get required input keys."""
        return ["query", "passages"]

    def promptize(self, **inputs: Any) -> str:
        """
        Convert inputs to prompt format.

        Args:
            **inputs: Dictionary containing query and passages.

        Returns:
            str: The formatted prompt string.
        """
        query = inputs.get("query", "")
        passages = inputs.get("passages", [])
        return f"Query: {query}\nPassages: {' '.join(passages)}"

    def parse_response(self, response: str) -> dict[str, Any]:
        """
        Parse the router response.

        Args:
            response: Raw response string from the router.

        Returns:
            Dict[str, Any]: Parsed response dictionary.
        """
        if not response:
            return {"error": "Empty response"}
        return {"response": response}

    def get_result(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Get ambiguity detection result for input query and passages.

        Args:
            inputs: Dict containing:
                - query: The query to analyze
                - passages: List of relevant passages

        Returns:
            Dict[str, Any]: Dictionary containing ambiguity analysis result.
        """
        try:
            query = inputs["query"]
            passages = inputs["passages"]

            result = self.router.detect_ambiguity(query, passages)

            self.logger.log_final_results(
                {"query": query, "ambiguity_result": result}  # type: ignore
            )

            return result if isinstance(result, dict) else {"error": "Invalid response"}

        except Exception as e:
            self.logger.log_error(f"Error in ambiguity detection: {str(e)}")  # type: ignore
            return {"error": f"Detection failed: {str(e)}"}
