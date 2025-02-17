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


import logging

from vllm import LLM, SamplingParams

from arctic_agentic_rag.llm.oai import OAI
from arctic_agentic_rag.logging import Logger

from .base_llm_calling import BaseLLM, Message


class VLLMProvider(BaseLLM):
    def __init__(
        self,
        model: str,
        uid: str,
        logger: logging.Logger,
        max_tokens: int = 2048,
        **kwargs,
    ) -> None:
        """
        Initialize the VLLM provider.

        Args:
            model (str): Name or path of the model to load
            uid (str): Unique identifier for this provider instance
            logger (logging.Logger): Standard Python logger instance
            max_tokens (int, optional): Maximum number of tokens in output. Defaults to 2048.
        """
        self.model = model
        self.uid = uid
        self.logger = logger
        self.max_tokens = max_tokens

        engine_kwargs = {
            "model": model,
            "trust_remote_code": True,
        }

        try:
            self.logger.info("Initializing VLLM...")
            self.engine = LLM(**engine_kwargs)
            self.logger.info(f"Successfully initialized VLLM with model: {model}")
        except Exception as e:
            error_msg = f"Failed to initialize VLLM engine: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a given prompt.

        Args:
            prompt (str): The input prompt for the LLM
            **kwargs: Additional parameters for generation (temperature, top_p, etc.)

        Returns:
            str: Generated response
        """
        try:
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", 1.0),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            self.logger.debug(f"Generating response for prompt: {prompt[:100]}...")
            outputs = self.engine.generate([prompt], sampling_params)

            if not outputs:
                self.logger.error("VLLM generated no output")
                return ""

            response = outputs[0].outputs[0].text
            self.logger.debug(f"Generated response: {response[:100]}...")
            return response

        except Exception as e:
            error_msg = f"VLLM generation error: {str(e)}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"

    def generate_response_from_messages(
        self,
        messages: list[Message],
        **kwargs,
    ) -> str:
        """
        Generate a response from a list of messages.

        Args:
            messages (list[Message]): The messages containing the conversation history
            **kwargs: Additional parameters for generation

        Returns:
            str: Generated response
        """
        try:
            # Convert messages to a chat format that VLLM can understand
            prompt = self._convert_messages_to_prompt(messages)

            return self.generate_response(prompt, **kwargs)

        except Exception as e:
            error_msg = f"VLLM message generation error: {str(e)}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"

    def _convert_messages_to_prompt(self, messages: list[Message]) -> str:
        """
        Convert a list of messages to a single prompt string.

        Args:
            messages (list[Message]): List of message dictionaries

        Returns:
            str: Combined prompt string
        """
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        return prompt

    def get_usage_stats(self) -> dict:
        """
        Get usage statistics for the VLLM engine.
        Note: VLLM doesn't provide built-in token counting,
        so this returns basic model information.

        Returns:
            dict: Usage statistics
        """
        try:
            return {"model": self.model, "provider": "vllm"}
        except Exception as e:
            error_msg = f"Error getting VLLM stats: {str(e)}"
            self.logger.error(error_msg)
            return {}


class VLLMOaiServer(OAI):
    """Integration for VLLM running in OpenAI-compatible server mode"""

    def __init__(
        self,
        model: str,
        uid: str,
        logger: Logger,
        base_url: str = "http://localhost:8000/v1",
        max_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(model=model, uid=uid, logger=logger, max_retries=max_retries)
        self.base_url = base_url
        # Override the OpenAI client with custom base URL
        self.client.base_url = self.base_url
