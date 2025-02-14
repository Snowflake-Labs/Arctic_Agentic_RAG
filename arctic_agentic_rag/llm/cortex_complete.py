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


from typing import Optional

from snowflake.cortex import Complete, CompleteOptions
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkClientException

from ..logging import Logger
from ..utils import get_snowflake_credentials  # Modified credentials helper
from .base_llm_calling import BaseLLM, Message


class CortexComplete(BaseLLM):
    def __init__(
        self,
        model: str,
        uid: str,
        logger: Logger,
        max_retries: int = 3,
        authentication: str = "password",
    ) -> None:
        super().__init__(model, uid=uid, logger=logger)
        self.max_retries = max_retries
        self.authentication = authentication
        self.session = self._create_session()

    def _create_session(self) -> Session:
        """
        Create authenticated Snowflake session using credentials.

        Returns:
            Session: A snowflake session to be used.
        """
        creds = get_snowflake_credentials(
            self.authentication
        )  # Should return dict with connection params
        return Session.builder.configs(creds).create()

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Single-prompt interface for compatibility.
        """
        return self.generate_response_from_messages(
            [Message(role="user", content=prompt)], **kwargs
        )

    def generate_response_from_messages(
        self,
        messages: list[Message],
        **kwargs,
    ) -> Optional[str]:
        """
        Generate response using Snowflake Cortex Complete

        Args:
            messages: List of Message objects containing conversation history
            temperature: Controls randomness (0.0 = deterministic)
            top_p: Controls diversity via nucleus sampling
            max_tokens: Maximum length of generated response

        Returns:
            Generated response string or None if failed
        """
        options = CompleteOptions(
            temperature=kwargs.get("temperature", 0.01),
            top_p=kwargs.get("top_p", 0.99),
            max_tokens=kwargs.get("max_tokens", 512),
        )

        # Convert messages to prompt (customize based on your needs)
        prompt = "\n".join(
            [
                f"{msg['role'] if isinstance(msg, dict) else msg.role}:"
                f" {msg['content'] if isinstance(msg, dict) else msg.content}"
                for msg in messages
            ]
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = Complete(
                    model=self.model,
                    prompt=prompt,
                    session=self.session,
                    options=options,
                    stream=False,
                )
                return response
            except SnowparkClientException as e:
                self.logger.log_error(f"Attempt {attempt} failed: {str(e)}")
                if attempt == self.max_retries:
                    raise
            except Exception as e:
                self.logger.log_error(f"Unexpected error: {str(e)}")
                break

        return None

    def get_usage_stats(self) -> dict:
        """Return Snowflake query usage statistics"""
        # Implement based on your Snowflake usage tracking needs
        return {"service": "snowflake-cortex", "model": self.model}
