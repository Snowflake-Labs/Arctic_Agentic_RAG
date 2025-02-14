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


from openai import APIConnectionError, OpenAI, RateLimitError

from ..logging import Logger
from ..utils import get_api_key
from .base_llm_calling import BaseLLM, Message


class OAI(BaseLLM):
    def __init__(
        self,
        model: str,
        uid: str,
        logger: Logger,
        max_retries: int = None,
    ) -> None:
        super().__init__(model, uid=uid, logger=logger)
        self.max_retries = max_retries
        self.api_key = get_api_key("openai")
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, prompt, **kwargs):
        self.logger.log_error("OpenAI client does not support using prompts")
        raise NotImplementedError

    def generate_response_from_messages(
        self,
        messages: list[Message],
        **kwargs,
    ) -> str:
        """
        Answer a given request using the target language model.

        Args:
            messages (list[Message]): The messages containing the request.
            topk (int, optional): k for top-k sampling, 0 if unused.
            topp (float, optional): p for nucleus sampling, 0.0 if unused.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The resulting response string.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    seed=kwargs.get("seed", 0),
                    temperature=kwargs.get("temperature", 0.0),
                    top_p=kwargs.get("top_p", 1.0),
                )
                return response.choices[0].message.content
            except (APIConnectionError, RateLimitError):
                print(f"Failed API call on attempt {attempt}")
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        return None

    def get_usage_stats(self) -> dict:
        return {}
