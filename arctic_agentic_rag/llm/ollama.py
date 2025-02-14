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

import requests

from .base_llm_calling import BaseLLM, Message


class Ollama(BaseLLM):
    def __init__(self, base_url: str, **kwargs):
        """
        Initialize the Ollama API with the base URL.
        https://github.com/ollama/ollama/blob/main/docs/api.md#parameters
        """
        super().__init__(**kwargs)
        self.base_url = base_url
        self.input_token_usage = 0
        self.output_token_usage = 0

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the Ollama API.
        Handles streamed responses.
        """
        return self._generate_response("/completions", {"prompt": prompt}, **kwargs)

    def generate_response_from_messages(self, messages: list[Message], **kwargs) -> str:
        return self._generate_response(
            "/chat/completions", {"messages": messages}, **kwargs
        )

    def _generate_response(self, url_suffix: str, request_dict: dict, **kwargs) -> str:
        url = self.base_url + url_suffix
        payload = {
            **request_dict,
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()

            # Collect streamed chunks
            result = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        result += data["choices"][0]["messages"]["content"]
                        self.input_token_usage += data["usage"]["prompt_tokens"]
                        self.output_token_usage += data["usage"]["completion_tokens"]
                        break
                    except json.JSONDecodeError as e:
                        self.logger.log_error(
                            "Failed to decode JSON while streaming model responses"
                        )
                        return f"Error decoding JSON: {e}"
            return result
        except Exception as e:
            return f"Error: {e}"

    def get_usage_stats(self) -> dict:
        """
        Retrieve usage statistics (mock implementation).
        """
        return {
            "input_tokens": self.input_token_usage,
            "output_tokens": self.output_token_usage,
            "total_tokens_used": self.input_token_usage + self.output_token_usage,
        }
