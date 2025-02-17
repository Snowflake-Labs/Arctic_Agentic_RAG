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


from abc import ABC, abstractmethod
from typing import Literal, TypedDict

from ..component import Component


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class BaseLLM(Component, ABC):
    """
    Abstract base class for LLM API calling.
    This defines a consistent interface for different LLMs.
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a given prompt.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: Generated response.
        """
        pass

    @abstractmethod
    def generate_response_from_messages(self, messages: list[Message], **kwargs) -> str:
        """
        Generate a response to follow, for the given history of messages.

        Args:
            messages (list[Message]): The messages containing the request.

        Returns:
            str: Generated response.
        """
        pass

    @abstractmethod
    def get_usage_stats(self) -> dict:
        """
        Retrieve API usage statistics (e.g., token usage, cost).

        Returns:
            dict: Usage statistics.
        """
        pass
