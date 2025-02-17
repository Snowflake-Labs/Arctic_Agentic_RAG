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
from typing import Any

from ..component import Component
from ..llm import BaseLLM
from ..logging import Logger
from ..template import Template


class PromptAgent(Component, ABC):
    def __init__(
        self,
        backbone: BaseLLM,
        uid: str,
        logger: Logger,
        system_prompt: str,
        action_space: list[str] = [],
        **llm_kwargs: Any,
    ) -> None:
        super().__init__(uid=uid, logger=logger)
        self.backbone = backbone
        self.system_prompt = system_prompt
        self.action_space = action_space
        self.llm_kwargs = llm_kwargs

        self.history: list[dict[str, Any]] = []

    def get_result(self, inputs: dict[str, Any]) -> Any:
        messages = self.construct_messages(inputs)

        raw_response = self.backbone.generate_response_from_messages(
            messages,
            **self.llm_kwargs,
        )
        self.logger.log_info(f"LLM Output:\n {raw_response} \n")

        result = self.parse_response(raw_response)
        return result

    def construct_messages(self, inputs: dict[str, Any]) -> list[dict[str, str]]:
        """
        Converts the input, using the task-specific system prompt and others, to messages.

        Args:
            inputs (dict): The relevant inputs to be consumed by this agent.

        Returns:
            list[dict[str, str]]: The list of messages to be fed to the backbone BaseLLM.
        """
        for key in self.required_keys:
            if key not in inputs:
                raise ValueError(
                    f"{self.__class__.__name__} needs '{key}' in inputs, but missing"
                )

        prompt = self.promptize(**inputs)
        self.logger.log_info(f"System Prompt:\n {self.system_prompt} \n")
        self.logger.log_info(f"User Prompt:\n {prompt} \n")

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    @property
    @abstractmethod
    def required_keys(self) -> list[str]:
        """
        The keys that must appear in `inputs` dictionary for `construct_messages`.
        """
        pass

    @abstractmethod
    def promptize(self, **inputs: Any) -> str:
        """
        Construct a string message that can be consumed by the LLM.
        """
        pass

    @abstractmethod
    def parse_response(self, raw_response: str) -> Any:
        """
        Parses the raw response and extracts the relevant part.
        """
        pass


class TemplateAgent(PromptAgent):
    def __init__(self, template: type[Template], **kwargs: Any) -> None:
        system_prompt = template.build_system_prompt()
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.template = template

    @property
    def required_keys(self) -> list[str]:
        return []

    def promptize(self, **inputs: Any) -> str:
        return self.template.build_user_prompt(**inputs)

    def parse_response(self, raw_response: str) -> Any:
        return self.template.parse_response(raw_response)
