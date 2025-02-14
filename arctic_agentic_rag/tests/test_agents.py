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


from unittest.mock import MagicMock

from arctic_agentic_rag.agent import (
    PromptAgent,
    TemplateAgent,
    agent_name_to_agent_class,
)
from arctic_agentic_rag.llm import BaseLLM
from arctic_agentic_rag.logging import Logger
from arctic_agentic_rag.template import Template


def test_prompt_agent_initialization():
    """
    Tests the initialization of the PromptAgent class.

    This test performs the following steps:
    1. Instantiates a PromptAgent with mock dependencies.
    2. Asserts that the created agent is an instance of PromptAgent.
    3. Verifies that the agent's system prompt and action space are set correctly.
    """
    mock_llm = MagicMock(spec=BaseLLM)
    mock_logger = MagicMock(spec=Logger)
    system_prompt = "Test System Prompt"
    action_space = ["action1", "action2"]

    class TestPromptAgent(PromptAgent):
        @property
        def required_keys(self):
            return ["input_key"]

        def promptize(self, **inputs):
            return f"Formatted: {inputs['input_key']}"

        def parse_response(self, raw_response: str):
            return raw_response.upper()

    agent = TestPromptAgent(
        backbone=mock_llm,
        uid="test_agent",
        logger=mock_logger,
        system_prompt=system_prompt,
        action_space=action_space,
    )

    assert isinstance(agent, PromptAgent), "Agent should be an instance of PromptAgent."
    assert (
        agent.system_prompt == system_prompt
    ), "Agent's system_prompt should be correctly set."
    assert (
        agent.action_space == action_space
    ), "Agent's action_space should be correctly set."


def test_template_agent_initialization():
    """
    Tests the initialization of the TemplateAgent class.

    This test performs the following steps:
    1. Instantiates a TemplateAgent with a mock template and mock dependencies.
    2. Asserts that the created agent is an instance of TemplateAgent.
    3. Verifies that the agent's system prompt is derived from the template.
    """
    mock_llm = MagicMock(spec=BaseLLM)
    mock_logger = MagicMock(spec=Logger)

    mock_template = MagicMock(spec=Template)
    mock_template.build_system_prompt.return_value = "Mock System Prompt"
    mock_template.build_user_prompt.return_value = "Mock User Prompt"
    mock_template.parse_response.return_value = "Parsed Response"

    agent = TemplateAgent(
        backbone=mock_llm,
        uid="test_template_agent",
        logger=mock_logger,
        template=mock_template,
    )

    assert isinstance(
        agent, TemplateAgent
    ), "Agent should be an instance of TemplateAgent."
    assert (
        agent.system_prompt == "Mock System Prompt"
    ), "Agent's system_prompt should be derived from the template."


def test_template_agent_promptize():
    """
    Verifies that the TemplateAgent correctly utilizes its template to format prompts.
    """
    mock_llm = MagicMock(spec=BaseLLM)
    mock_logger = MagicMock(spec=Logger)

    mock_template = MagicMock(spec=Template)
    mock_template.build_system_prompt.return_value = "Mock System Prompt"
    mock_template.build_user_prompt.return_value = "Formatted Prompt"

    agent = TemplateAgent(
        backbone=mock_llm,
        uid="test_template_agent",
        logger=mock_logger,
        template=mock_template,
    )

    prompt = agent.promptize(input_data="test_input")
    assert (
        prompt == "Formatted Prompt"
    ), "TemplateAgent should use template to format prompts."


def test_template_agent_parse_response():
    """
    Verifies that the TemplateAgent correctly delegates response parsing to the template.
    """
    mock_llm = MagicMock(spec=BaseLLM)
    mock_logger = MagicMock(spec=Logger)

    mock_template = MagicMock(spec=Template)
    mock_template.build_system_prompt.return_value = "Mock System Prompt"
    mock_template.parse_response.return_value = "Parsed Response"

    agent = TemplateAgent(
        backbone=mock_llm,
        uid="test_template_agent",
        logger=mock_logger,
        template=mock_template,
    )

    response = agent.parse_response("raw response")
    assert (
        response == "Parsed Response"
    ), "TemplateAgent should use template to parse responses."


def test_agent_class_mapping():
    """
    Verifies that the agent class registry contains the expected mappings.

    This test performs the following steps:
    1. Verifies that 'TemplateAgent' maps to the PromptAgent class.
    2. Verifies that 'TemplateAgent' maps to the TemplateAgent class.
    """

    # Assert that 'TemplateAgent' maps to the correct agent class.
    assert (
        agent_name_to_agent_class["PromptAgent"] == PromptAgent
    ), "'PromptAgent' should map to TemplateAgent."

    # Assert that 'TemplateAgent' maps to the correct agent class.
    assert (
        agent_name_to_agent_class["TemplateAgent"] == TemplateAgent
    ), "'TemplateAgent' should map to TemplateAgent."
