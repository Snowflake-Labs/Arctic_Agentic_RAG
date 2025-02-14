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
import uuid

import pytest  # type: ignore

from arctic_agentic_rag.llm import BaseLLM


# Mock LLM class to test BaseLLM
class MockLLM(BaseLLM):
    """
    A mock implementation of the BaseLLM class for testing purposes.

    This class overrides the generate_response and get_usage_stats methods
    to provide predictable outputs without relying on actual LLM functionality.
    """

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Simulates generating a response to a given prompt.

        Args:
            prompt (str): The input prompt to which the LLM should respond.
            **kwargs: Additional keyword arguments (not used in this mock).

        Returns:
            str: A fixed mock response.
        """
        return "This is a mock response."

    def get_usage_stats(self) -> dict:
        """
        Simulates retrieving usage statistics of the LLM.

        Returns:
            dict: A dictionary containing mock usage statistics.
        """
        return {"tokens_used": 10}


class ConcreteMockLLM(MockLLM):
    """
    A concrete implementation of MockLLM that provides an implementation
    for generate_response_from_messages method.
    """

    def generate_response_from_messages(self, messages) -> str:
        """
        Simulates generating a response based on a sequence of messages.

        Args:
            messages: A list or structure containing message data.

        Returns:
            str: A fixed mock response.
        """
        return "Mock response"


# Fixture to initialize a ConcreteMockLLM instance
@pytest.fixture
def llm_instance():
    """
    Pytest fixture to create and return a ConcreteMockLLM instance.
    """
    # Initialize a mock logger
    mock_logger = logging.getLogger("mock_logger")
    mock_logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG for detailed logs

    # Create a ConcreteMockLLM instance with mock parameters
    llm = ConcreteMockLLM(
        model="mock_model",  # Mock model name
        logger=mock_logger,  # Mock logger instance
        uid=str(uuid.uuid4()),  # Generate a unique identifier for the instance
    )
    return llm


def test_generate_response(llm_instance):
    """
    Tests whether the generate_response method returns the expected mock response.
    """
    # Define a sample prompt
    prompt = "Hello, how are you?"

    # Call the generate_response method with the sample prompt
    response = llm_instance.generate_response(prompt)

    # Assert that the response matches the expected mock response
    assert (
        response == "This is a mock response."
    ), "generate_response should return the predefined mock response."


def test_get_usage_stats(llm_instance):
    """
    Tests whether the get_usage_stats method returns the expected mock usage statistics.
    """
    # Call the get_usage_stats method to retrieve usage statistics
    stats = llm_instance.get_usage_stats()

    # Assert that the 'tokens_used' in stats matches the expected value
    assert (
        stats["tokens_used"] == 10
    ), "get_usage_stats should return {'tokens_used': 10}."


def test_generate_response_from_messages(llm_instance):
    """
    Tests whether the generate_response_from_messages method returns the expected mock response.
    """
    # Define a sample list of messages
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
    ]

    # Call the generate_response_from_messages method with the sample messages
    response = llm_instance.generate_response_from_messages(messages)

    # Assert that the response matches the expected mock response
    assert (
        response == "Mock response"
    ), "generate_response_from_messages should return the predefined mock response."


def test_uid_generation():
    """
    Tests whether each ConcreteMockLLM instance has a unique UID.
    """
    # Initialize first ConcreteMockLLM instance
    mock_logger1 = logging.getLogger("mock_logger1")
    mock_logger1.setLevel(logging.DEBUG)
    llm1 = ConcreteMockLLM(
        model="mock_model",
        logger=mock_logger1,
        uid=str(uuid.uuid4()),
    )

    # Initialize second ConcreteMockLLM instance
    mock_logger2 = logging.getLogger("mock_logger2")
    mock_logger2.setLevel(logging.DEBUG)
    llm2 = ConcreteMockLLM(
        model="mock_model_2",
        logger=mock_logger2,
        uid=str(uuid.uuid4()),
    )

    # Assert that the UIDs are not equal
    assert (
        llm1.uid != llm2.uid
    ), "Each ConcreteMockLLM instance should have a unique UID."


def test_logger_configuration(llm_instance):
    """
    Tests whether the logger is correctly configured in the ConcreteMockLLM instance.
    """
    # Retrieve the logger from the LLM instance
    logger = llm_instance.logger

    # Assert that the logger is an instance of logging.Logger
    assert isinstance(
        logger, logging.Logger
    ), "Logger should be an instance of logging.Logger."

    # Assert that the logger has the correct name
    assert logger.name == "mock_logger", "Logger name should be 'mock_logger'."


def test_model_attribute(llm_instance):
    """
    Tests whether the model attribute is correctly set in the ConcreteMockLLM instance.
    """
    # Assert that the model attribute matches the initialized value
    assert (
        llm_instance.model == "mock_model"
    ), "Model attribute should be set to 'mock_model'."
