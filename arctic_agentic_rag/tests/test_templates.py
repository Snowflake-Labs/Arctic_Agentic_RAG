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


import pytest  # type: ignore

from arctic_agentic_rag.template import create_template_from_yaml

# Define the YAML content as a constant for reusability
YAML_CONTENT = """
name: TestTemplate
system_prompt_file: dummy_prompt.txt
fields:
  input_field:
    desc: Test input field
    mode: input
  output_field:
    desc: Test output field
    mode: output
"""


@pytest.fixture
def temp_dir_with_prompt(tmp_path):
    """Fixture to create a temporary directory with a dummy system prompt file."""
    # Create the dummy prompt file
    prompt_file = tmp_path / "dummy_prompt.txt"
    prompt_file.write_text("Sample system prompt")

    # Return the temporary directory path as a string
    return str(tmp_path)


def test_template_from_yaml(temp_dir_with_prompt):
    """Test template creation from YAML with a system prompt file."""
    # Generate the TemplateClass using the YAML content and temp directory
    TemplateClass = create_template_from_yaml(YAML_CONTENT, temp_dir_with_prompt)

    # Instantiate the template class
    instance = TemplateClass()

    # Assert expected attributes exist
    assert hasattr(
        instance, "input_field"
    ), "Template instance should have 'input_field' attribute."
    assert hasattr(
        instance, "output_field"
    ), "Template instance should have 'output_field' attribute."
