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

# This section is inspired by DSPy: https://github.com/stanfordnlp/dspy

import os
from typing import Any, Callable, Union, cast

import yaml

from .field import Field, ListField


class TemplateMeta(type):
    def __new__(mcs: type[type], name: str, bases: tuple, class_dict: dict) -> type:
        # Collect all `Field` instances defined in the class
        fields = {
            key: value for key, value in class_dict.items() if isinstance(value, Field)
        }

        # Attach the collected fields as a class attribute
        class_dict["_fields"] = fields

        examples = class_dict.get("examples", [])
        class_dict["_examples"] = examples

        # Create the class
        cls = type.__new__(mcs, name, bases, class_dict)
        return cls


class Template(metaclass=TemplateMeta):
    _fields: dict = {}  # To pass static type check
    _examples: list = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Validation logic
        fields = cls._fields
        if not any(f.mode == "input" for f in fields.values()):
            raise TypeError(f"{cls.__name__} must define at least one input field.")
        if not any(f.mode == "output" for f in fields.values()):
            raise TypeError(f"{cls.__name__} must define at least one output field.")
        if any(f.mode not in ("input", "output") for f in fields.values()):
            raise TypeError("Currently, only input and output fields are used.")
        cls._validate_examples()

    @classmethod
    def get_fields(cls) -> dict[str, Field]:
        return cls._fields

    @classmethod
    def _validate_examples(cls) -> None:
        """Ensures that few-shot examples match the defined input and output fields.

        Raises:
            ValueError: If there are missing/surplus fields provided.
            TypeError: If the type of value is not valid, for instance, something other than str for Field.
        """
        if not hasattr(cls, "_examples"):
            return  # No examples provided

        defined_fields = cls.get_fields()  # Get field definitions

        for idx, example in enumerate(cls._examples):
            example_fields = set(example.keys())

            if example_fields != set(defined_fields.keys()):
                raise ValueError(
                    f"Example {idx + 1} does not match template fields!\n"
                    f"Expected fields: {set(defined_fields.keys())}\n"
                    f"Got fields: {example_fields}"
                )

            # Validate field types
            for name, field in defined_fields.items():
                value = example[name]

                if isinstance(field, ListField) and not isinstance(value, list):
                    raise TypeError(
                        f"Example {idx + 1}, field '{name}' should be a list (due to"
                        f" ListField), but got {type(value).__name__}."
                    )
                if not isinstance(field, ListField) and not isinstance(value, str):
                    raise TypeError(
                        f"Example {idx + 1}, field '{name}' should be a string, but got"
                        f" {type(value).__name__}."
                    )

    @classmethod
    def build_system_prompt(cls) -> str:
        """Constructs the system prompt.

        Returns:
            str: The system prompt.
        """
        input_fields, output_fields = [], []
        for name, field in cls.get_fields().items():
            field_dict = {"name": field.display_name, "description": field.desc}
            if field.mode == "input":
                input_fields.append(field_dict)
            elif field.mode == "output":
                output_fields.append(field_dict)

        input_descriptions = "\n".join(
            f"{field['name']}: {field['description']}" for field in input_fields
        )
        output_descriptions = "\n".join(
            f"{field['name']}: {field['description']}" for field in output_fields
        )

        system_prompt = (
            f"Your input fields are:\n{input_descriptions}\n\nYour output fields"
            f" are:\n{output_descriptions}\n\nAll interactions will be structured in"
            " the following way, with the appropriate values filled in:\n\n"
        )

        for f in input_fields + output_fields:
            system_prompt += f"[[ ## {f['name']} ## ]]\n{{{f['name']}}}\n\n"

        system_prompt += "[[ ## Completed ## ]]\n\n"
        system_prompt += "In adhering to this structure, your objective is:\n"

        if cls.__doc__:
            system_prompt += f"{cls.__doc__.strip()}\n"

        # Add few-shot examples, if there are any
        examples = getattr(cls, "_examples", [])
        if examples:
            for i, example in enumerate(examples, start=1):
                system_prompt += (
                    f"\nExample {i}.\n" if len(examples) > 1 else "\nExample.\n"
                )
                for name, field in cls.get_fields().items():
                    value = example[name]
                    # Format ListField correctly
                    formatted_value = field.format(
                        value
                    )  # if isinstance(field, ListField) else value
                    system_prompt += (
                        f"[[ ## {field.display_name} ## ]]\n{formatted_value}\n\n"
                    )

        return system_prompt

    @classmethod
    def build_user_prompt(cls, **inputs: Any) -> str:
        """Constructs the user message by populating input field values.

        Args:
            **inputs: Keyword arguments specifying values of input fields.

        Returns:
            str: The user message.
        """
        user_prompt = "We have provided the following input fields:\n\n"
        for name, field in cls.get_fields().items():
            if field.mode == "input":
                value = inputs.get(name, "<placeholder>")

                user_prompt += (
                    f"[[ ## {field.display_name} ## ]]\n{field.format(value)}\n\n"
                )

        # Add response instructions for output fields
        output_fields = [
            field.display_name
            for name, field in cls.get_fields().items()
            if field.mode == "output"
        ]

        if output_fields:
            if len(output_fields) == 1:
                user_prompt += (
                    "Respond with the corresponding output field, starting with the"
                    f" field [[ ## {output_fields[0]} ## ]], and then ending with the"
                    " marker for [[ ## Completed ## ]].\n"
                )
            else:
                output_field_list = "".join(
                    f"[[ ## {name} ## ]]\n" for name in output_fields
                )
                user_prompt += (
                    "Respond with the corresponding output fields in the following"
                    f" order:\n{output_field_list}\nFinally, end with the marker for [["
                    " ## Completed ## ]].\n"
                )

        return user_prompt

    @classmethod
    def parse_response(cls, response: str) -> dict[str, Union[bool, str]]:
        """Parses the response from the LLM to extract the output fields.
        If parsing fails, the validity flag (_valid) is set as False.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            dict[str, Union[bool, str]]: A dictionary mapping output field names to their extracted values.

        # noqa: DAR401
        """
        # Ensure cls.get_fields() returns a valid object
        fields = cls.get_fields()
        if not fields or not isinstance(fields, dict):
            raise TypeError("cls.get_fields() returned None or an invalid type.")

        output_fields = {
            field.display_name: name
            for name, field in fields.items()
            if field.mode == "output"
        }

        # Initialize parsed output with "" for all output fields
        parsed_output = {name: "" for name in output_fields.values()}
        completed_marker = "[[ ## Completed ## ]]"

        # Ensure response is a valid string before processing
        if not isinstance(response, str) or not response.strip():
            return {
                "_valid": False,
                **parsed_output,
            }  # If response is None or empty, return all ""

        response = response.strip()

        # Handle missing `[[ ## Completed ## ]]`
        completed_index = response.find(completed_marker)
        if completed_index != -1:
            response = response[:completed_index]  # Trim response safely

        # Detect invalid responses: If no expected field markers exist, return all ""
        has_valid_field = any(
            f"[[ ## {display_name} ## ]]" in response for display_name in output_fields
        )
        if not has_valid_field:
            return {"_valid": False, **parsed_output}

        # Process each expected field
        is_valid = True
        for display_name, identifier in output_fields.items():
            try:
                marker_start = f"[[ ## {display_name} ## ]]"
                start_idx = response.find(marker_start)

                if start_idx != -1:
                    start_idx += len(marker_start)

                    # Extract until the next marker or end of response
                    end_idx = response.find("[[ ## ", start_idx)
                    extracted_content = (
                        response[start_idx:end_idx]
                        if end_idx != -1
                        else response[start_idx:]
                    ).strip()

                    # Clean extracted content
                    parsed_output[identifier] = (
                        extracted_content if extracted_content else ""
                    )

            except (TypeError, AttributeError, ValueError):
                parsed_output[identifier] = ""

            finally:
                if parsed_output[identifier] == "":
                    is_valid = False

        return {"_valid": is_valid, **parsed_output}


def set_instructions(instructions: str) -> Callable:
    def wrapper(cls: type[Template]) -> type[Template]:
        cls.__doc__ = instructions
        return cls

    return wrapper


def create_template_from_yaml(yaml_str: str, base_path: str) -> type[Template]:
    """
    Create a Template-derived class from a YAML definition.

    Args:
        yaml_str (str): A YAML-formatted string defining the template.
        base_path (str): The base path for resolving system prompt file paths.

    Returns:
        type[Template]: A dynamically created Template subclass.

    Raises:
        ValueError: If the YAML definition is invalid.
        FileNotFoundError: If the path to the file defining system prompt is invalid.
    """
    data = yaml.safe_load(yaml_str)

    if "name" not in data:
        raise ValueError("The YAML must specify a 'name' for the template.")
    if "fields" not in data:
        raise ValueError("The YAML must specify 'fields' for the template.")

    class_name = data["name"]
    fields = data["fields"]

    description = data.get("description", None)
    if description is None:
        path_to_system_prompt = data.get("system_prompt_file", "")
        path_to_system_prompt = os.path.join(base_path, path_to_system_prompt)
        if os.path.isfile(path_to_system_prompt):
            with open(path_to_system_prompt) as file:
                description = file.read()
        else:
            raise FileNotFoundError(
                f"The provided path {path_to_system_prompt} does not exist or is not a"
                " regular file."
            )

    # Prepare the class dictionary
    class_dict = {}
    for field_name, field_spec in fields.items():
        desc = field_spec.get("desc", "")
        mode = field_spec.get("mode", "")
        display_name = field_spec.get("display_name", field_name.title())

        new_field: Field
        if "formatting" in field_spec:
            # Is a ListField
            formatting = field_spec["formatting"]
            new_field = ListField(
                desc=desc, mode=mode, display_name=display_name, formatting=formatting
            )
        else:
            # Is a regular Field
            new_field = Field(desc=desc, mode=mode, display_name=display_name)
        class_dict[field_name] = new_field

    if description:
        class_dict["__doc__"] = description.strip()

    return cast(type[Template], type(class_name, (Template,), class_dict))
