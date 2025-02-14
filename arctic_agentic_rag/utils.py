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
import os
from collections.abc import Iterator
from typing import Callable, Optional


def get_api_key(api_type: str) -> Optional[str]:
    """
    Fetches the right api_key from environment variable.

    Args:
        api_type (str): The type of API used, e.g., "openai".

    Returns:
        str: The api_key, if exists. Otherwise returns "None".
    """
    if api_type == "openai":
        return os.environ.get("OPENAI_API_KEY")

    return None


def get_data_processor_from_filename(input_file: str) -> Callable[[dict], dict]:
    def example_to_inputs_crag(example: dict) -> dict:
        return {
            "question": example["question"],
            "contexts": [x["excerpt"] for x in example["search_results"]],
            "ground_truth": example["answer"],
        }

    def example_to_inputs_marco(example: dict) -> dict:
        return {
            "question": example["query"],
            "contexts": [x["passage_text"] for x in example["passages"]],
            "ground_truth": [example["answers"]],
        }

    def example_to_inputs_nq(example: dict) -> dict:
        return {
            "id": example["id"],
            "question": example["question"],
            "ground_truth": example["answers"],
        }

    if "crag" in input_file:
        return example_to_inputs_crag
    elif "msmarco" in input_file:
        return example_to_inputs_marco
    elif "nq" in input_file:
        return example_to_inputs_nq
    else:
        raise NotImplementedError


def stream_from_file(file_path: str) -> Iterator[dict]:
    """
    Streams content from a JSON or JSONL file.

    Args:
        file_path (str): Path to the file to be streamed.

    Yields:
        dict: Each element from a JSON list (for `.json` files) or each JSON object from `.jsonl` files.

    Raises:
        ValueError: If the JSON file does not contain a list.
        NotImplementedError: If the file extension is not `.json` or `.jsonl`.
    """
    if file_path.endswith(".json"):
        # JSON file: Assume it contains a list and stream each element
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise ValueError("JSON file does not contain a list.")
            yield from data
    elif file_path.endswith(".jsonl"):
        # JSONL file: Read line by line, yield each line as a JSON object
        with open(file_path, encoding="utf-8") as file:
            for line in file:
                yield json.loads(line.strip())
    else:
        raise NotImplementedError("Only .json and .jsonl files are supported.")


def get_snowflake_credentials(
    authentication: str = "password",
) -> dict[str, Optional[str]]:
    connection_params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "role": os.environ["SNOWFLAKE_ROLE"],
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
    }

    if authentication == "password":
        connection_params.update({"password": os.environ["SNOWFLAKE_PASSWORD"]})
    elif authentication == "sso":
        connection_params.update({"authenticator": "externalbrowser"})
    else:
        raise NotImplementedError(
            f"Authentication type {authentication} is not supported."
        )

    return connection_params


def get_snowflake_search_params() -> dict[str, Optional[str]]:
    service_params = {
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
        "service_name": os.environ.get("SNOWFLAKE_CORTEX_SEARCH_SERVICE"),
    }
    return service_params
