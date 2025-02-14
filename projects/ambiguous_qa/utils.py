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
from typing import Any, Union


def load_from_file(path: Union[str, os.PathLike]) -> Any:
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    elif path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(x) for x in f]
    else:
        raise NotImplementedError


def get_instruction_and_examples(
    path: Union[str, os.PathLike],
) -> tuple[str, list[dict[str, str]]]:
    """
    Fetch instruction and few-shot examples (if any) from the given path.
    These can be used to build system messages.

    Args:
        path (Union[str, os.PathLike]): Path to the config file specifying
            instruction and few-shot examples for the task.

    Returns:
        tuple[str, list[dict[str, str]]]: A tuple containing the instruction
            string and a list of few-shot examples. Each example is a
            dictionary specifying values for input/output fields.
    """
    with open(path) as f:
        prompt_source = json.load(f)

    instruction = prompt_source["instruction"]
    fewshot_examples = prompt_source["examples"]
    return instruction, fewshot_examples
