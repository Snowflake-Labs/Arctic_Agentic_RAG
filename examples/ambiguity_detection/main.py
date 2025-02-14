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
from argparse import ArgumentParser

from typing import Iterator

import yaml

from arctic_agentic_rag.agent import agent_name_to_agent_class
from arctic_agentic_rag.llm import ModelLoader
from arctic_agentic_rag.logging import Logger


def parse_args():
    parser = ArgumentParser(
        description="Run ambiguity detection with specified configuration."
    )
    parser.add_argument(
        "--agent_config", type=str, required=True, help="Path to agent config YAML"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input data file"
    )
    parser.add_argument(
        "--result_path", type=str, required=True, help="Path to store results"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of examples to process"
    )

    args = parser.parse_args()

    with open(args.agent_config) as f:
        config = yaml.safe_load(f)

    return args, config


def load_examples(file_path: str) -> Iterator[dict]:
    """
    Load examples from various dataset formats and stream them.

    Args:
        file_path (str): Path to input data file.

    Yields:
        dict: Each example.
    """
    with open(file_path) as f:
        if "msmarco" in file_path.lower():
            data = json.load(f)
            for item in data:
                yield {
                    "query": item["query"],
                    "passages": [p["passage_text"] for p in item["passages"]],
                }
        elif "asqa" in file_path.lower():
            data = json.load(f)
            for item in data:
                yield {
                    "query": item["ambiguous_query"],
                    "passages": [
                        p["text"]
                        for p in sorted(
                            item["retrieved_passages"],
                            key=lambda x: x["qwen_score"],
                            reverse=True,
                        )[:5]
                    ],
                }
        else:  # SQUAD format
            data = json.load(f)
            for article in data["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        yield {"query": qa["question"], "passages": [context]}


def main():
    args, config = parse_args()

    # Setup logging
    logger = Logger(base_path=args.result_path)

    # Initialize LLM
    model_class = ModelLoader.get_model_class(config["backbone"]["type"])
    llm = model_class(
        uid=f"{config['name']}.{config['backbone']['name']}",
        logger=logger,
        max_retries=1,
        **{k: v for k, v in config["backbone"].items() if k not in ("name", "type")},
    )

    # Initialize agent
    agent_class = agent_name_to_agent_class[config["type"]]
    agent = agent_class(
        backbone=llm,
        uid=config["name"],
        logger=logger,
        prompt_version=config.get("prompt_version", "v4"),
    )

    # Process examples
    for i, example in enumerate(load_examples(args.input_file)):
        if i >= args.limit:
            break

        try:
            result = agent.get_result(example)
            logger.log_final_results(
                {"example_id": i, "query": example["query"], "result": result}
            )
        except Exception as e:
            logger.log_error(f"Error processing example {i}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
