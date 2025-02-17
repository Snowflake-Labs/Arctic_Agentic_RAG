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


from argparse import ArgumentParser

import yaml

from arctic_agentic_rag.agent import agent_name_to_agent_class
from arctic_agentic_rag.llm import ModelLoader
from arctic_agentic_rag.logging import Logger
from arctic_agentic_rag.template import create_template_from_yaml
from arctic_agentic_rag.utils import stream_from_file


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--agent_config", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    # parser.add_argument('--output_file', type=str, required=True)
    # parser.add_argument('--log_file', type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--template_config", type=str, default=None)
    parser.add_argument("--root_dir", type=str, required=True)

    args = parser.parse_args()

    with open(args.agent_config) as f:
        config = yaml.safe_load(f)

    return args, config


def main(args, config):
    logger = Logger(base_path=args.result_path)

    # TODO: change to recursive design
    model_class = ModelLoader.get_model_class(config["backbone"]["type"])
    llm = model_class(
        uid=f"{config['name']}.{config['backbone']['name']}",
        logger=logger,
        max_retries=1,
        **{k: v for k, v in config["backbone"].items() if k not in ("name", "type")},
    )

    agent_class = agent_name_to_agent_class[config["type"]]

    kwargs = {
        "backbone": llm,
        "uid": config["name"],
        "logger": logger,
    }
    if args.template_config is not None:
        with open(args.template_config) as f:
            template_config = f.read()
        dynamic_template = create_template_from_yaml(template_config, args.root_dir)
        kwargs.update({"template": dynamic_template})
    agent = agent_class(**kwargs)

    # Custom data preprocessing
    def example_to_inputs(example):
        return {
            "question": example["question"],
            "contexts": [
                {"pid": i, "text": x["excerpt"]}
                for i, x in enumerate(example["search_results"], start=1)
            ],
            "ground_truth": example["answer"],
        }

    # testing with the first 10 examples
    for example, _ in zip(stream_from_file(args.input_file), range(10)):
        result = agent.get_result(example_to_inputs(example))
        logger.log_final_results({"result": result})


if __name__ == "__main__":
    args, config = parse_args()
    main(args, config)
