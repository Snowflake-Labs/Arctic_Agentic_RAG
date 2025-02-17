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


from dataclasses import dataclass, field
from typing import Literal

from arctic_agentic_rag.llm import AzureOAI
from arctic_agentic_rag.llm.cortex_complete import CortexComplete


@dataclass
class RetrievalArguments:
    retrieval_backend: Literal["cortex", "offline"] = "offline"
    source_file_path: str = field(
        default="data/asqa_top20.json",
        metadata={
            "help": "Path to the JSON file containing the offline retrieval results."
        },
    )


@dataclass
class LLMArguments:
    model: str
    backend: Literal["snowflake", "openai", "azure", "vllm", "ollama"] = "snowflake"


@dataclass
class VDArguments:
    encoder_name_or_path: str = field(
        default="cortex_768",
        metadata={
            "help": (
                "Name or path to HF sentence_transformer model checkpoint, or either"
                " 'cortex_768' or 'cortex_1024' to use arctic-embed through cortex"
                " embed."
            )
        },
    )

    topk: int = 5
    clustering: Literal["hdbscan", "kmeans"] = "hdbscan"
    embedding: Literal["q", "qa"] = field(
        default="qa",
        metadata={
            "help": (
                "Inputs to the encoder when obtaining embeddings, 'q' denotes"
                " question-only while 'qa' denotes question and answer concatenated."
            )
        },
    )


@dataclass
class IOArguments:
    num_examples: int = field(
        default=None,
        metadata={
            "help": (
                "Number of test examples to run; first 'num_examples' examples will be"
                " tested. Set a smaller value for debugging purpose."
            )
        },
    )
    input_path: str = "data/asqa_top20.json"
    output_path: str = "./results/"


@dataclass
class SnowflakeCompleteArguments:
    authentication: Literal["password", "sso"] = "sso"


@dataclass
class AzureOpenAIArguments:
    azure_endpoint: str
    api_version: str = "2024-07-01-preview"
    max_retries: int = 1


backend_name_to_backend_class = {"snowflake": CortexComplete, "azure": AzureOAI}

backend_to_backend_specific_argument_class = {
    "snowflake": SnowflakeCompleteArguments,
    "azure": AzureOpenAIArguments,
}
