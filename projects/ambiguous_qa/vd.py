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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Union, cast

from arguments import (
    IOArguments,
    LLMArguments,
    RetrievalArguments,
    VDArguments,
    backend_name_to_backend_class,
    backend_to_backend_specific_argument_class,
)
from clustering import Clusterer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import HfArgumentParser
from utils import get_instruction_and_examples, load_from_file

from arctic_agentic_rag.agent import TemplateAgent
from arctic_agentic_rag.llm import BaseLLM
from arctic_agentic_rag.logging import Logger
from arctic_agentic_rag.ranking.encoder import CortexEncoder
from arctic_agentic_rag.retrieval import CortexRetrievalAgent
from arctic_agentic_rag.template import Field, Template
from arctic_agentic_rag.template.template import set_instructions
from arctic_agentic_rag.utils import (
    get_snowflake_credentials,
    get_snowflake_search_params,
)


class OfflineRetriever:
    """Returns precomputed retrieval result."""

    def __init__(self, fname: str, result_key: str = "retrieved_passages"):
        self.fname = fname
        self.result_key = result_key
        self.data = load_from_file(self.fname)
        self.query_to_offline_retrieval_result = self.index_with_key(self.data, "query")
        self.query_id_to_offline_retrieval_result = self.index_with_key(
            self.data, "query_id"
        )

    def index_with_key(self, retrieval_result, key: str = "query") -> dict:
        result_key = self.result_key
        if isinstance(retrieval_result, list):
            return {x[key]: x[result_key] for x in retrieval_result}
        elif isinstance(retrieval_result, dict):
            return {x[key]: x[result_key] for x in retrieval_result.values()}
        else:
            raise ValueError

    def retrieve_topk(
        self, query: str = None, query_id: str = None, k: int = 5
    ) -> list:
        if query_id is not None:
            contexts = self.query_id_to_offline_retrieval_result[query_id]
        elif query is not None:
            contexts = self.query_to_offline_retrieval_result[query]
        else:
            raise ValueError("Either 'query' or 'query_id' must be provided")
        return contexts[:k]


class Retrieval:
    def __init__(self, args: RetrievalArguments):
        self.type = args.retrieval_backend

        if args.retrieval_backend == "offline":
            self.retriever = OfflineRetriever(args.source_file_path)
        elif args.retrieval_backend == "cortex":
            connection_parameters = get_snowflake_credentials(authentication="sso")
            service_parameters = get_snowflake_search_params()
            self.retriever = CortexRetrievalAgent(
                connection_parameters, service_parameters
            )
        else:
            raise ValueError("Backend must be either 'offline' or 'cortex'")

    def retrieve_topk(
        self, query: str = None, query_id: str = None, k: int = 5
    ) -> list:
        if self.type == "offline":
            return cast(OfflineRetriever, self.retriever).retrieve_topk(
                query, query_id, k
            )
        elif self.type == "cortex":
            if query is None:
                raise ValueError("For online retrieval, query must be provided.")
            return cast(CortexRetrievalAgent, self.retriever).get_retrieval(query, k)
        else:
            return []


class VD:
    def __init__(
        self,
        logger: Logger,
        llm: BaseLLM,
        topk: int,
        clustering: str,
        embedding: str,
        retrieval: Retrieval,
        encoder: Union[SentenceTransformer, CortexEncoder],
        max_threads: int = 8,
    ):
        self.logger = logger
        self.llm = llm
        self.topk = topk
        self.clustering = clustering
        self.embedding = embedding
        self.retrieval = retrieval
        self.encoder = encoder
        self.max_threads = max_threads
        if self.max_threads == 1:
            # Sequential processing
            self.verified_diversify = self._verified_diversify_sequential
        else:
            # Issue multiple api calls asynchronously
            self.verified_diversify = self._verified_diversify

        instruction, fewshot_examples = get_instruction_and_examples("prompts/vd.json")

        @set_instructions(instruction)
        class ExecutionFeedbackTemplate(Template):
            question = Field(
                desc="The input ambiguous question.",
                mode="input",
                display_name="Ambiguous Question",
            )
            passage = Field(
                desc="The retrieved passage that may be relevant.",
                mode="input",
                display_name="Passage",
            )
            reason = Field(
                desc="Explanation for you decision and reasoning.",
                mode="output",
                display_name="Reason",
            )
            disambiguation = Field(
                desc=(
                    "The disambiguated question, that can be answered by the given"
                    " passage."
                ),
                mode="output",
                display_name="Disambiguated Subquestion",
            )
            answer = Field(
                desc=(
                    "The answer to the disambiguated question, derived based on the"
                    " given passage."
                ),
                mode="output",
                display_name="Answer",
            )
            examples = fewshot_examples

        self.vd = TemplateAgent(
            template=ExecutionFeedbackTemplate, backbone=llm, logger=logger, uid="vd"
        )

    def relax_query(self, query: str) -> str:
        return query

    def _verified_diversify_sequential(
        self, query: str, universe: list
    ) -> tuple[list[str], list, list[str]]:
        interpretations, passages, answers = [], [], []
        for passage in universe:
            passage_text = passage if isinstance(passage, str) else passage["text"]
            result = self.vd.get_result({"question": query, "passage": passage_text})

            if not result["_valid"]:
                logger.log_warning(
                    f"Skipping invalid response for\nquery: {query}\nand passage:"
                    f" {passage}"
                )
                continue

            if result["disambiguation"] != "null" and result["answer"] != "null":
                interpretations.append(result["disambiguation"])
                passages.append(passage_text)
                answers.append(result["answer"])
        return interpretations, passages, answers

    def process_single_passage(self, query: str, passage: Union[str, dict]) -> dict:
        passage_text = passage if isinstance(passage, str) else passage["text"]
        result = self.vd.get_result(
            {"question": query, "passage": passage_text}
        )  # Blocking API call

        if not result["_valid"]:
            self.logger.log_warning(
                f"Skipping invalid response for\nquery: {query}\nand passage: {passage}"
            )
            return None

        if result["disambiguation"] == "null" or result["answer"] == "null":
            return None
        return {
            "interpretation": result["disambiguation"],
            "passage": passage_text,
            "answer": result["answer"],
        }

    def _verified_diversify(
        self, query: str, universe: list
    ) -> tuple[list[str], list, list[str]]:
        """
        Extracts interpretations and answers from the universe of retrieved passages.
        Unlike the sequential version, parallelizes API calls.

        Args:
            query (str): The ambiguous query to be answered.
            universe (list): List of retrieved passages.

        Returns:
            tuple[list[str], list, list[str]]: List of interpretations, passages and answers, respectively.
        """
        interpretations, passages, answers = [], [], []

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_passage = {
                executor.submit(self.process_single_passage, query, passage): passage
                for passage in universe
            }

            for future in as_completed(future_to_passage):
                result = future.result()
                if result is not None:
                    interpretations.append(result["interpretation"])
                    passages.append(result["passage"])
                    answers.append(result["answer"])
        self.logger.log_info("Parallel execution completed.")

        return interpretations, passages, answers

    def consolidate_feedback(
        self, interpretations: list[str], passages: list[str], answers: list[str]
    ) -> list[dict[str, str]]:
        chosen_indices, clusters = Clusterer.cluster(
            interpretations, answers, self.encoder, self.clustering, self.embedding
        )
        self.logger.log_info(
            f"Clustering: out of {len(interpretations)} interpretations, selected"
            f" {', '.join(map(str, chosen_indices))}."
        )
        return [
            {
                "q": interpretations[i],
                "p": passages[i],
                "a": answers[i],
            }
            for i in chosen_indices
        ]

    def run(self, query: str, query_id: str = None) -> list[dict[str, str]]:
        self.logger.log_info(f"Processing the query {query}")
        relaxed_query = self.relax_query(query)
        universe = self.retrieval.retrieve_topk(relaxed_query, k=self.topk)
        interpretations, passages, answers = self.verified_diversify(query, universe)
        results = self.consolidate_feedback(interpretations, passages, answers)
        self.logger.log_info(
            f"Obtained interpretations, supporting passages and answers: {results}"
        )

        return results


if __name__ == "__main__":
    parser = HfArgumentParser(
        (RetrievalArguments, LLMArguments, VDArguments, IOArguments)
    )
    re_args, llm_args, vd_args, io_args, extra_args = (
        parser.parse_args_into_dataclasses(return_remaining_strings=True)
    )

    # Each backend has specific arguments need to be configured,
    # e.g., 'azure_endpoint' for Azure OpenAI backend.
    backend_class = backend_name_to_backend_class[llm_args.backend]
    backend_specific_argument_class = backend_to_backend_specific_argument_class[
        llm_args.backend
    ]
    subparser = HfArgumentParser((backend_specific_argument_class,))
    (backend_specific_args,) = subparser.parse_args_into_dataclasses(extra_args)

    # Shared logger and backbone LLM
    logger = Logger(base_path=io_args.output_path)
    llm = backend_class(
        model=llm_args.model,
        uid="generator",
        logger=logger,
        **asdict(backend_specific_args),
    )

    # Prepare retrieval system
    retrieval = Retrieval(re_args)

    # Prepare encoder (used in clustering)
    if vd_args.encoder_name_or_path in ("cortex_768", "cortex_1024"):
        encoder = CortexEncoder(vd_args.encoder_name_or_path, False)
    else:
        encoder = SentenceTransformer(
            vd_args.encoder_name_or_path, trust_remote_code=True, device="cuda"
        )

    vd = VD(
        logger,
        llm,
        vd_args.topk,
        vd_args.clustering,
        vd_args.embedding,
        retrieval,
        encoder,
    )

    # Load test data
    with open(io_args.input_path) as f:
        data: list[dict] = json.load(f)

    # Run ours
    results = []
    for d in tqdm(data[: io_args.num_examples]):
        result = vd.run(d["query"])
        results.append(
            {
                "query": d["query"],
                "query_id": d.get("query_id", None),
                "interpretations": result,
            }
        )

    with open(os.path.join(io_args.output_path, "output.json"), "w") as f:
        json.dump(results, f, indent=2)
