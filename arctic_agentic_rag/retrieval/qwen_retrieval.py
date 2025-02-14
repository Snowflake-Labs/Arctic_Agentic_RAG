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
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .base_retrieval import RetrievalAgent


class QwenRetrievalAgent(RetrievalAgent):
    def initialize(
        self,
        model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        device: str = "cuda",
        max_seq_length: int = 8192,
        batch_size: int = 1,
    ):
        """
        Initializes the QwenRetrievalAgent with the SentenceTransformer model.

        Args:
            model_name (str): The Hugging Face model name.
            device (str): Device to load the model on ('cuda' or 'cpu').
            max_seq_length (int): Maximum sequence length for the model.
            batch_size (int): Batch size for passage encoding.
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length
        self.model = self.model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.passages = None  # Will be loaded later

    def load_passages(self, filepath: str):
        """Loads passages from a JSON file."""
        with open(filepath) as f:
            self.passages = json.load(f)

    def get_query_embedding(self, query: str) -> list[float]:
        """Generates an embedding for a given query using the Qwen model."""
        query_embedding = self.model.encode([query], prompt_name="query")
        return query_embedding.tolist()[0]

    def get_retrieval(self, k: int = 10) -> dict[str, list[dict]]:
        """
        Retrieves the top-k passages for all queries.

        Args:
            k (int): The number of top-ranked passages to return per query.

        Returns:
            dict: A dictionary where keys are queries and values are lists of top-k ranked passages.
                  Each passage is represented as a dictionary with its text and Qwen similarity score.
        """
        if self.passages is None:
            raise ValueError("Passages not loaded. Call `load_passages` first.")

        results = {}

        with torch.inference_mode():
            for idx, (query, passages) in tqdm(
                enumerate(self.passages.items()), desc="Processing Queries"
            ):
                # Encode query embedding
                query_embedding = np.array(self.get_query_embedding(query))

                scores_agg = []
                len_psg = len(passages)

                for s in tqdm(
                    range(0, len_psg, self.batch_size),
                    desc=f"Encoding Passages for Query {idx}",
                ):
                    document_embeddings = self.model.encode(
                        passages[s : s + self.batch_size]
                    )
                    scores = (query_embedding @ document_embeddings.T) * 100
                    scores_agg.append(scores[0].tolist())

                # Flatten scores
                scores = np.concatenate(scores_agg)

                # Rank passages by score
                sorted_passages = [
                    {"text": passages[i], "qwen_score": scores[i]}
                    for i in np.argsort(scores)[::-1]
                ]

                # Store top-k results for the query
                results[query] = sorted_passages[:k]

        return results
