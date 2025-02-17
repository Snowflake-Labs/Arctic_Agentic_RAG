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


from typing import Literal, Union

from sentence_transformers import SentenceTransformer

from arctic_agentic_rag.ranking.clustering import Clustering
from arctic_agentic_rag.ranking.encoder import CortexEncoder


class Clusterer:
    @staticmethod
    def cluster(
        interpretations: list[str],
        answers: list[str],
        encoder: Union[SentenceTransformer, CortexEncoder],
        clustering: Literal["hdbscan", "kmeans"],
        embedding: Literal["q", "qa"],
    ) -> tuple[list[int], dict[int, list[int]]]:
        inputs = []
        for q, a in zip(interpretations, answers):
            if embedding == "qa":
                inputs.append(q + " " + a)
            elif embedding == "q":
                inputs.append(q)
        embeddings = encoder.encode(inputs, prompt_name="query")
        if clustering == "hdbscan":
            return Clustering.choose_subset_hdb(embeddings)
        elif clustering == "kmeans":
            return Clustering.choose_subset_kmeans(embeddings)
