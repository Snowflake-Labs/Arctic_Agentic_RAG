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


from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer
from snowflake.cortex import embed_text_768, embed_text_1024
from snowflake.snowpark import Session

from ..utils import get_snowflake_credentials


class CortexEncoder:
    def __init__(
        self,
        embed_type: Literal["cortex_768", "cortex_1024"] = "cortex_768",
        use_password=True,
    ):
        if embed_type == "cortex_768":
            self.embed = embed_text_768
            self.embedding_dim = 768
            self.model = "snowflake-arctic-embed-m-v1.5"
        elif embed_type == "cortex_1024":
            self.embed = embed_text_1024
            self.embedding_dim = 1024
            self.model = "snowflake-arctic-embed-l-v2.0"
        else:
            raise ValueError("'embed_type' must be either cortex_768 or cortex_1024")

        self.authentication = "password" if use_password else "sso"
        self.session = self._create_session()

    def _create_session(self):
        creds = get_snowflake_credentials(self.authentication)
        return Session.builder.configs(creds).create()

    def encode(self, inputs: list[str], prompt_name: str = None) -> np.array:
        embeddings = np.zeros((len(inputs), self.embedding_dim))
        for i, text in enumerate(inputs):
            embeddings[i] = self.embed(
                model=self.model, text=text, session=self.session
            )
        return embeddings


class HFSentenceTransformerEncoder:
    def __init__(self, model_name_or_path: str, device="cpu") -> None:
        model = SentenceTransformer(model_name_or_path)
        self.model = model.to(device)
        self.device = device
        self.embedding_dim = model.get_sentence_embedding_dimension()

    def encode(self, inputs: list[str], prompt_name: str = None) -> np.array:
        embeddings = np.zeros((len(inputs), self.embedding_dim))
        # for now, batch size of 1
        for i, text in enumerate(inputs):
            embeddings[i] = self.model.encode(text, prompt_name=prompt_name)
        return embeddings
