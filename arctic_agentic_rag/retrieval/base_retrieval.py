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


from abc import ABC, abstractmethod


class RetrievalAgent(ABC):
    def __init__(self, **kwargs) -> None:
        """
        Initializes the RetrievalAgent.

        Args:
            kwargs: Additional parameters for subclasses (e.g., connection details, model paths).
        """
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self, **kwargs):
        """Subclasses should implement this to initialize resources like models or databases."""
        pass

    @abstractmethod
    def get_query_embedding(self, query: str) -> list[float]:
        """Generates an embedding for a given query."""
        pass

    @abstractmethod
    def get_retrieval(self, query: str, k: int) -> list[str]:
        """
        Retrieves the top-k passages for a query.
        """
        pass
