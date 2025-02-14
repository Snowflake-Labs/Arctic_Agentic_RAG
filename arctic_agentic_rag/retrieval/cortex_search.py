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
from typing import Union

from .base_retrieval import RetrievalAgent

from snowflake.core import Root
from snowflake.snowpark import Session


class CortexRetrievalAgent(RetrievalAgent):
    def __init__(
        self, connection_config: Union[str, dict], service_config: Union[str, dict]
    ) -> None:
        """
        Initializes the CortexRetrievalAgent by reading connection and service parameters.

        Args:
            connection_config (Union[str, dict]): Path to the JSON file or dictionary containing connection parameters.
            service_config (Union[str, dict]): Path to the JSON file or dictionary containing service parameters.
        """
        super().__init__(
            connection_config=connection_config, service_config=service_config
        )

    def initialize(self, **kwargs):
        connection_config = kwargs.get("connection_config")
        service_config = kwargs.get("service_config")

        # Load connection parameters
        if isinstance(connection_config, str):
            with open(connection_config) as file:
                connection_parameters = json.load(file)
        else:
            connection_parameters = connection_config

        self.session: Session = Session.builder.configs(connection_parameters).create()
        self.root: Root = Root(self.session)

        # Load service parameters
        if isinstance(service_config, str):
            with open(service_config) as file:
                service_parameters = json.load(file)
        else:
            service_parameters = service_config

        database: str = service_parameters["database"]
        schema: str = service_parameters["schema"]
        service_name: str = service_parameters["service_name"]

        self.service = (
            self.root.databases[database]
            .schemas[schema]
            .cortex_search_services[service_name]
        )

    def get_query_embedding(self, query: str) -> list[float]:
        """
        Generates a query embedding. This is a placeholder to comply with the abstract class.
        """
        return []  # Placeholder return value

    def get_retrieval(self, query: str, k: int) -> list[str]:
        """
        Retrieves the top-k passages based on the query string.
        """
        if not self.service:
            raise ValueError(
                "Service not set. Use `set_service` method to initialize the service."
            )

        resp = self.service.search(query=query, columns=["text"], limit=k)

        response = json.loads(resp.to_json())
        retrieval_result: list[str] = [result["text"] for result in response["results"]]

        return retrieval_result
