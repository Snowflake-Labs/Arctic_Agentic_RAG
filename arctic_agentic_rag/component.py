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


from arctic_agentic_rag.logging import Logger


class Loggable:
    """
    Mixin class for logging.
    """

    def __init__(self, logger: Logger, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.logger: Logger = logger


class UniqueIDMixin:
    """
    Mixin class for assigning UIDs.
    """

    def __init__(self, uid: str, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.uid: str = uid


class Component(Loggable, UniqueIDMixin):
    pass
