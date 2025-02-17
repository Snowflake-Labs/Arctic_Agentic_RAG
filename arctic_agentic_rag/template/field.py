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


from string import Template as StringTemplate
from typing import Any, Mapping, Optional, Union, cast, overload


class Field:
    def __init__(self, desc: str, mode: str, display_name: Optional[str] = None):
        if mode not in ["input", "output"]:
            raise ValueError("Parameter 'mode' must be either 'input' or 'output'.")
        self.desc = desc
        self.mode = mode
        self.display_name = display_name
        self.name: Optional[str] = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        if self.display_name is None:
            self.display_name = name.title()
            # May define special markers, to be replaced with spaces

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return self
        return instance.__dict__.get(self.name, None)

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self.name] = value

    def format(self, value: str) -> str:
        return value


class ListField(Field):
    """
    Represents fields that take sequential lists as value.
    """

    def __init__(
        self,
        desc: str,
        mode: str,
        display_name: Optional[str] = None,
        formatting: Optional[str] = None,
    ):
        super().__init__(desc, mode, display_name)
        self.formatting: Optional[StringTemplate] = (
            StringTemplate(formatting) if formatting is not None else None
        )

    # We need the following to pass type checks :(
    @overload
    def format(self, value: str) -> str: ...

    @overload
    def format(self, value: Union[list[Mapping[str, Any]], list[str]]) -> str: ...

    def format(self, value: Union[list[Mapping[str, Any]], list[str], str]) -> str:
        if self.formatting is not None:
            formatted_items = [
                self.formatting.safe_substitute(cast(Mapping[str, Any], item))
                for item in value
            ]
            return "\n".join(formatted_items)
        else:
            if isinstance(value, list):
                return "\n".join(cast(list[str], value))
            else:
                return cast(str, value)
