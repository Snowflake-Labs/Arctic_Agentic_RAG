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


import importlib

from .azure_oai import AzureOAI
from .base_llm_calling import BaseLLM
from .cortex_complete import CortexComplete
from .oai import OAI

_optional_backends = {
    "Ollama": ("ollama", "Ollama"),
    "VLLM": ("vllm_provider", "VLLMProvider"),
    "VLLMOaiServer": ("vllm_provider", "VLLMOaiServer"),
}


class ModelLoader:
    _model_classes = {
        "AzureOAI": AzureOAI,
        "BaseLLM": BaseLLM,
        "CortexComplete": CortexComplete,
        "OAI": OAI,
    }

    @classmethod
    def get_model_class(cls, model_name):
        if model_name in cls._model_classes:
            return cls._model_classes[model_name]

        if model_name in _optional_backends:
            module_name, class_name = _optional_backends[model_name]
            try:
                module = importlib.import_module(f".{module_name}", package=__name__)
                model_class = getattr(module, class_name)
                cls._model_classes[model_name] = model_class
                return model_class
            except ModuleNotFoundError as e:
                raise ImportError(
                    f"Required module for '{model_name}' is missing. "
                    f"Please install the necessary dependencies.\n{str(e)}"
                )

        raise ValueError(f"Unknown model name: {model_name}")


__all__ = ["AzureOAI", "BaseLLM", "CortexComplete", "ModelLoader", "OAI"]
