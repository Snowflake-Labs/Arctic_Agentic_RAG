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


from setuptools import find_packages, setup

# Base dependencies (common ones)
base_requirements = [
    "loguru==0.7.2",
    "PyYAML==6.0.2",
    "requests==2.32.3",
    "setuptools==75.1.0",
    "pandas",
    "httpx==0.27.2",
    "snowflake-connector-python",
    "snowflake-ml-python",
    "snowflake-snowpark-python",
    "snowflake.core",
    "openai==1.55.3",
    "sentence_transformers==3.3.0",
]

extras = {
    "ollama": ["ollama"],
    "vllm": ["vllm"],
    "dev": ["black", "pre-commit", "pytest"],
}

# Versioning
version_str = open("version.txt").read().strip()

setup(
    name="arctic_agentic_rag",
    version=version_str,
    include_package_data=True,
    packages=find_packages(),
    install_requires=base_requirements,
    extras_require=extras,
)
