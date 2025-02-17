# Arctic Agentic RAG Framework: Simplifying Agentic Retrieval-Augmented Generation

## Overview
The **Arctic Agentic RAG Framework** is a sophisticated retrieval-augmented generation (RAG) system tailored for agentic workflows. It supports various strategies for retrieval, reasoning, and response generation using large language models (LLMs). The framework's modular design allows for easy customization to fit different agentic setups. Arctic Agentic RAG Framework is built upon [Arctic Agentic RAG series](PLACEHOLD) and will continue release features associted with the eposide rolls out.

## Features
- **Agent-Based Modular Design**: Supports template-based agents for various usage.
- **Multiple LLM-backend Integrations**: Compatible with Snowflake, OpenAI, Azure OpenAI, Ollama, vLLM, and custom LLM providers.
- **Configurable Pipelines**: Utilizes YAML-based configuration for straightforward experiment management.
- **Prebuilt Examples**: Includes ready-to-run examples for QA tasks using different agent types.

## Installation
To install the framework in editable mode:
```sh
pip install -e .
```
This setup allows modifications to be reflected directly in the runtime environment.

## Running Examples

### Simple RAG
Please refer to [Demo](examples/simple_RAG/rag_demo.ipynb) to utilize Cortex Search and Cortex Complete functionality to build your first RAG workflow.

### Ambigious QA
Please refer to [Ambigious QA](projects/ambiguous_qa) and the associted README to reproduce the results of [paper](PLACEHOLDER)

## Contributing
Please refer to [contribution](CONTRIBUTION.md) if you want to contribute to this project.
