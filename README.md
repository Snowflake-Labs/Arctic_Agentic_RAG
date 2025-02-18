# Arctic Agentic RAG Framework: Smarter, Faster, Easier, and More Reliable AI

## Overview
Enterprise AI demands more than static retrieval---it needs intelligence that can understand, reason, and adapt in real time.
Traditional Retrieval-Augmented Generation (RAG) systems struggle with ambiguous queries, multimodal data, and complex reasoning, making it difficult for businesses to extract accurate and reliable information.
Arctic Agentic RAG is designed to go beyond these limitations.
It's an enterprise-ready framework built for agentic workflows, enabling AI-driven retrieval that is smarter, faster, easier to use, and more reliable than ever before.
By integrating reasoning, adaptive retrieval, and modular agent-based execution, Arctic Agentic RAG transforms how businesses interact with AI.

📢 **Learn more about how Agentic RAG redefines enterprise AI in our [overview blog](https://www.snowflake.com/en/engineering-blog/arctic-agentic-rag-enterprise-ai/).**

## An Iterative, Research-Driven, Open-Source Approach
AI is evolving rapidly, and so is our research.
Arctic Agentic RAG is an open-source project designed to advance agentic RAG workflows through a collaborative, iterative approach.
Rather than offering a one-size-fits-all solution, we take an incremental path, tackling one enterprise AI challenge at a time.
Our Innovation Episodes share ongoing progress, delving into specific challenges, problem formulations, new insights, and their corresponding open-source implementations.
By making our research open, we invite contributions, real-world experimentation, and feedback from the broader AI and developer community.

### Episode 1: Tackling Ambiguous Queries with VerDICT
One of the biggest challenges in AI-driven retrieval is handling ambiguous queries---where multiple interpretations exist, and incorrect retrieval can lead to irrelevant or misleading responses.
Arctic Agentic RAG tackles this head-on in Episode 1 of our innovation series.

🚀 Introducing VerDICT (Verified DIversity with ConsolidaTion)---a breakthrough approach that:

✅ Enhances response accuracy by systematically verifying sources.  
✅ Reduces computational cost by minimizing unnecessary retrievals.  
✅ Improves user experience with faster, more precise, and contextually grounded answers.

📢 Dive deeper into Episode 1 and see VerDICT in action in our [blog](https://www.snowflake.com/en/engineering-blog/arctic-agentic-rag-query-clarification/).  
📄 For a detailed technical analysis, check out our research paper: "[Agentic Verification for Ambiguous Query Disambiguation](https://arxiv.org/abs/2502.10352)".  
Stay tuned for more innovations and episodes!

## Key Software Features
Our framework's modular design allows for easy customization to fit different agentic setups.

- **Agent-Based Modular Design**: Supports template-based agents for various usage.
- **Multiple LLM-Backend Integrations**: Compatible with Snowflake, OpenAI, Azure OpenAI, Ollama, vLLM, and custom LLM providers.
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
Please refer to [Ambigious QA](projects/ambiguous_qa) and the associted README.

## Contributing
Please refer to [contribution](CONTRIBUTION.md) if you want to contribute to this project.

## Join Our Community
Connect with researchers and practitioners shaping the future of AI.
Join our [AI Research Community](https://snowflake.discourse.group/c/ai-research-and-development-community) now!
