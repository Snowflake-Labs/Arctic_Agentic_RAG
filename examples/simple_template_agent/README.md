## Template and TemplateAgents

### Template

Template helps to compose structured LLM prompts and parse results with ease.
You can define your own template by inheriting Template class as follows:
```python
from arctic_agentic_rag.template import Template, Field

class SimpleTemplate(Template):
    """You are a helpful assistant providing an answer to the given question."""
    question = Field(desc="The question seeking some information.", mode="input", display_name="Question")
    answer = Field(desc="The correct answer to the question.", mode="output", display_name="Answer")
```
The above code defines a simple template, with single input field (question) and single output field (answer).
Note that there should be always at least one input field and one output field!

Concrete prompts are derived from templates, when the input fields are populated with concrete values.
For this purpose, `Template` has `build_system_prompt` and `build_user_prompt` methods defined.
An example of system and user prompts derived from the above template looks like:
```
/* System prompt */
Your input fields are:
Question: The question seeking some information.

Your output fields are:
Answer: The correct answer to the question.

All interactions will be structured in the following way, with the appropriate values filled in:

[[ ## Question ## ]]
{Question}

[[ ## Answer ## ]]
{Answer}

[[ ## Completed ## ]]

In adhering to this structure, your objective is:
You are a helpful assistant providing an answer to the given question.


/* User prompt */
We have provided the following input fields:

[[ ## Question ## ]]
Average life expectancy for a west highland terrier?

Respond with the corresponding output fields in the following order:
[[ ## Answer ## ]]

Finally, end with the marker for [[ ## Completed ## ]].
```
To obtain the user prompt shown above, the specific question "Average life expectancy for a west highland terrier?" should be provided, as a keyword argument as follows:
```python
SimpleTemplate.build_user_prompt(question="Average life expectancy for a west highland terrier?")
```

If instructions are long, or should be dynamically determined at runtime, you can use the `set_instructions` decorator.
```python
from arctic_agentic_rag.template.template import set_instructions

@set_instructions(some_long_instructions)
class TemplateWithLongInstructions(Template):
    ...
```

Sometimes, fields naturally take sequence-type values.
A representative example in retrieval-augmented generation is the list of retrieved passages.
`ListField` handles formatting each item in such list and aggregating them.
The formatting can be specified with python's `string.Template`.
```python
from arctic_agentic_rag.template import Template, Field, ListField

class SimpleRAGTemplate(Template):
    """You are a helpful assistant providing an answer to the given question, based on the passages.."""
    question = Field(desc="The question seeking some information.", mode="input", display_name="Question")
    passages = ListField(desc="The retrieved passages that might be relevant.", mode="input", display_name="Passages", formatting="[$idx] $text")
    answer = Field(desc="The correct answer to the question.", mode="output", display_name="Answer")
```
In this example, passages can be passed as a list of dictionaries, where each dictionary contains `idx` and `text` as keys.

Finally, `Template` can also take few-shot examples, to automatically format and include them in the system prompt.
It can be provided as a list of dictionaries, `examples`, where each dictionary corresponds to a demonstration and should contain all (both input and output fields) as keys.
```python
from arctic_agentic_rag.template import Template, Field, ListField

class SimpleRAGTemplate(Template):
    """You are a helpful assistant providing an answer to the given question, based on the passages.."""
    question = Field(desc="The question seeking some information.", mode="input", display_name="Question")
    passages = ListField(desc="The retrieved passages that might be relevant.", mode="input", display_name="Passages", formatting="[$idx] $text")
    answer = Field(desc="The correct answer to the question.", mode="output", display_name="Answer")

    examples = [{"question": "example question 1", ...}, ...]
```

### TemplateAgent
Using LLMs with `Template` is simple enough with `TemplateAgent`.
Once you define your own template based on your target task/setting, simply passing that template to create a dedicated `TemplateAgent` and it returns the resulting output fields as a dictionary.
```python
SimpleRAGAgent = TemplateAgent(
    template=SimpleRAGTemplate, backbone=llm, logger=logger, uid="unique ID"
)

result = SimpleRAGAgent.get_result({"question": "your question", "passages": ["your passages", ...]})
if result["_valid"]:
    print(result["answer"])
```
As shown above, it can be combined with any of the diverse backends supported by Arctic Agentic RAG to actually make LLM calls and obtain responses.

### Runtime definition of template/agents
While defining a template as hardcoded python class is the most straightforward approach, `main_dynamic.py` also shows an alternative---how to use `create_template_from_yaml` from `arctic_agentic_rag.template` to import config from external files and dynamically instantiate a template at runtime.

#### How to run the example script
```
./run_dynamic_template.sh
```
It sources the instruction provided in `configs/example_system_prompt.txt` to define a simple `TemplateAgent`, then uses Snowflake Cortex backend to call the LLaMA 3.3 70B model to generate answers.
