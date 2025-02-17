## Agentic Verification for Ambiguous Question Disambiguation

### How to run
```
PYTHONPATH=your_path_to_arctic_agentic_rag ./run_{backend}.sh
```
Currently, `azure` and `cortex` backends are available.

### How to run eval
```
PYTHONPATH=your_path_to_arctic_agentic_rag ./eval.sh {result_path} {metric}
```

### Note
* In order to use the azure backend, `AZURE_GPT4O`, `AZURE_GPT4O_MINI`, `AZURE_ENDPOINT` env variables should be defined. Similarly, for the cortex backend, Snowflake account credentials such as `SNOWFLAKE_ACCOUNT` should be available.
* The example script uses cortex embed for embedding generated interpretations during clustering.
It can be optionally replaced with locally running an encoder model, in which case access to gpu is recommended.
