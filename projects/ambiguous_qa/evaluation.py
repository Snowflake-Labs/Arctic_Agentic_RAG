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


import argparse
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Union

import numpy as np
from utils import get_instruction_and_examples, load_from_file

from arctic_agentic_rag.agent import TemplateAgent
from arctic_agentic_rag.llm import AzureOAI
from arctic_agentic_rag.logging import Logger
from arctic_agentic_rag.template import Field, ListField, Template
from arctic_agentic_rag.template.template import set_instructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--result_filename", type=str, default="output.json")
    parser.add_argument("--test_data_path", type=str, default="data/asqa_top20.json")
    parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "precision",
            "recall",
            "grounded_precision",
            "grounded_recall",
            "intrinsic_diversity",
            "error_analysis",
        ],
    )
    parser.add_argument("--azure_endpoint", type=str, required=True)
    parser.add_argument("--azure_model", type=str, required=True)
    return parser.parse_args()


def get_diversity(data: list, llm: AzureOAI) -> dict:
    r"""
    Evaluates intrinsic diversity metrics.

    Args:
        data (list): Model output to be evaluated.
        llm (AzureOAI): Azure OpenAI backend.

    Returns:
        dict: Intrinsic diversity metrics and raw results.
    """

    instruction, fewshot_examples = get_instruction_and_examples(
        "prompts/eval_dedup.json"
    )

    @set_instructions(instruction)
    class DedupInterpretationsTemplate(Template):
        ambiguous_question = Field(
            desc="The input ambiguous question.",
            mode="input",
            display_name="Ambiguous Question",
        )
        generated_subquestions = ListField(
            desc="The list of generated subquestions.",
            mode="input",
            display_name="Disambiguated Subquestions",
        )
        unique_subquestions = ListField(
            desc="The list of unique subquestions after deduplication.",
            mode="output",
            display_name="Unique Subquestions",
        )
        examples = fewshot_examples

    judge = TemplateAgent(
        template=DedupInterpretationsTemplate,
        backbone=llm,
        logger=llm.logger,
        uid="judge",
    )

    results = []
    len_before: list[int] = []
    len_after: list[int] = []
    for d in data:
        ambiguous_question = d["query"]
        interpretations = d["interpretations"]

        output = judge.get_result(
            {
                "ambiguous_question": ambiguous_question,
                "generated_subquestions": [x["q"] for x in interpretations],
            }
        )

        if not output["_valid"]:
            llm.logger.log_warning(
                f"Skipping invalid response for\ninterpretations: {interpretations}"
            )
            continue

        unique_interpretations = output["unique_subquestions"].strip().split("\n")
        result = {
            "query": d["query"],
            "query_id": d["query_id"],
            "unique_interpretations": unique_interpretations,
            "num_before": len(interpretations),
            "num_after": len(unique_interpretations),
        }

        results.append(result)
        len_before.append(len(interpretations))
        len_after.append(len(unique_interpretations))

    len_before: np.ndarray = np.array(len_before)
    len_after: np.ndarray = np.array(len_after)
    avg_num_generated = len_before.mean()
    sufficient_at_ks = {
        f"Sufficient% @{k}": (len_after >= k).mean() for k in range(1, 6)
    }

    return {
        "summary": {
            "avg_num_generated": avg_num_generated,
            **sufficient_at_ks,
        },
        "results": results,
    }


def get_precision(data: list, test_data: list, llm: AzureOAI) -> dict:
    r"""
    Evaluates ungrounded precision.

    Args:
        data (list): Model output to be evaluated.
        test_data (list): Test data, containing GT interpretations, etc.
        llm (AzureOAI): Azure OpenAI backend.

    Returns:
        dict: Precision score and raw results.
    """

    instruction, fewshot_examples = get_instruction_and_examples(
        "prompts/eval_precision.json"
    )

    @set_instructions(instruction)
    class UngroundedPrecisionTemplate(Template):
        ambiguous_question = Field(desc="The input ambiguous question.", mode="input")
        generated_subquestions = ListField(
            desc="The list of generated subquestions.",
            mode="input",
            display_name="Generated Subquestions",
        )
        gt_subquestions = ListField(
            desc="The list of ground-truth subquestions.",
            mode="input",
            display_name="Ground-truth Subquestions",
        )

        reason = Field(
            desc="Any reasoning needed to determine matches.",
            mode="output",
            display_name="Reason",
        )
        decisions = ListField(
            desc="List of binary decision for each generated subquestion.",
            mode="output",
            display_name="Decisions",
        )
        examples = fewshot_examples

    judge = TemplateAgent(
        template=UngroundedPrecisionTemplate,
        backbone=llm,
        logger=llm.logger,
        uid="judge",
    )

    qid_to_gt_interpretations = {}
    for t in test_data:
        qid_to_gt_interpretations[t["query_id"]] = [
            x["question"] for x in t["qa_pairs"]
        ]

    results = []
    all_decisions = []
    for d in data:
        ambiguous_question = d["query"]
        interpretations = d["interpretations"]

        output = judge.get_result(
            {
                "ambiguous_question": ambiguous_question,
                "generated_subquestions": [x["q"] for x in interpretations],
                "gt_subquestions": qid_to_gt_interpretations[d["query_id"]],
            }
        )

        if not output["_valid"]:
            llm.logger.log_warning(
                f"Skipping invalid response for\ninterpretations: {interpretations}"
            )
            continue

        decisions = [
            int(x.lower() == "yes") for x in output["decisions"].strip().split("\n")
        ]
        result = {
            "query": d["query"],
            "query_id": d["query_id"],
            "reason": output["reason"],
            "decisions": decisions,
        }

        results.append(result)
        all_decisions.append(decisions)

    precision_macro = np.array(
        [np.array(decisions).mean() for decisions in all_decisions]
    ).mean()
    precision_micro = np.array(
        [x for decisions in all_decisions for x in decisions]
    ).mean()

    return {
        "summary": {
            "precision": precision_macro,
            "precision_micro": precision_micro,
        },
        "results": results,
    }


def get_recall(data: list, test_data: list, llm: AzureOAI) -> dict:
    r"""
    Evaluates ungrounded recall.

    Args:
        data (list): Model output to be evaluated.
        test_data (list): Test data, containing GT interpretations, etc.
        llm (AzureOAI): Azure OpenAI backend.

    Returns:
        dict: Recall score and raw results.
    """
    instruction, fewshot_examples = get_instruction_and_examples(
        "prompts/eval_recall.json"
    )

    @set_instructions(instruction)
    class UngroundedRecallTemplate(Template):
        ambiguous_question = Field(desc="The input ambiguous question.", mode="input")
        generated_subquestions = ListField(
            desc="The list of generated subquestions.",
            mode="input",
            display_name="Generated Subquestions",
        )
        gt_subquestions = ListField(
            desc="The list of ground-truth subquestions.",
            mode="input",
            display_name="Ground-truth Subquestions",
        )

        reason = Field(desc="Any reasoning needed to determine matches.", mode="output")
        decisions = ListField(
            desc="List of binary decision for each generated subquestion.",
            mode="output",
            display_name="Decisions",
        )
        examples = fewshot_examples

    judge = TemplateAgent(
        template=UngroundedRecallTemplate,
        backbone=llm,
        logger=llm.logger,
        uid="judge",
    )

    qid_to_gt_interpretations = {}
    for t in test_data:
        qid_to_gt_interpretations[t["query_id"]] = [
            x["question"] for x in t["qa_pairs"]
        ]

    results = []
    all_decisions = []
    for d in data:
        ambiguous_question = d["query"]
        interpretations = d["interpretations"]

        output = judge.get_result(
            {
                "ambiguous_question": ambiguous_question,
                "generated_subquestions": [x["q"] for x in interpretations],
                "gt_subquestions": qid_to_gt_interpretations[d["query_id"]],
            }
        )

        if not output["_valid"]:
            llm.logger.log_warning(
                f"Skipping invalid response for\ninterpretations: {interpretations}"
            )
            continue

        decisions = [
            int(x.lower() == "yes") for x in output["decisions"].strip().split("\n")
        ]
        result = {
            "query": d["query"],
            "query_id": d["query_id"],
            "reason": output["reason"],
            "decisions": decisions,
        }

        results.append(result)
        all_decisions.append(decisions)

    recall_macro = np.array(
        [np.array(decisions).mean() for decisions in all_decisions]
    ).mean()
    recall_micro = np.array(
        [x for decisions in all_decisions for x in decisions]
    ).mean()

    return {
        "summary": {
            "recall": recall_macro,
            "recall_micro": recall_micro,
        },
        "results": results,
    }


def get_grounded_precision(data: list, test_data: list, llm: AzureOAI) -> dict:
    """Evaluates grounded precision, by verifying whether generated interpretation
    is supported by the associated evidence.
    If such evidence is not provided, use top-k retrieved passages instead.

    Args:
        data (list): Model output to be evaluated.
        test_data (list): Test data, containing GT interpretations, etc.
        llm (AzureOAI): Azure OpenAI backend.

    Returns:
        dict: Grounded precision score and raw results.
    """

    instruction, fewshot_examples = get_instruction_and_examples(
        "prompts/eval_verify.json"
    )

    @set_instructions(instruction)
    class VerifyTemplate(Template):
        question = Field(
            desc="The question to be tested.", mode="input", display_name="Question"
        )
        answer = Field(
            desc="The generated answer given the question and passage.",
            mode="input",
            display_name="Answer",
        )
        passage = Field(
            desc="A relevant passage that might support the question-answer or not.",
            mode="input",
            display_name="Passage",
        )

        reason = Field(
            desc="Any reasoning needed to determine matches.",
            mode="output",
            display_name="Reason",
        )
        decision = Field(
            desc=(
                "Binary decision for whether the passage supports question-answer or"
                " not."
            ),
            mode="output",
            display_name="Decision",
        )
        examples = fewshot_examples

    judge = TemplateAgent(
        template=VerifyTemplate,
        backbone=llm,
        logger=llm.logger,
        uid="judge",
    )

    if any("p" in x for d in data[:5] for x in d["interpretations"]):
        # The system has provided supporting passage for each interpretation
        results = []
        all_decisions = []
        for d in data:
            interpretations = d["interpretations"]

            reasons = []
            decisions = []
            for x in interpretations:
                interpretation = x["q"]
                passage = x["p"]
                answer = x["a"]

                output = judge.get_result(
                    {
                        "question": interpretation,
                        "answer": answer,
                        "passage": passage,
                    }
                )

                if not output["_valid"]:
                    llm.logger.log_warning(
                        "Skipping invalid response for\nquestion:"
                        f" {interpretation}\nanswer: {answer}\npassage: {passage}"
                    )
                    continue

                reasons.append(output["reason"])
                decision = int(output["decision"].strip().lower() == "yes")
                decisions.append(decision)

            result = {
                "query": d["query"],
                "query_id": d["query_id"],
                "reasons": reasons,
                "decisions": decisions,
            }

            results.append(result)
            all_decisions.append(decisions)
    else:
        # The system has NOT provided supporting passage for each interpretation
        # Verify with top-k passages instead
        qid_to_topk = {}
        for t in test_data:
            qid_to_topk[t["query_id"]] = [
                x["text"] if isinstance(x, dict) else x for x in t["retrieved_passages"]
            ]

        results = []
        all_decisions = []

        def process_single_passage(idx, interpretation, passage, answer):
            output = judge.get_result(
                {
                    "question": interpretation,
                    "answer": answer,
                    "passage": passage,
                }
            )

            if not output["_valid"]:
                llm.logger.log_warning(
                    "Skipping invalid response for\nquestion:"
                    f" {interpretation}\nanswer: {answer}\npassage: {passage}"
                )
                return idx, None

            return idx, {
                "reason": output["reason"],
                "decision": int(output["decision"].strip().lower() == "yes"),
            }

        for d in data:
            interpretations = d["interpretations"]
            topk_passages = qid_to_topk[d["query_id"]]

            reasons_list = []
            decisions_list = []

            for x in interpretations:
                interpretation = x["q"]
                answer = x.get("a", "")
                reasons = [None] * len(topk_passages)
                decisions = [None] * len(topk_passages)

                with ThreadPoolExecutor(max_workers=20) as executor:
                    future_to_idx = {
                        executor.submit(
                            process_single_passage, idx, interpretation, passage, answer
                        ): idx
                        for idx, passage in enumerate(topk_passages)
                    }

                    for future in as_completed(future_to_idx):
                        idx, result = future.result()
                        if result is not None:
                            reasons[idx] = result["reason"]
                            decisions[idx] = result["decision"]

                reasons_list.append(reasons)
                decisions_list.append(decisions)

            result = {
                "query": d["query"],
                "query_id": d["query_id"],
                "reasons": reasons_list,
                "decisions": decisions_list,
            }

            results.append(result)
            all_decisions.append([int(any(decisions)) for decisions in decisions_list])

    precision_macro = np.array(
        [np.array(decisions).mean() for decisions in all_decisions]
    ).mean()
    precision_micro = np.array(
        [x for decisions in all_decisions for x in decisions]
    ).mean()

    return {
        "summary": {
            "grounded_precision": precision_macro,
            "grounded_precision_micro": precision_micro,
        },
        "results": results,
    }


def get_grounded_recall(data: list, prerequisites: dict[str, dict]) -> dict:
    r"""
    Evaluates grounded recall.

    Args:
        data (list): Model output to be evaluated.
        prerequisites (dict[str, dict): Evaluation results with prerequisite metrics.

    Returns:
        dict: Grounded recall score and raw results.

    Raises:
        ValueError: If one or more prerequisite metrics are missing.
    """

    # To obtain grounded recall, eval results for recall, precision and
    # grounded precision are required.
    # If they are available at expected paths, load them.
    # If not, call respective functions to obtain them.
    if (
        "recall" not in prerequisites
        or "precision" not in prerequisites
        or "grounded_precision" not in prerequisites
    ):
        raise ValueError("Missing prerequisite(s) for grounded recall")

    qid_to_recall = prerequisites["recall"]
    qid_to_precision = prerequisites["precision"]
    qid_to_grounded_precision = prerequisites["grounded_precision"]

    results = []
    all_decisions = []
    for d in data:
        query_id = d["query_id"]
        interpretations = d["interpretations"]

        precisions = qid_to_precision[query_id]["decisions"]
        recalls = qid_to_recall[query_id]["decisions"]
        grounded_precisions = qid_to_grounded_precision[query_id]["decisions"]
        added = 0

        for interpretation, p, gp in zip(
            interpretations, precisions, grounded_precisions
        ):
            if p == 0 and gp == 1:
                # This interpretation has not been matched with any humna interpretation (precision == 0)
                # yet it is grounded (grounded precision == 1)
                added += 1

        result = {
            "query": d["query"],
            "query_id": d["query_id"],
            "additionally_matched": added,
        }

        results.append(result)
        all_decisions.append(recalls + ([1] * added))

    recall_macro = np.array(
        [np.array(decisions).mean() for decisions in all_decisions]
    ).mean()
    recall_micro = np.array(
        [x for decisions in all_decisions for x in decisions]
    ).mean()

    return {
        "summary": {
            "grounded_recall": recall_macro,
            "grounded_recall_micro": recall_micro,
        },
        "results": results,
    }


def get_error_types(data: list, llm: AzureOAI):
    """
    Evaluates correctness of interpretation/answer.

    Args:
        data (list): Model output to be evaluated.
        llm (AzureOAI): Azure OpenAI backend.

    Returns:
        dict: Accuracy score and error categorization results.
    """

    instruction, fewshot_examples = get_instruction_and_examples(
        "prompts/eval_correctness.json"
    )

    @set_instructions(instruction)
    class CorrectnessTemplate(Template):
        ambiguous_question = Field(
            desc="The ambiguous question.",
            mode="input",
            display_name="Ambiguous Question",
        )
        question = Field(
            desc="The generated disambiguation question to be tested.",
            mode="input",
            display_name="Question",
        )
        answer = Field(
            desc="The generated answer given the question and passage.",
            mode="input",
            display_name="Answer",
        )
        passage = Field(
            desc="A relevant passage that might support the question-answer or not.",
            mode="input",
            display_name="Passage",
        )

        reason = Field(
            desc="Any reasoning needed to determine matches.",
            mode="output",
            display_name="Reason",
        )
        decision = Field(
            desc=(
                "Decision for the given question-passage-answer triple: Irrelevant,"
                " Unanswerable, Wrong answer or Correct."
            ),
            mode="output",
            display_name="Decision",
        )
        examples = fewshot_examples

    judge = TemplateAgent(
        template=CorrectnessTemplate,
        backbone=llm,
        logger=llm.logger,
        uid="judge",
    )

    results = []
    all_decisions = []
    for d in data:
        ambiguous_question = d["query"]
        interpretations = d["interpretations"]

        reasons = []
        decisions = []
        for x in interpretations:
            interpretation = x["q"]
            passage = x["p"]
            answer = x["a"]

            output = judge.get_result(
                {
                    "ambiguous_question": ambiguous_question,
                    "question": interpretation,
                    "answer": answer,
                    "passage": passage,
                }
            )

            if not output["_valid"]:
                llm.logger.log_warning(
                    "Skipping invalid response for\nquestion:"
                    f" {interpretation}\nanswer: {answer}\npassage: {passage}"
                )
                reasons.append("")
                decisions.append(-1)
            else:
                reasons.append(output["reason"])
                decision_str = output["decision"].strip().lower()
                if decision_str == "correct":
                    decision = 1
                elif decision_str == "wrong answer":
                    decision = 0
                elif decision_str == "unanswerable":
                    decision = -2
                elif decision_str == "irrelevant":
                    decision = -3
                else:
                    decision = -1
                decisions.append(decision)

        result = {
            "query": d["query"],
            "query_id": d["query_id"],
            "reasons": reasons,
            "decisions": decisions,
        }

        results.append(result)
        all_decisions.append(decisions)

    strict_accuracy_macro = np.array(
        [np.mean(np.array(decisions) == 1) for decisions in all_decisions]
    ).mean()
    flattened_decisions = np.array(
        [x for decisions in all_decisions for x in decisions]
    )
    strict_accuracy_micro = np.mean(flattened_decisions == 1)

    return {
        "summary": {
            "strict_accuracy": strict_accuracy_macro,
            "strict_accuracy_micro": strict_accuracy_micro,
            "# Correct": np.sum(flattened_decisions == 1).item(),
            "# Wrong answer": np.sum(flattened_decisions == 0).item(),
            "# Unanswerable": np.sum(flattened_decisions == -2).item(),
            "# Irrelevant": np.sum(flattened_decisions == -3).item(),
            "# Parse error": np.sum(flattened_decisions == -1).item(),
        },
        "results": results,
    }


def get_required_args(func: Callable) -> set[str]:
    r"""
    Extracts required arguments from a function func.

    Args:
        func (Callable): Function to extract required arguments from.

    Returns:
        set[str]: Set of name of required arguments.
    """
    signature = inspect.signature(func)
    return {
        param.name
        for param in signature.parameters.values()
        if param.default == inspect.Parameter.empty
    }


def evaluate_metric(metric: str, result_path: Union[str, os.PathLike], **kwargs):
    if metric == "intrinsic_diversity":
        func = get_diversity
    elif metric == "precision":
        func = get_precision
    elif metric == "recall":
        func = get_recall
    elif metric == "grounded_precision":
        func = get_grounded_precision
    elif metric == "grounded_recall":
        func = get_grounded_recall

        results = {}
        for prerequisite in ("recall", "precision", "grounded_precision"):
            try:
                fpath = os.path.join(
                    args.result_path, f"eval_result_{prerequisite}.json"
                )
                results[prerequisite] = {
                    x["query_id"]: x for x in load_from_file(fpath)["results"]
                }
            except FileNotFoundError:
                # func = globals()[f"get_{metric}"]
                # eval_result = func(data, test_data, llm)
                results[prerequisite] = {
                    x["query_id"]: x
                    for x in evaluate_metric(prerequisite, result_path, **kwargs)[
                        "results"
                    ]
                }

        kwargs.update({"prerequisites": results})

    elif metric == "error_analysis":
        func = get_error_types
    else:
        raise ValueError(f"Unknown metric {metric}")

    # Check if required arguments are all present
    required_args = get_required_args(func)
    missing_args = required_args - kwargs.keys()
    if missing_args:
        raise ValueError(f"Missing required argument(s): {', '.join(missing_args)}")

    eval_result = func(**{k: kwargs[k] for k in required_args})

    for k, v in eval_result["summary"].items():
        print(k, ":", v)
    with open(os.path.join(result_path, f"eval_result_{metric}.json"), "w") as f:
        json.dump(eval_result, f, indent=2)

    return eval_result


def main(args: argparse.Namespace) -> None:
    data = load_from_file(os.path.join(args.result_path, args.result_filename))
    test_data = load_from_file(args.test_data_path)

    logger = Logger(base_path=args.result_path, log_file=f"eval_{args.metric}.log")
    llm = AzureOAI(
        model=args.azure_model,
        uid="llm_judge",
        logger=logger,
        api_version="2024-07-01-preview",
        azure_endpoint=args.azure_endpoint,
        max_retries=1,
    )

    evaluate_metric(
        args.metric, args.result_path, data=data, test_data=test_data, llm=llm
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
