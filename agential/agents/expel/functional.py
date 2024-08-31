"""Functional module for ExpeL."""

import random
import re

from itertools import chain
from typing import Any, Dict, List, Tuple

from agential.agents.expel.output import ExpeLGenerateOutput
from agential.agents.expel.prompts import (
    CRITIQUE_SUMMARY_SUFFIX_FULL,
    CRITIQUE_SUMMARY_SUFFIX_NOT_FULL,
    EXISTING_INSIGHTS_AI_NAME,
    HUMAN_CRITIQUE_EXISTING_INSIGHTS_ALL_SUCCESS_TEMPLATE,
    HUMAN_CRITIQUE_EXISTING_INSIGHTS_TEMPLATE,
    NON_EXISTENT_INSIGHTS_AT_NAME,
    SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_INSIGHTS_INSTRUCTION,
    SYSTEM_CRITIQUE_EXISTING_INSIGHTS_INSTRUCTION,
    SYSTEM_TEMPLATE,
)
from agential.agents.reflexion.agent import ReflexionReAct
from agential.agents.reflexion.output import ReflexionReActOutput
from agential.llm.llm import BaseLLM, Response

# ============================================== Experience Gathering ==============================================


def gather_experience(
    reflexion_react_agent: ReflexionReAct,
    questions: List[str],
    keys: List[str],
    examples: str,
    prompt: str,
    reflect_examples: str,
    reflect_prompt: str,
    reflect_strategy: str = "reflexion",
    additional_keys: List[Dict[str, str]] = [],
    reflect_additional_keys: List[Dict[str, str]] = [],
    patience: int = 3,
) -> List[Dict[str, Any]]:
    """Collects and aggregates experiences from a ReflexionReAct by generating trajectories and reflections for a set of questions and keys.

    The function iterates over each question-key pair, generates a trajectory using the specified strategy, and records the reflections generated by the agent. Each trajectory and its corresponding reflections are appended to their respective lists within the 'experiences' dictionary.

    Parameters:
        reflexion_react_agent (ReflexionReAct): The agent from which experiences are generated.
        questions (List[str]): A list of questions to be processed by the agent.
        keys (List[str]): A list of keys that are paired with the questions to guide the agent's generation.
        examples (str, optional): Fewshot examples.
        prompt (str, optional): Prompt template string.
        reflect_examples (str, optional): Reflection fewshot examples.
        reflect_prompt (str, optional): Reflect prompt template string.
        reflect_strategy (str, optional): The strategy used to generate experiences. Defaults to "reflexion" if not specified.
        additional_keys (List[Dict[str, str]]): Additional keys for the prompt. Defaults to [].
        reflect_additional_keys (List[Dict[str, str]]): Additional keys for the reflect prompt. Defaults to [].
        patience (int, optional): The patience for the agent. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the question, key, trajectory, and reflections.
    """
    if not additional_keys:
        additional_keys = [{} for _ in range(len(questions))]

    if not reflect_additional_keys:
        reflect_additional_keys = [{} for _ in range(len(questions))]

    experiences = []
    for question, key, main_keys, reflect_keys in zip(
        questions, keys, additional_keys, reflect_additional_keys
    ):
        trajectory = reflexion_react_agent.generate(
            question=question,
            key=key,
            examples=examples,
            prompt=prompt,
            reflect_examples=reflect_examples,
            reflect_prompt=reflect_prompt,
            reflect_strategy=reflect_strategy,
            additional_keys=main_keys,  # type: ignore
            reflect_additional_keys=reflect_keys,  # type: ignore
            patience=patience,
            reset=True,
        )

        reflections = [
            trial.reflections
            for trial in trajectory.additional_info
            if trial.reflections
        ]
        selected_reflections = list(set(list(chain.from_iterable(reflections))))  # type: ignore
        experience = {
            "question": question,
            "key": key,
            "trajectory": trajectory,
            "reflections": selected_reflections,
        }
        experiences.append(experience)

    return experiences


# ============================================== Insight Extraction ==============================================


def categorize_experiences(experiences: List[Dict[str, Any]]) -> Dict[str, List]:
    """Categorizes experiences based on the success of trials in the trajectories.

    This function iterates over each index in the experiences and categorizes them into 'compare', 'success', or 'fail' based on the outcomes of the trials. Each trial is represented by a tuple, with the first element indicating success (True) or failure (False).

    Parameters:
        experiences (List[Dict[str, Any]]): A list of dictionaries, each containing the question, key, trajectory, and reflections.

    Returns:
        Dict[str, List]: A dictionary with the indices of tasks categorized into 'compare', 'success', and 'fail'.

    Raises:
    - ValueError: If a trajectory does not fit into any category, indicating an unhandled scenario.
    """
    count_dict: Dict[str, List] = {"compare": [], "success": [], "fail": []}

    for idx, experience in enumerate(experiences):
        trajectory = experience["trajectory"]
        trials_are_correct = [
            trial.steps[-1].is_correct for trial in trajectory.additional_info
        ]

        # Success.
        if (
            all(trials_are_correct) and len(trials_are_correct) == 1
        ):  # If success @ first trial, then stop generation.
            count_dict["success"].append(idx)
        # Compare.
        elif trials_are_correct[
            -1
        ]:  # If fail(s), then succeeds, then only last trial is True.
            count_dict["compare"].append(idx)
        # Fail.
        elif not all(trials_are_correct):  # All trials failed, then fail case.
            count_dict["fail"].append(idx)
        else:
            raise ValueError(f"Unhandled scenario for trajectory at index {idx}.")

    return count_dict


def get_folds(
    categories: Dict[str, List], n_instances: int, n_folds: int = 2, seed: int = 42
) -> Dict[int, List]:
    """Distributes indices into a specified number of stratified folds for cross-validation.

    Indices from each category ('compare', 'success', 'fail') are shuffled and then distributed across the folds. Each fold will serve as a validation set once during cross-validation, with the remaining data used for training.

    Parameters:
        categories (Dict[str, List]): A dictionary containing lists of indices for each category.
        n_instances (int): The total number of instances across all categories.
        n_folds (int, optional): The number of folds to create for cross-validation. Default is 2.

    Returns:
        Dict[int, List]: A dictionary where keys are fold indices and values are the lists of indices representing the training set for that fold.
    """
    random.seed(seed)

    folds: Dict[int, List] = {fold: [] for fold in range(n_folds)}

    # Assign labels for 'compare', 'success', and  'fail'.
    for _, indices in categories.items():
        indices = random.sample(indices, len(indices))
        for count, idx in enumerate(indices):
            folds[count % n_folds].append(idx)

    # Each fold is a validation set. Take the difference to get the training set of each fold.
    folds = {
        fold: list(set(list(range(n_instances))).difference(values))
        for fold, values in folds.items()
    }

    return folds


def _build_compare_prompt(
    insights: List[Dict[str, Any]],
    question: str,
    success_trial: str,
    failed_trial: str,
    is_full: bool,
) -> str:
    """Constructs a comparison prompt for an AI by combining system instructions, task details, and a list of existing insights.

    This function formats a prompt intended for AI to critique existing insights based on a given task. The task is described by a question and includes examples of both successful and failed trials.

    Parameters:
        insights (List[Tuple[str, int]]): A list of strings where each string represents an existing insight with a score. If the list is empty, it is treated as if there are no existing insights.
        question (str): The question that defines the task.
        success_trial (str): A description or example of a successful trial for the task.
        failed_trial (str): A description or example of a failed trial for the task.
        is_full (bool): A flag indicating whether the prompt should be in its full form or not. This affects the suffix of the critique summary.

    Returns:
        str: A fully constructed prompt ready to be presented to the AI. The prompt includes a prefixed system instruction, task details formatted according to human critique template,
            and a suffix based on whether the prompt is in its full form.
    """
    # System prompt.
    prefix = SYSTEM_TEMPLATE.format(
        ai_name=(
            NON_EXISTENT_INSIGHTS_AT_NAME if not insights else EXISTING_INSIGHTS_AI_NAME
        ),
        instruction=SYSTEM_CRITIQUE_EXISTING_INSIGHTS_INSTRUCTION,
    )

    # Task prompt.
    human_format_dict = {
        "question": question,
        "failed_traj": failed_trial,
        "success_traj": success_trial,
        "existing_insights": (
            "\n".join(
                [f"{i}. {insight['insight']}" for i, insight in enumerate(insights)]
            )
            if insights
            else ""
        ),
    }

    human_critique_summary_message = HUMAN_CRITIQUE_EXISTING_INSIGHTS_TEMPLATE.format(
        **human_format_dict
    )
    critique_summary_suffix = (
        CRITIQUE_SUMMARY_SUFFIX_FULL if is_full else CRITIQUE_SUMMARY_SUFFIX_NOT_FULL
    )

    prompt = prefix + "\n" + human_critique_summary_message + critique_summary_suffix

    return prompt


def _build_all_success_prompt(
    insights: List[Dict[str, Any]],
    success_trajs_str: str,
    is_full: bool,
) -> str:
    """Constructs a prompt focused on critiquing and enhancing existing insights based on successful task trials.

    This function generates a prompt for AI interaction that incorporates a series of successful trials and existing insights.

    Parameters:
        insights (List[Dict[str, Any]]): A list of strings where each string represents an existing insight with a score. If the list is empty, it is treated as if there are no existing insights.
        success_trajs_str (str): A string containing descriptions of successful trials related to the task. These descriptions are meant to provide context for the AI's critique of the existing insights.
        is_full (bool): A boolean flag that determines the verbosity of the critique summary's suffix. If `True`, a more comprehensive suffix is used.

    Returns:
        str: A string that combines the system's instruction, the task context with successful trials, and the existing insights into a coherent prompt.
    """
    # System prompt.
    prefix = SYSTEM_TEMPLATE.format(
        ai_name=(
            NON_EXISTENT_INSIGHTS_AT_NAME if not insights else EXISTING_INSIGHTS_AI_NAME
        ),
        instruction=SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_INSIGHTS_INSTRUCTION,
    )

    # Task prompt.
    human_format_dict = {
        "success_trajs": success_trajs_str,
        "existing_insights": (
            "\n".join(
                [f"{i}. {insight['insight']}" for i, insight in enumerate(insights)]
            )
            if insights
            else ""
        ),
    }

    human_critique_summary_message = (
        HUMAN_CRITIQUE_EXISTING_INSIGHTS_ALL_SUCCESS_TEMPLATE.format(
            **human_format_dict
        )
    )
    critique_summary_suffix = (
        CRITIQUE_SUMMARY_SUFFIX_FULL if is_full else CRITIQUE_SUMMARY_SUFFIX_NOT_FULL
    )

    prompt = prefix + "\n" + human_critique_summary_message + critique_summary_suffix

    return prompt


def _prompt_compare_critique(
    llm: BaseLLM,
    insights: List[Dict[str, Any]],
    question: str,
    success_trial: str,
    failed_trial: str,
    is_full: bool,
) -> Response:
    """Generates a critique from an LLM based on a comparison between successful and failed task trials, within the context of existing insights.

    This function constructs a prompt that juxtaposes successful and failed trials of a task with a set of existing insights. It then requests a critique from the Large Language Model (LLM) based on this information. The critique aims to evaluate the insights' effectiveness and suggest modifications if necessary. An option is provided to format the LLM's output by removing newline characters.

    Parameters:
        llm (BaseLLM): The Large Language Model instance used to generate the critique.
        insights (List[Dict[str, Any]]): A list of strings where each string represents an existing insight with a score. If the list is empty, it is treated as if there are no existing insights.
        question (str): The task question related to the trials.
        success_trial (str): A description of a successful trial for the task.
        failed_trial (str): A description of a failed trial for the task.
        is_full (bool): A flag indicating if the full version of the critique summary should be used.

    Returns:
        Response: The critique generated by the LLM, potentially with newline characters removed, based on the `replace_newline` parameter.
    """
    prompt = _build_compare_prompt(
        insights=insights,
        question=question,
        success_trial=success_trial,
        failed_trial=failed_trial,
        is_full=is_full,
    )

    out = llm(prompt)

    return out


def _prompt_all_success_critique(
    llm: BaseLLM,
    insights: List[Dict[str, Any]],
    success_trajs_str: str,
    is_full: bool,
) -> Response:
    """Generates a critique from an LLM based on a compilation of successful task trials in the context of existing insights.

    This function constructs a prompt emphasizing the successes in task trials and existing insights, and requests a critique from the Large Language Model (LLM).

    Parameters:
        llm (BaseLLM): The Large Language Model instance used for generating the critique.
        insights (List[Dict[str, Any]]): A list of strings where each string represents an existing insight with a score. If the list is empty, it is treated as if there are no existing insights.
        success_trajs_str (str): A string concatenating descriptions of successful trials related to the task.
        is_full (bool): Indicates whether the full critique summary is to be used in the prompt.

    Returns:
        Response: The generated critique from the LLM, optionally with newline characters removed depending on the `replace_newline` parameter.
    """
    prompt = _build_all_success_prompt(
        insights=insights,
        success_trajs_str=success_trajs_str,
        is_full=is_full,
    )

    out = llm(prompt)

    return out


def parse_insights(llm_text: str) -> List[Tuple[str, str]]:
    """Parses and extracts insight operations and their descriptions from a given text.

    This function searches through the provided text for occurrences of insight operations (ADD, REMOVE, EDIT, AGREE) followed by their descriptions.
    It applies specific criteria to ensure the extracted insights are valid: the insight description must not be empty, must not
    contain certain banned words (to avoid inclusion of formatting instructions or similar), and must end with a period.

    Parameters:
        llm_text (str): The text from which to extract insight operations and descriptions.
            This text is expected to contain one or more statements formatted according to predefined insight operation patterns.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains two elements: the operation (ADD, REMOVE, EDIT, AGREE) and the clean, validated insight description.
            The insights that do not meet the validation criteria are omitted.
    """
    pattern = r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)"
    matches = re.findall(pattern, llm_text)

    res = []
    banned_words = ["ADD", "AGREE", "EDIT"]
    for operation, text in matches:
        text = text.strip()
        if (
            text != ""
            and not any([w in text for w in banned_words])
            and text.endswith(".")
        ):
            # If text is not empty.
            # If text doesn't contain banned words (avoid weird formatting cases from llm).
            # If text ends with a period (avoid cut off sentences from llm).
            if "ADD" in operation:
                res.append(("ADD", text))
            else:
                res.append((operation.strip(), text))
    return res


def retrieve_insight_index(
    insights: List[Dict[str, Any]], operation_rule_text: str
) -> int:
    """Retrieves the index of a rule based on its text.

    Searches through a list of insights to find the index of the rule that matches part of the given operation rule text. This function is useful for identifying which rule is being referred to in operations like EDIT, REMOVE, or AGREE, where the rule text is included in the operation.

    Parameters:
        insights (List[Dict[str, Any]]): A list of tuples, where each tuple contains the rule text and its associated strength or any other numeric value.
        operation_rule_text (str): The text of the operation which may contain or exactly match the text of a rule.

    Returns:
        int: The index of the rule within the list if found; otherwise, -1.
    """
    for i in range(len(insights)):
        if insights[i]["insight"] in operation_rule_text:
            return i
    return -1


def remove_err_operations(
    insights: List[Dict[str, Any]], operations: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Cleans a list of rule operations by removing or modifying erroneous entries.

    This function iterates through a list of operations intended to modify a set of insights. It removes operations that are incorrect or not applicable (e.g., attempting to add a rule that already exists) and modifies certain operations based on their context (e.g., changing an EDIT to AGREE if the edited rule matches an existing rule). The goal is to ensure that the resulting list of operations is coherent and can be applied to update the insights without causing inconsistencies.

    Parameters:
        insights (List[Dict[str, Any]]): A list of tuples representing the existing insights. Each tuple contains the rule text and an associated numeric value, which could represent the rule's strength or priority.
        operations (List[Tuple[str, str]]): A list of tuples representing the operations to be performed on the insights. Each tuple contains an operation type (ADD, REMOVE, EDIT, AGREE) and the associated rule text or modification.

    Returns:
        List[Tuple[str, str]]: A cleaned list of operations where erroneous or inapplicable operations have been removed or modified to ensure consistency and correctness when applied to the set of existing insights.
    """
    corrected_operations = []
    for operation, text in operations.copy():
        operation_type = operation.split(" ")[0]
        insight_idx = int(operation.split(" ")[1]) if " " in operation else None
        index = retrieve_insight_index(insights, text)

        # ADDing an insight that doesn't exist.
        if operation_type == "ADD" and retrieve_insight_index(insights, text) == -1:
            corrected_operations.append((operation, text))
        # REMOVEing or AGREEing with an insight given that it exists.
        elif (operation_type == "REMOVE" or operation_type == "AGREE") and index != -1:
            corrected_operations.append((operation, text))
        # EDITing an insight (AGREEing) given that it exists.
        elif operation_type == "EDIT" and index != -1:
            corrected_operations.append((f"AGREE {index}", text))
        # EDITing an insight given:
        # - it doesn't exist (text match) in the insights
        # - the insight index to EDIT is not None
        # - the insight index to EDIT is less than or equal to the length of insights (within range of the length of the insights)
        elif (
            operation_type == "EDIT"
            and insight_idx is not None
            and insight_idx <= len(insights)
        ):
            corrected_operations.append((operation, text))

    return corrected_operations


def accumulate_metrics(
    compares_response: List[List[Response]],
    successes_response: List[List[Response]],
    experiences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Accumulates various metrics from a set of responses and experiences.

    This function takes in lists of comparison responses, success responses, and experiences, and calculates various metrics such as total prompt tokens, completion tokens, total tokens, prompt cost, completion cost, total cost, and prompt time. The results are returned as a dictionary.

    Parameters:
        compares_response (List[List[Response]]): A list of lists of comparison responses.
        successes_response (List[List[Response]]): A list of lists of success responses.
        experiences (List[Dict[str, Any]]): A list of experiences.

    Returns:
        Dict[str, Any]: A dictionary containing the accumulated metrics.
    """
    total_prompt_tokens = 0.0
    total_completion_tokens = 0.0
    total_tokens = 0.0
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    total_cost = 0.0
    total_prompt_time = 0.0

    for compare_response, success_response in zip(
        compares_response, successes_response
    ):
        for compare in compare_response:
            total_prompt_tokens += compare.prompt_tokens
            total_completion_tokens += compare.completion_tokens
            total_tokens += compare.total_tokens
            total_prompt_cost += compare.prompt_cost
            total_completion_cost += compare.completion_cost
            total_cost += compare.total_cost
            total_prompt_time += compare.prompt_time

        for success in success_response:
            total_prompt_tokens += success.prompt_tokens
            total_completion_tokens += success.completion_tokens
            total_tokens += success.total_tokens
            total_prompt_cost += success.prompt_cost
            total_completion_cost += success.completion_cost
            total_cost += success.total_cost
            total_prompt_time += success.prompt_time

    for experience in experiences:
        trajectory: ReflexionReActOutput = experience["trajectory"]
        total_prompt_tokens += trajectory.total_prompt_tokens
        total_completion_tokens += trajectory.total_completion_tokens
        total_tokens += trajectory.total_tokens
        total_prompt_cost += trajectory.total_prompt_cost
        total_completion_cost += trajectory.total_completion_cost
        total_cost += trajectory.total_cost
        total_prompt_time += trajectory.total_prompt_time

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_prompt_cost": total_prompt_cost,
        "total_completion_cost": total_completion_cost,
        "total_cost": total_cost,
        "total_prompt_time": total_prompt_time,
    }
