"""Functional module for ExpeL."""

import random
import re

from typing import Dict, List, Optional, Tuple, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts.prompt import PromptTemplate

from discussion_agents.cog.agent.reflexion import ReflexionReActAgent
from discussion_agents.cog.prompts.react import REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES
from discussion_agents.cog.prompts.reflexion import (
    REFLEXION_REACT_REFLECT_FEWSHOT_EXAMPLES,
    REFLEXION_REACT_REFLECT_INSTRUCTION,
    REFLEXION_REACT_INSTRUCTION
)
from discussion_agents.cog.prompts.expel import (
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
from discussion_agents.utils.general import shuffle_chunk_list

# ============================================== Experience Gathering ==============================================


def gather_experience(
    reflexion_react_agent: ReflexionReActAgent,
    questions: List[str],
    keys: List[str],
    strategy: Optional[str] = "reflexion",
    prompt: str = REFLEXION_REACT_INSTRUCTION,
    examples: Optional[str] = REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
    reflect_examples: str = REFLEXION_REACT_REFLECT_FEWSHOT_EXAMPLES,
    reflect_prompt: str = REFLEXION_REACT_REFLECT_INSTRUCTION,
) -> Dict[str, List]:
    """Collects and aggregates experiences from a ReflexionReActAgent by generating trajectories and reflections for a set of questions and keys.

    The function iterates over each question-key pair, generates a trajectory using the specified strategy, and records the reflections generated by the agent. Each trajectory and its corresponding reflections are appended to their respective lists within the 'experiences' dictionary.

    Parameters:
        reflexion_react_agent (ReflexionReActAgent): The agent from which experiences are generated.
        questions (List[str]): A list of questions to be processed by the agent.
        keys (List[str]): A list of keys that are paired with the questions to guide the agent's generation.
        strategy (Optional[str]): The strategy used to generate experiences. Defaults to "reflexion" if not specified.
        prompt (str, optional): Prompt template string. Defaults to REFLEXION_REACT_INSTRUCTION.
            Must include examples, reflections, question, scratchpad, and max_steps.
        examples (str, optional): Fewshot examples. Defaults to REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES.
        reflect_examples (str, optional): Reflection fewshot examples. Defaults to REFLEXION_REACT_REFLECT_FEWSHOT_EXAMPLES.
        reflect_prompt (str, optional): Reflect prompt template string. Defaults to REFLEXION_REACT_REFLECT_INSTRUCTION.
            Must include examples, question, and scratchpad.

    Returns:
        Dict[str, List]: A dictionary containing lists of indices ('idxs'), questions ('questions'), keys ('keys'), generated trajectories ('trajectories'), and reflections ('reflections').

    Each index in 'idxs' corresponds to the respective question, key, trajectory, and reflections at the same position in their lists.
    """
    experiences: Dict[str, List] = {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    for idx, (question, key) in enumerate(zip(questions, keys)):
        trajectory = reflexion_react_agent.generate(
            question=question, 
            key=key, 
            strategy=strategy, 
            reset=True,
            prompt=prompt,
            examples=examples,
            reflect_examples=reflect_examples,
            reflect_prompt=reflect_prompt
        )

        experiences["idxs"].append(idx)
        experiences["questions"].append(question)
        experiences["keys"].append(key)
        experiences["trajectories"].append(trajectory)
        experiences["reflections"].append(reflexion_react_agent.reflector.reflections)

    return experiences


# ============================================== Insight Extraction ==============================================


def categorize_experiences(experiences: Dict[str, List]) -> Dict[str, List]:
    """Categorizes experiences based on the success of trials in the trajectories.

    This function iterates over each index in the experiences and categorizes them into 'compare', 'success', or 'fail' based on the outcomes of the trials. Each trial is represented by a tuple, with the first element indicating success (True) or failure (False).

    Parameters:
        experiences (Dict[str, List]): A dictionary containing the trajectories to be categorized. The dictionary should have the following structure:
            {
                "idxs": List[int],  # Indices of the tasks
                "trajectories": List[List[Tuple[bool, Any, Any]]]  # Trajectories as a list of tuples
            }

    Returns:
        Dict[str, List]: A dictionary with the indices of tasks categorized into 'compare', 'success', and 'fail'.

    Raises:
    - ValueError: If a trajectory does not fit into any category, indicating an unhandled scenario.
    """
    count_dict: Dict[str, List] = {"compare": [], "success": [], "fail": []}

    for idx in experiences["idxs"]:  # Index for a particular task.
        trajectory = experiences["trajectories"][idx]  # type: ignore
        trials_are_correct = [
            trial[0] for trial in trajectory
        ]  # (is_correct, answer, output)[0].

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
    prefix = PromptTemplate.from_template(SYSTEM_TEMPLATE).format(
        ai_name=NON_EXISTENT_INSIGHTS_AT_NAME if not insights else EXISTING_INSIGHTS_AI_NAME,
        instruction=SYSTEM_CRITIQUE_EXISTING_INSIGHTS_INSTRUCTION,
    )

    # Task prompt.
    human_format_dict = {
        "question": question,
        "failed_traj": failed_trial,
        "success_traj": success_trial,
        "existing_insights": "\n".join(
            [f"{i}. {insight['insight']}" for i, insight in enumerate(insights, 1)]
        )  if insights else "",
    }

    human_critique_summary_message = PromptTemplate.from_template(
        HUMAN_CRITIQUE_EXISTING_INSIGHTS_TEMPLATE
    ).format(**human_format_dict)
    critique_summary_suffix = (
        CRITIQUE_SUMMARY_SUFFIX_FULL if is_full else CRITIQUE_SUMMARY_SUFFIX_NOT_FULL
    )

    prompt = prefix + "\n" + human_critique_summary_message + critique_summary_suffix

    return prompt


def _build_all_success_prompt(
    insights: List[Tuple[str, int]],
    success_trajs_str: str,
    is_full: bool,
) -> str:
    """Constructs a prompt focused on critiquing and enhancing existing insights based on successful task trials.

    This function generates a prompt for AI interaction that incorporates a series of successful trials and existing insights.

    Parameters:
        insights (List[Tuple[str, int]]): A list of strings where each string represents an existing insight with a score. If the list is empty, it is treated as if there are no existing insights.
        success_trajs_str (str): A string containing descriptions of successful trials related to the task. These descriptions are meant to provide context for the AI's critique of the existing insights.
        is_full (bool): A boolean flag that determines the verbosity of the critique summary's suffix. If `True`, a more comprehensive suffix is used.

    Returns:
        str: A string that combines the system's instruction, the task context with successful trials, and the existing insights into a coherent prompt.
    """
    # System prompt.
    prefix = PromptTemplate.from_template(SYSTEM_TEMPLATE).format(
        ai_name=NON_EXISTENT_INSIGHTS_AT_NAME if not insights else EXISTING_INSIGHTS_AI_NAME,
        instruction=SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_INSIGHTS_INSTRUCTION,
    )

    # Task prompt.
    human_format_dict = {
        "success_trajs": success_trajs_str,
        "existing_insights": "\n".join(
            [f"{i}. {insight['insight']}" for i, insight in enumerate(insights, 1)]
        ) if insights else "",
    }

    human_critique_summary_message = PromptTemplate.from_template(
        HUMAN_CRITIQUE_EXISTING_INSIGHTS_ALL_SUCCESS_TEMPLATE
    ).format(**human_format_dict)
    critique_summary_suffix = (
        CRITIQUE_SUMMARY_SUFFIX_FULL if is_full else CRITIQUE_SUMMARY_SUFFIX_NOT_FULL
    )

    prompt = prefix + "\n" + human_critique_summary_message + critique_summary_suffix

    return prompt


def _prompt_compare_critique(
    llm: BaseChatModel,
    insights: List[Tuple[str, int]],
    question: str,
    success_trial: str,
    failed_trial: str,
    is_full: bool,
    replace_newline: bool = False,
) -> str:
    """Generates a critique from an LLM based on a comparison between successful and failed task trials, within the context of existing insights.

    This function constructs a prompt that juxtaposes successful and failed trials of a task with a set of existing insights. It then requests a critique from the Large Language Model (LLM) based on this information. The critique aims to evaluate the insights' effectiveness and suggest modifications if necessary. An option is provided to format the LLM's output by removing newline characters.

    Parameters:
        llm (BaseChatModel): The Large Language Model instance used to generate the critique.
        insights (List[Tuple[str, int]]): A list of strings where each string represents an existing insight with a score. If the list is empty, it is treated as if there are no existing insights.
        question (str): The task question related to the trials.
        success_trial (str): A description of a successful trial for the task.
        failed_trial (str): A description of a failed trial for the task.
        is_full (bool): A flag indicating if the full version of the critique summary should be used.
        replace_newline (bool, optional): If `True`, newline characters in the LLM's output will be replaced with empty strings, defaulting to `False`.

    Returns:
        str: The critique generated by the LLM, potentially with newline characters removed, based on the `replace_newline` parameter.
    """
    prompt = _build_compare_prompt(
        insights=insights,
        question=question,
        success_trial=success_trial,
        failed_trial=failed_trial,
        is_full=is_full,
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    out = out.strip("\n").strip()  # type: ignore

    if replace_newline:
        out = out.replace("\n", "")
    return out


def _prompt_all_success_critique(
    llm: BaseChatModel,
    insights: List[Tuple[str, int]],
    success_trajs_str: str,
    is_full: bool,
    replace_newline: bool = False,
) -> str:
    """Generates a critique from an LLM based on a compilation of successful task trials in the context of existing insights.

    This function constructs a prompt emphasizing the successes in task trials and existing insights, and requests a critique from the Large Language Model (LLM).

    Parameters:
        llm (BaseChatModel): The Large Language Model instance used for generating the critique.
        insights (List[Tuple[str, int]]): A list of strings where each string represents an existing insight with a score. If the list is empty, it is treated as if there are no existing insights.
        success_trajs_str (str): A string concatenating descriptions of successful trials related to the task.
        is_full (bool): Indicates whether the full critique summary is to be used in the prompt.
        replace_newline (bool, optional): If set to `True`, newline characters in the LLM output will be replaced with empty strings. The default is `False`.

    Returns:
        str: The generated critique from the LLM, optionally with newline characters removed depending on the `replace_newline` parameter.
    """
    prompt = _build_all_success_prompt(
        insights=insights, success_trajs_str=success_trajs_str, is_full=is_full
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    out = out.strip("\n").strip()  # type: ignore

    if replace_newline:
        out = out.replace("\n", "")
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


def retrieve_insight_index(insights: List[Tuple[str, int]], operation_rule_text: str) -> int:
    """Retrieves the index of a rule based on its text.

    Searches through a list of insights to find the index of the rule that matches part of the given operation rule text. This function is useful for identifying which rule is being referred to in operations like EDIT, REMOVE, or AGREE, where the rule text is included in the operation.

    Parameters:
        insights (List[Tuple[str, int]]): A list of tuples, where each tuple contains the rule text and its associated strength or any other numeric value.
        operation_rule_text (str): The text of the operation which may contain or exactly match the text of a rule.

    Returns:
        int: The index of the rule within the list if found; otherwise, -1.
    """
    for i in range(len(insights)):
        if insights[i]['insight'] in operation_rule_text:
            return i
    return -1


def is_existing_rule(insights: List[Tuple[str, int]], operation_rule_text: str) -> bool:
    """Checks if a rule exists based on its text.

    Determines whether any rule's text in the provided list of insights matches part of the given operation rule text. This is useful for verifying if an operation like ADD is attempting to add a rule that already exists based on its text.

    Parameters:
        insights (List[Tuple[str, int]]): A list of tuples, where each tuple contains the rule text and its associated strength or any other numeric value.
        operation_rule_text (str): The text of the operation which may contain or exactly match the text of an existing rule.

    Returns:
        bool: True if the rule exists in the list, otherwise False.
    """
    for i in range(len(insights)):
        if insights[i]['insight'] in operation_rule_text:
            return True
    return False


def remove_err_operations(
    insights: List[Tuple[str, int]], operations: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Cleans a list of rule operations by removing or modifying erroneous entries.

    This function iterates through a list of operations intended to modify a set of insights. It removes operations that are incorrect or not applicable (e.g., attempting to add a rule that already exists) and modifies certain operations based on their context (e.g., changing an EDIT to AGREE if the edited rule matches an existing rule). The goal is to ensure that the resulting list of operations is coherent and can be applied to update the insights without causing inconsistencies.

    Parameters:
        insights (List[Tuple[str, int]]): A list of tuples representing the existing insights. Each tuple contains the rule text and an associated numeric value, which could represent the rule's strength or priority.
        operations (List[Tuple[str, str]]): A list of tuples representing the operations to be performed on the insights. Each tuple contains an operation type (ADD, REMOVE, EDIT, AGREE) and the associated rule text or modification.

    Returns:
        List[Tuple[str, str]]: A cleaned list of operations where erroneous or inapplicable operations have been removed or modified to ensure consistency and correctness when applied to the set of existing insights.
    """
    cleaned_operations = operations.copy()

    delete_indices = []
    for i in range(len(cleaned_operations)):
        # Split the operation into action type and optional rule number.
        operation, operation_rule_text = cleaned_operations[i]
        operation_type = operation.split(" ")[0]
        rule_num = int(operation.split(" ")[1]) if " " in operation else None

        if operation_type == "ADD":
            if is_existing_rule(
                insights, operation_rule_text
            ):  # If new rule_text is an existing rule ('in').
                delete_indices.append(i)
        else:
            if operation_type == "EDIT":
                if is_existing_rule(
                    insights, operation_rule_text
                ):  # If rule is matching ('in') existing rule, change it to AGREE.
                    rule_num = retrieve_insight_index(insights, operation_rule_text)
                    cleaned_operations[i] = (f"AGREE {rule_num+1}", insights[rule_num][0])
                elif (rule_num is None) or (
                    rule_num > len(insights)
                ):  # If rule doesn't exist, remove.
                    delete_indices.append(i)

            elif operation_type == "REMOVE" or operation_type == "AGREE":
                if not is_existing_rule(
                    insights, operation_rule_text
                ):  # If new operation_rule_text is not an existing rule.
                    delete_indices.append(i)

    # Remove problematic operations.
    cleaned_operations = [
        cleaned_operations[i]
        for i in range(len(cleaned_operations))
        if i not in delete_indices
    ]

    return cleaned_operations


def update_rules(
    insights: List[Tuple[str, int]],
    operations: List[Tuple[str, str]],
    is_full: bool = False,
) -> List[Tuple[str, int]]:
    """Updates a set of insights based on provided operations, adjusting their strengths and possibly adding or editing them.

    Operations are processed in a specific order: 'REMOVE', 'AGREE', 'EDIT', and 'ADD'. This ensures that removals and agreements are handled first, followed by edits and additions.
    The function supports dynamically adjusting the impact of a 'REMOVE' operation based on the `is_full` flag, which represents whether the set of insights is considered comprehensive.

    Parameters:
        insights (List[Tuple[str, int]]): The current set of insights, where each rule is represented by a tuple containing the rule text and its associated numeric value (e.g., strength, priority).
        operations (List[Tuple[str, str]]): Operations to be applied to the insights, with each operation being a tuple of the operation type and the rule text.
        is_full (bool, optional): A flag indicating if the insights set is considered comprehensive, affecting the strength adjustment of 'REMOVE' operations.

    Returns:
        List[Tuple[str, int]]: The updated set of insights after applying the operations, sorted by their numeric values in descending order. insights with a numeric value of 0 or less are removed from the set.
    """
    updated_rules = insights.copy()

    for i in range(len(operations)):
        operation, operation_rule_text = operations[i]
        operation_type = operation.split(" ")[0]

        if operation_type == "REMOVE":  # remove rule: -1
            rule_index = retrieve_insight_index(
                updated_rules, operation_rule_text
            )  # if rule_num doesn't match but text does
            remove_strength = 3 if is_full else 1
            updated_rules[rule_index] = (
                updated_rules[rule_index][0],
                updated_rules[rule_index][1] - remove_strength,
            )  # -1 (-3 if list full) to the counter
        elif operation_type == "AGREE":  # agree with rule: +1
            rule_index = retrieve_insight_index(
                updated_rules, operation_rule_text
            )  # if rule_num doesn't match but text does
            updated_rules[rule_index] = (
                updated_rules[rule_index][0],
                updated_rules[rule_index][1] + 1,
            )  # +1 to the counter
        elif (
            operation_type == "EDIT"
        ):  # edit the rule: +1 // NEED TO BE AFTER REMOVE AND AGREE
            rule_index = int(operation.split(" ")[1]) - 1
            updated_rules[rule_index] = (
                operation_rule_text,
                updated_rules[rule_index][1] + 1,
            )  # +1 to the counter
        elif operation_type == "ADD":  # add new rule: +2
            updated_rules.append((operation_rule_text, 2))

    updated_rules = [
        updated_rules[i] for i in range(len(updated_rules)) if updated_rules[i][1] > 0
    ]  # remove insights when counter reach 0
    updated_rules.sort(key=lambda x: x[1], reverse=True)

    return updated_rules


def get_operations_compare(
    llm: BaseChatModel,
    insights: List[Tuple[str, int]],
    question: str,
    success_trial: str,
    failed_trial: str,
    is_full: bool
) -> List[Tuple[str, str]]:
    # Prompt.
    out = _prompt_compare_critique(
        llm,
        insights,
        question,
        success_trial,
        failed_trial,
        is_full,
    )

    # Parse.
    operations = parse_insights(out)

    # Remove no-ops.
    operations = remove_err_operations(insights, operations)

    return operations


def get_operations_success(
    llm: BaseChatModel,
    success_trials: str,
    insights: List[Tuple[str, int]],
    is_full: bool
) -> List[Tuple[str, str]]:
    # Prompt.
    out = _prompt_all_success_critique(
        llm, insights, success_trials, is_full
    )

    # Parse.
    operations = parse_insights(out)

    # Remove no-ops.
    operations = remove_err_operations(insights, operations)

    return operations
    

def create_rules(
    llm: BaseChatModel,
    experiences: Dict[str, List],
    categories: Dict[str, int],
    train_idxs: List[int],
    insights: List[Tuple[str, int]],
    max_num_rules: int,
    success_critique_num: int = 8,
) -> List[Tuple[str, int]]:
    """Generates and updates insights based on experiences categorized as compare and success.

    This function iteratively refines a set of insights by evaluating experiences through the lens of compare and success categories.
    For compare experiences, it juxtaposes successful trials against failed ones to draw insights.
    For success experiences, it aggregates successful trials to distill overarching successful strategies.
    The insights drawn from these analyses are used to add, edit, remove, or agree with existing insights, thereby refining the rule set.

    Parameters:
        llm (BaseChatModel): An instance of a Large Language Model used to generate critiques and insights.
        experiences (Dict[str, List]): A dictionary containing lists of questions, keys, and trajectories categorized by task indices.
        categories (Dict[str, int]): A dictionary categorizing experiences into compare, success, and fail based on indices.
        train_idxs (List[int]): Indices of training data to be used for rule generation.
        insights (List[Tuple[str, int]]): The current set of insights and their associated strength scores.
        max_num_rules (int): The maximum number of insights to retain.
        success_critique_num (int, optional): The number of successes to batch together for critique. Defaults to 8.

    Returns:
        List[Tuple[str, int]]: A list of updated insights as strings and their associated strength scores.

    Note:
        - The function internally utilizes helper functions to prompt the LLM, parse its output for operations on insights, and update the rule set accordingly.
        - The operation of adding, editing, removing, or agreeing with insights is based on the critique provided by the LLM in response to the prompts generated from experiences.
        - The function ensures that the rule set does not exceed the specified maximum number of insights, prioritizing insights by their strength scores.
    """
    # Intersection between train_idxs and each category (compare, success, fail).
    train_category_idxs = {
        category: list(set(train_idxs).intersection(set(category_idxs)))  # type: ignore
        for category, category_idxs in categories.items()
    }

    # Compare.
    for train_idx in train_category_idxs["compare"]:
        question = experiences["questions"][train_idx]
        trajectory = experiences["trajectories"][
            train_idx
        ]  # List[Tuple[bool, str, List[Tuple[str, str, str]]]].

        # Compare the successful trial with all previous failed trials.
        success_trial = "\n".join(["\n".join(step) for step in trajectory[-1][-1]])
        for failed_trial in trajectory[:-1]:
            failed_trial = "\n".join(["\n".join(step) for step in failed_trial[-1]])

            # Prompt.
            out = _prompt_compare_critique(
                llm,
                insights,
                question,
                success_trial,
                failed_trial,
                max_num_rules < len(insights),
            )

            # Parse.
            operations = parse_insights(out)

            # Remove no-ops.
            operations = remove_err_operations(insights, operations)

            # Update insights with comparison insights.
            insights = update_rules(
                insights,
                operations,
                is_full=max_num_rules + 5 <= len(insights),
            )

    # Success.
    batched_success_trajs_idxs = shuffle_chunk_list(
        train_category_idxs["success"], success_critique_num
    )
    for success_idxs in batched_success_trajs_idxs:
        # Concatenate batched successful trajectories.
        concat_success_trajs = []
        for idx in success_idxs:
            success_traj_str = "\n".join(
                ["\n".join(step) for step in experiences["trajectories"][idx][0][-1]]
            )
            concat_success_trajs.append(
                f"{experiences['questions'][idx]}\n{success_traj_str}"
            )
        success_trajs_str = "\n\n".join(concat_success_trajs)

        # Prompt.
        out = _prompt_all_success_critique(
            llm, insights, success_trajs_str, max_num_rules < len(insights)
        )

        # Parse.
        operations = parse_insights(out)

        # Remove no-ops.
        operations = remove_err_operations(insights, operations)

        # Update insights with success insights.
        insights = update_rules(
            insights,
            operations,
            is_full=max_num_rules + 5 <= len(insights),
        )

    return insights