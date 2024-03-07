"""Functional module for ExpeL."""
import re
import random
from typing import List, Dict, Optional, Tuple
from discussion_agents.cog.agent.reflexion import ReflexionReActAgent
from langchain_core.messages.human import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.prompt import PromptTemplate

from discussion_agents.cog.prompts.expel import (
    SYSTEM_TEMPLATE,
    SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION,
    EXISTING_RULES_AI_NAME,
    NON_EXISTENT_RULES_AT_NAME,
    HUMAN_CRITIQUE_EXISTING_RULES_TEMPLATE,
    CRITIQUE_SUMMARY_SUFFIX_FULL,
    CRITIQUE_SUMMARY_SUFFIX_NOT_FULL,
    SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION,
    HUMAN_CRITIQUE_EXISTING_RULES_ALL_SUCCESS_TEMPLATE,
)
from discussion_agents.utils.general import shuffle_chunk_list

# ============================================== Experience Gathering ==============================================


def gather_experience(
    reflexion_react_agent: ReflexionReActAgent,
    questions: List[str],
    keys: List[str],
    strategy: Optional[str] = "reflexion",
) -> Dict[str, List]:
    """Collects and aggregates experiences from a ReflexionReActAgent by generating trajectories and reflections for a set of questions and keys.

    The function iterates over each question-key pair, generates a trajectory using the specified strategy, and records the reflections generated by the agent. Each trajectory and its corresponding reflections are appended to their respective lists within the 'experiences' dictionary.

    Parameters:
        reflexion_react_agent (ReflexionReActAgent): The agent from which experiences are generated.
        questions (List[str]): A list of questions to be processed by the agent.
        keys (List[str]): A list of keys that are paired with the questions to guide the agent's generation.
        strategy (Optional[str]): The strategy used to generate experiences. Defaults to "reflexion" if not specified.

    Returns:
        Dict[str, List]: A dictionary containing lists of indices ('idxs'), questions ('questions'), keys ('keys'), generated trajectories ('trajectories'), and reflections ('reflections').

    Each index in 'idxs' corresponds to the respective question, key, trajectory, and reflections at the same position in their lists.
    """
    experiences = {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    for idx, (question, key) in enumerate(zip(questions, keys)):
        trajectory = reflexion_react_agent.generate(
            question=question, key=key, strategy=strategy, reset=True
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
    count_dict = {"compare": [], "success": [], "fail": []}

    for idx in experiences["idxs"]:  # Index for a particular task.
        trajectory = experiences["trajectories"][idx]
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
    categories: Dict[str, List], n_instances: int, n_folds: int = 2
) -> Dict[str, List]:
    """Distributes indices into a specified number of stratified folds for cross-validation.

    Indices from each category ('compare', 'success', 'fail') are shuffled and then distributed across the folds. Each fold will serve as a validation set once during cross-validation, with the remaining data used for training.

    Parameters:
        categories (Dict[str, List]): A dictionary containing lists of indices for each category.
        n_instances (int): The total number of instances across all categories.
        n_folds (int, optional): The number of folds to create for cross-validation. Default is 2.

    Returns:
        Dict[str, List]: A dictionary where keys are fold indices and values are the lists of indices representing the training set for that fold.
    """
    folds = {fold: [] for fold in range(n_folds)}

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
    rules: List[str],
    question: str,
    success_trial: str,
    failed_trial: str,
    is_full: bool,
) -> str:

    if rules == []:
        rules = [""]

    # System prompt.
    prefix = PromptTemplate.from_template(SYSTEM_TEMPLATE).format(
        ai_name=NON_EXISTENT_RULES_AT_NAME if not rules else EXISTING_RULES_AI_NAME,
        instruction=SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION,
    )

    # Task prompt.
    human_format_dict = {
        "question": question,
        "failed_traj": failed_trial,
        "success_traj": success_trial,
        "existing_rules": "\n".join([f"{i}. {r}" for i, r in enumerate(rules, 1)]),
    }

    human_critique_summary_message = PromptTemplate.from_template(
        HUMAN_CRITIQUE_EXISTING_RULES_TEMPLATE
    ).format(**human_format_dict)
    critique_summary_suffix = (
        CRITIQUE_SUMMARY_SUFFIX_FULL if is_full else CRITIQUE_SUMMARY_SUFFIX_NOT_FULL
    )

    prompt = prefix + "\n" + human_critique_summary_message + critique_summary_suffix

    return prompt


def _build_all_success_prompt(
    rules: List[str],
    success_trajs_str: str,
    is_full: bool,
) -> str:

    if rules == []:
        rules = [""]

    # System prompt.
    prefix = PromptTemplate.from_template(SYSTEM_TEMPLATE).format(
        ai_name=NON_EXISTENT_RULES_AT_NAME if not rules else EXISTING_RULES_AI_NAME,
        instruction=SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION,
    )

    # Task prompt.
    human_format_dict = {
        "success_trajs": success_trajs_str,
        "existing_rules": "\n".join([f"{i}. {rule}" for i, rule in enumerate(rules, 1)]),
    }

    human_critique_summary_message = PromptTemplate.from_template(
        HUMAN_CRITIQUE_EXISTING_RULES_ALL_SUCCESS_TEMPLATE
    ).format(**human_format_dict)
    critique_summary_suffix = (
        CRITIQUE_SUMMARY_SUFFIX_FULL if is_full else CRITIQUE_SUMMARY_SUFFIX_NOT_FULL
    )

    prompt = prefix + "\n" + human_critique_summary_message + critique_summary_suffix

    return prompt


def _prompt_compare_critique(
    llm: BaseChatModel,
    rules: List[str],
    question: str,
    success_trial: str,
    failed_trial: str,
    is_full: bool,
    replace_newline: bool = False,
) -> str:
    prompt = _build_compare_prompt(
        rules=rules,
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
    ).content.strip("\n").strip()

    if replace_newline:
        out = out.replace("\n", "")
    return out


def _prompt_all_success_critique(
    llm: BaseChatModel,
    rules: List[str],
    success_trajs_str: str,
    is_full: bool,
    replace_newline: bool = False,
) -> str:
    prompt = _build_all_success_prompt(
        rules=rules, success_trajs_str=success_trajs_str, is_full=is_full
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content.strip("\n").strip()
    
    if replace_newline:
        out = out.replace("\n", "")
    return out


def parse_rules(llm_text: str) -> str:
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
            # if text is not empty
            # if text doesn't contain banned words (avoid weird formatting cases from llm)
            # if text ends with a period (avoid cut off sentences from llm)
            if "ADD" in operation:
                res.append(("ADD", text))
            else:
                res.append((operation.strip(), text))
    return res


def retrieve_rule_index(rules: List[Tuple[str, int]], operation_rule_text: str) -> int:
    for i in range(len(rules)):
        if rules[i][0] in operation_rule_text:
            return i
    return -1


def is_existing_rule(rules: List[Tuple[str, int]], operation_rule_text: str) -> bool:
    for i in range(len(rules)):
        if rules[i][0] in operation_rule_text:
            return True
    return False


def remove_err_operations(
    rules: List[Tuple[str, int]], operations: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    cleaned_operations = operations.copy()

    delete_indices = []
    for i in range(len(cleaned_operations)):
        # Split the operation into action type and optional rule number.
        operation, operation_rule_text = cleaned_operations[i]
        operation_type = operation.split(" ")[0]
        rule_num = int(operation.split(" ")[1]) if " " in operation else None

        if operation_type == "ADD":
            if is_existing_rule(
                rules, operation_rule_text
            ):  # If new rule_text is an existing rule ('in').
                delete_indices.append(i)
        else:
            if operation_type == "EDIT":
                if is_existing_rule(
                    rules, operation_rule_text
                ):  # If rule is matching ('in') existing rule, change it to AGREE.
                    rule_num = retrieve_rule_index(rules, operation_rule_text)
                    cleaned_operations[i] = (f"AGREE {rule_num+1}", rules[rule_num][0])
                elif (rule_num is None) or (
                    rule_num > len(rules)
                ):  # If rule doesn't exist, remove.
                    delete_indices.append(i)

            elif operation_type == "REMOVE" or operation_type == "AGREE":
                if not is_existing_rule(
                    rules, operation_rule_text
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
    rules: List[Tuple[str, int]],
    operations: List[Tuple[str, str]],
    is_full: bool = False,
) -> List[Tuple[str, int]]:
    updated_rules = rules.copy()

    for op in ["REMOVE", "AGREE", "EDIT", "ADD"]:  # Order is important
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(" ")[0]
            if operation_type != op:
                continue

            if operation_type == "REMOVE":  # remove rule: -1
                rule_index = retrieve_rule_index(
                    updated_rules, operation_rule_text
                )  # if rule_num doesn't match but text does
                remove_strength = 3 if is_full else 1
                updated_rules[rule_index] = (
                    updated_rules[rule_index][0],
                    updated_rules[rule_index][1] - remove_strength,
                )  # -1 (-3 if list full) to the counter
            elif operation_type == "AGREE":  # agree with rule: +1
                rule_index = retrieve_rule_index(
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
    ]  # remove rules when counter reach 0
    updated_rules.sort(key=lambda x: x[1], reverse=True)

    return updated_rules


def create_rules(
    llm: BaseChatModel,
    experiences: Dict[str, List],
    categories: Dict[str, int],
    train_idxs: List[int],
    rules: List[str],
    rules_with_count: List[Tuple[str, int]],
    max_num_rules: int,
    success_critique_num: int = 8,
) -> Tuple[List[str], List[Tuple[str, int]]]:
    # Intersect between train_idxs and each category (compare, success, fail).
    train_category_idxs = {
        category: list(set(train_idxs).intersection(set(category_idxs)))
        for category, category_idxs in categories.items()
    }

    # Compare.
    for train_idx in train_category_idxs["compare"]:
        question = experiences["questions"][train_idx]
        trajectory = experiences["trajectories"][train_idx]

        # Compare the successful trial with all previous failed trials.
        success_trial = trajectory[-1][-1]
        for failed_trial in trajectory[:-1]:
            # Prompt.
            out = _prompt_compare_critique(
                llm,
                rules,
                question,
                success_trial,
                failed_trial,
                max_num_rules < len(rules_with_count),
            )

            # Parse.
            operations = parse_rules(out)

            # Remove no-ops.
            operations = remove_err_operations(rules_with_count, operations)

            # Update rules_with_count and rules with comparison insights.
            rules_with_count = update_rules(
                rules_with_count,
                operations,
                is_full=max_num_rules + 5 <= len(rules_with_count),
            )
            rules = [rule[0] for rule in rules_with_count]

    # Success.
    batched_success_trajs_idxs = shuffle_chunk_list(
        train_category_idxs["success"], success_critique_num
    )
    for success_idxs in batched_success_trajs_idxs:
        # Concatenate batched successful trajectories.
        concat_success_trajs = [
            f"{experiences['questions'][idx]}\n{experiences['trajectories'][idx][0][-1]}"  # Get this successful trajectory's zero-th trial output.
            for idx in success_idxs
        ]
        success_trajs_str = "\n\n".join(concat_success_trajs)

        # Prompt.
        out = _prompt_all_success_critique(
            llm, rules, success_trajs_str, max_num_rules < len(rules_with_count)
        )

        # Parse.
        operations = parse_rules(out)

        # Remove no-ops.
        operations = remove_err_operations(rules_with_count, operations)

        # Update rules_with_count and rules with success insights.
        rules_with_count = update_rules(
            rules_with_count,
            operations,
            is_full=max_num_rules + 5 <= len(rules_with_count),
        )
        rules = [rule[0] for rule in rules_with_count]

    return rules, rules_with_count


# ============================================== Inference ==============================================
