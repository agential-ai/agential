"""Functional module for Language Agent Tree Search (LATS)."""

import re

from typing import Any, Dict, List, Tuple, Union

from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSStepOutput
from agential.cog.lats.prompts import (
    LATS_FAILED_TRAJECTORY_FORMAT,
    LATS_REFLECTION_FORMAT,
)
from agential.llm.llm import BaseLLM, Response


def _build_reflection_format(trajectory: str, reflection: str) -> str:
    """Builds a formatted string for LATS reflection.

    This function takes a trajectory and a reflection as input and formats them
    according to the LATS_REFLECTION_FORMAT template.

    Args:
        trajectory (str): The trajectory string to be included in the format.
        reflection (str): The reflection string to be included in the format.

    Returns:
        str: A formatted string combining the trajectory and reflection using
             the LATS_REFLECTION_FORMAT template.
    """
    return LATS_REFLECTION_FORMAT.format(trajectory=trajectory, reflection=reflection)


def _build_failed_trajectory_format(
    question: str, trajectory: str, reflection: str
) -> str:
    """Builds a formatted string for a failed LATS trajectory.

    This function takes a question, trajectory, and reflection as input and formats them
    according to the LATS_FAILED_TRAJECTORY_FORMAT template.

    Args:
        question (str): The question that led to the failed trajectory.
        trajectory (str): The failed trajectory string.
        reflection (str): The reflection on the failed trajectory.

    Returns:
        str: A formatted string combining the question, trajectory, and reflection using
             the LATS_FAILED_TRAJECTORY_FORMAT template.
    """
    return LATS_FAILED_TRAJECTORY_FORMAT.format(
        question=question, trajectory=trajectory, reflection=reflection
    )


def _build_reflection_prompt(
    question: str,
    examples: str,
    trajectory: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a reflection prompt for the Language Agent Tree Search (LATS) agent.

    Args:
        question (str): The main question or task for the agent to reflect on.
        examples (str): Relevant examples to provide context for the reflection.
        trajectory (str): The agent's current trajectory or thought process.
        prompt (str): The base prompt template to be formatted.
        additional_keys (Dict[str, str], optional): Additional key-value pairs for formatting the prompt. Defaults to {}.

    Returns:
        str: The fully formatted reflection prompt ready for use with the language model.
    """
    prompt = prompt.format(
        question=question, examples=examples, trajectory=trajectory, **additional_keys
    )
    return prompt


def _prompt_reflection(
    llm: BaseLLM,
    question: str,
    examples: str,
    trajectory: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a reflection using the language model based on the given inputs.

    Args:
        llm (BaseLLM): The language model to use for generating the reflection.
        question (str): The main question or task.
        examples (str): Relevant examples to provide context.
        trajectory (str): The agent's current trajectory or thought process.
        prompt (str): The base prompt template to be used.
        additional_keys (Dict[str, str], optional): Additional formatting keys. Defaults to {}.

    Returns:
        Response: The generated reflection content.
    """
    prompt = _build_reflection_prompt(
        question=question,
        examples=examples,
        trajectory=trajectory,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


def _build_value_prompt(
    question: str,
    examples: str,
    trajectory: str,
    failed_trajectories: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a value prompt for the LATS agent.

    Args:
        question (str): The main question or task.
        examples (str): Relevant examples to provide context.
        trajectory (str): The agent's current trajectory.
        failed_trajectories (str): Previously failed trajectories.
        prompt (str): The base prompt template to be formatted.
        additional_keys (Dict[str, str], optional): Additional formatting keys. Defaults to {}.

    Returns:
        str: The fully formatted value prompt.
    """
    prompt = prompt.format(
        question=question,
        examples=examples,
        trajectory=trajectory,
        failed_trajectories=failed_trajectories,
        **additional_keys,
    )
    return prompt


def _prompt_value(
    llm: BaseLLM,
    question: str,
    examples: str,
    trajectory: str,
    failed_trajectories: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a value assessment using the language model based on the given inputs.

    Args:
        llm (BaseLLM): The language model to use for generating the value assessment.
        question (str): The main question or task.
        examples (str): Relevant examples to provide context.
        trajectory (str): The agent's current trajectory.
        failed_trajectories (str): Previously failed trajectories.
        prompt (str): The base prompt template to be used.
        additional_keys (Dict[str, str], optional): Additional formatting keys. Defaults to {}.

    Returns:
        Response: The generated value assessment content.
    """
    prompt = _build_value_prompt(
        question=question,
        examples=examples,
        trajectory=trajectory,
        failed_trajectories=failed_trajectories,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


def _build_agent_prompt(
    question: str,
    examples: str,
    trajectory: str,
    reflections: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs an agent prompt for the LATS agent.

    Args:
        question (str): The main question or task.
        examples (str): Relevant examples to provide context.
        trajectory (str): The agent's current trajectory.
        reflections (str): Previous reflections made by the agent.
        prompt (str): The base prompt template to be formatted.
        additional_keys (Dict[str, str], optional): Additional formatting keys. Defaults to {}.

    Returns:
        str: The fully formatted agent prompt.
    """
    prompt = prompt.format(
        question=question,
        examples=examples,
        trajectory=trajectory,
        reflections=reflections,
        **additional_keys,
    )
    return prompt


def _prompt_agent(
    llm: BaseLLM,
    question: str,
    examples: str,
    trajectory: str,
    reflections: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates an agent response using the language model based on the given inputs.

    Args:
        llm (BaseLLM): The language model to use for generating the agent response.
        question (str): The main question or task.
        examples (str): Relevant examples to provide context.
        trajectory (str): The agent's current trajectory.
        reflections (str): Previous reflections made by the agent.
        prompt (str): The base prompt template to be used.
        additional_keys (Dict[str, str], optional): Additional formatting keys. Defaults to {}.

    Returns:
        Response: The generated agent response content.
    """
    prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        trajectory=trajectory,
        reflections=reflections,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


def get_unique_trajectories(
    failed_trajectories: List[Dict[str, str]], max_unique: int
) -> List[str]:
    """Extracts a specified number of unique trajectories from the given failed trajectories.

    Args:
        failed_trajectories (List[Dict[str, str]]): A list of dictionaries containing failed trajectories.
        max_unique (int): The maximum number of unique trajectories to return.

    Returns:
        List[str]: A list of unique trajectory strings, up to the specified number.
    """
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj["final_answer"]
        if final_answer not in seen_final_answers:
            unique_trajectories.append(traj["trajectory"])
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= max_unique:
            break

    return unique_trajectories


def get_node_trajectory(node: Node) -> str:
    """Generates a string representation of the trajectory from the given node to the root.

    Args:
        node (Node): The current node in the tree.

    Returns:
        str: A string representation of the trajectory, including thoughts, actions, and observations.
    """
    trajectory = []

    while node:
        step = []
        if node.depth > 0:
            if node.state.thought:
                step.append(f"Thought {node.depth}: {node.state.thought}")
            if node.state.action_type and node.state.query:
                step.append(
                    f"Action {node.depth}: {node.state.action_type}[{node.state.query}]"
                )
            if node.state.observation:
                step.append(f"Observation {node.depth}: {node.state.observation}")
        step_str = "\n".join(step)
        trajectory.append(step_str)
        node = node.parent  # type: ignore

    return "\n".join(reversed(trajectory))


def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
    else:
        action_type = ""
        argument = ""
    return action_type, argument


def parse_value(string: str) -> Tuple[str, float]:
    """Extracts the explanation and correctness score from a given string.

    Args:
        string (str): The input string containing an explanation and correctness score.

    Returns:
        Tuple[str, float]: A tuple containing the explanation (str) and the correctness score (float).
        If parsing fails, returns ("Explanation not found", 0.0).
    """
    try:
        explanation_part = string.split("Explanation:")[1].strip()
        explanation, score_part = explanation_part.split("Correctness score:")
        score = float(int(score_part.strip()))
        return explanation.strip(), score
    except Exception:
        return "Explanation not found", 0.0


def parse_math_action(action: str) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`, `Calculate`) and extracts the
    corresponding code content enclosed within Markdown-style code blocks.
    The action type is case-insensitive and the code content is trimmed of
    leading and trailing whitespace.

    Args:
        action (str): The action string containing the action type and code content.

    Returns:
        Tuple[str, str]: A tuple containing the extracted action type (capitalized)
        and the extracted code content.
    """
    action_split = action.split("```python", maxsplit=1)
    match = re.search(r"\b(Finish|Calculate)\b", action_split[0], re.IGNORECASE)

    action_type = match.group(0).lower().capitalize() if match else ""
    try:
        query = action_split[1].split("```")[0].strip() if action_type else ""
    except:
        action_type = ""
        query = ""

    return action_type, query


def parse_latest_implement(text: str) -> str:
    """Extract the latest Python code implementation from the given text.

    This function searches for the last occurrence of Python code enclosed in
    'Implement[```python ... ```]' blocks within the input text.

    Args:
        text (str): The input text containing one or more code implementations.

    Returns:
        str: The extracted Python code as a string if found, or "" if no implementation is found.
    """
    pattern = re.compile(r"Implement\[\s*```python(.*?)```", re.DOTALL)

    matches = pattern.findall(text)

    if matches:
        latest_implement = matches[-1].strip()
        return latest_implement
    return ""


def parse_code_action(action: str) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`, `Calculate`) and extracts the
    corresponding code content enclosed within Markdown-style code blocks.
    The action type is case-insensitive and the code content is trimmed of
    leading and trailing whitespace.

    Args:
        action (str): The action string containing the action type and code content.

    Returns:
        Tuple[str, str]: A tuple containing the extracted action type (capitalized)
        and the extracted code content.
    """
    action_split = action.split("```python", maxsplit=1)
    match = re.search(r"\b(Finish|Test|Implement)\b", action_split[0], re.IGNORECASE)

    action_type = match.group(0).lower().capitalize() if match else ""
    try:
        query = action_split[1].split("```")[0].strip() if action_type else ""
    except:
        action_type = ""
        query = ""

    return action_type, query


def _accumulate_metric(step: LATSStepOutput, metric_type: str) -> Union[int, float]:
    """Accumulate total metrics from a list of LATSStepOutput objects.

    Args:
        step (LATSStepOutput): The LATSStepOutput object containing metrics.
        metric_type (str): The type of metric to accumulate.

    Returns:
        Union[int, float]: The accumulated metric value.
    """
    out = 0

    out += sum(
        [
            getattr(thought_response, metric_type)
            for thought_response in step.generate_response.thoughts_response
        ]
    )
    out += sum(
        [
            getattr(action_response, metric_type)
            for action_response in step.generate_response.actions_response
        ]
    )
    out += sum(
        [
            getattr(reflection_response, metric_type)
            for reflection_response in step.generate_response.reflections_response
        ]
    )

    if step.evaluate_response:
        for value_response in step.evaluate_response.values_response:
            if value_response:
                out += getattr(value_response, metric_type)

    if step.simulation_response:
        for sim_step_response in step.simulation_response.simulation_step_response:
            # generate_response.
            out += sum(
                [
                    getattr(thought_response, metric_type)
                    for thought_response in sim_step_response.generate_response.thoughts_response
                ]
            )
            out += sum(
                [
                    getattr(action_response, metric_type)
                    for action_response in sim_step_response.generate_response.actions_response
                ]
            )
            out += sum(
                [
                    getattr(reflection_response, metric_type)
                    for reflection_response in sim_step_response.generate_response.reflections_response
                ]
            )

            # evaluate_response.
            out += sum(
                [
                    getattr(value_response, metric_type)
                    for value_response in sim_step_response.evaluate_response.values_response
                    if value_response
                ]
            )

    return out


def accumulate_metrics(steps: List[LATSStepOutput]) -> Dict[str, Any]:
    """Accumulate total metrics from a list of LATSStepOutput objects.

    This function calculates and aggregates various metrics across all steps in the input list.
    It sums up token counts, costs, and time measurements for both thought and action components.

    Args:
        steps (List[LATSStepOutput]): A list of LATSStepOutput objects representing individual steps.

    Returns:
        Dict[str, Any]: A dictionary containing the following accumulated metrics:
            - total_prompt_tokens (int): Total number of prompt tokens used.
            - total_completion_tokens (int): Total number of completion tokens generated.
            - total_tokens (int): Total number of tokens (prompt + completion).
            - total_prompt_cost (float): Total cost associated with prompts.
            - total_completion_cost (float): Total cost associated with completions.
            - total_cost (float): Total overall cost (prompt + completion).
            - total_prompt_time (float): Total time spent on prompts.
    """
    total_prompt_tokens = 0.0
    total_completion_tokens = 0.0
    total_tokens = 0.0
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    total_cost = 0.0
    total_prompt_time = 0.0

    for step in steps:
        total_prompt_tokens += _accumulate_metric(step, "prompt_tokens")
        total_completion_tokens += _accumulate_metric(step, "completion_tokens")
        total_tokens += _accumulate_metric(step, "total_tokens")
        total_prompt_cost += _accumulate_metric(step, "prompt_cost")
        total_completion_cost += _accumulate_metric(step, "completion_cost")
        total_cost += _accumulate_metric(step, "total_cost")
        total_prompt_time += _accumulate_metric(step, "prompt_time")

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_prompt_cost": total_prompt_cost,
        "total_completion_cost": total_completion_cost,
        "total_cost": total_cost,
        "total_prompt_time": total_prompt_time,
    }
