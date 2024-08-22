"""LATS Agent strategies for Math."""

from typing import Any, Dict, List, Optional, Tuple

from agential.cog.lats.functional import (
    _build_failed_trajectory_format,
    _build_reflection_format,
    _prompt_agent,
    _prompt_value,
    get_node_trajectory_math,
    parse_math_action,
    parse_value,
)
from agential.cog.lats.node import Node
from agential.cog.lats.output import (
    LATSEvaluateMetrics,
    LATSGenerateMetrics,
    LATSReActStepOutput,
    LATSSimulationMetrics,
    LATSSimulationStepMetrics,
)
from agential.cog.lats.strategies.general import LATSGeneralStrategy
from agential.eval.em import EM
from agential.llm.llm import BaseLLM
from agential.utils.general import safe_execute
from agential.utils.metrics import PromptInfo, get_token_cost_time


class LATSMathStrategy(LATSGeneralStrategy):
    """A strategy class for Math benchmarks using the LATS agent.

    Attributes:
        llm: The language model to be used for generating responses.
        n_samples (int): Number of samples to generate, default is 5.
        max_reflections (int): Maximum number of reflections allowed, default is 4.
        depth_limit (int): Maximum depth of the search tree, default is 7.
        max_unique (int): Maximum number of unique samples to consider, default is 5.
        cache_values (bool): Whether to cache values, default is True.

    The strategy uses these parameters to fine-tune its behavior and performance
    in math reasoning tasks.
    """

    def __init__(
        self,
        llm: BaseLLM,
        n_samples: int = 5,
        max_reflections: int = 4,
        depth_limit: int = 7,
        max_unique: int = 5,
        cache_values: bool = True,
        testing: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__(
            llm=llm,
            n_samples=n_samples,
            max_reflections=max_reflections,
            depth_limit=depth_limit,
            max_unique=max_unique,
            cache_values=cache_values,
            testing=testing,
        )

        self.failed_trajectories: List[Dict[str, str]] = []
        self.reflection_map: List[Dict[str, str]] = []
        self.value_cache: Dict[str, str] = {}
        self.root: Optional[Node] = None

    def generate_children_nodes(
        self,
        node: Node,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        prompt: str,
        reflect_prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
    ) -> Tuple[List[Node], LATSGenerateMetrics]:
        """Generate child nodes for the given node.

        Args:
            node (Node): The current node to expand.
            question (str): The main question or task.
            key (str): The answer key for evaluation.
            examples (str): Examples for context.
            reflect_examples (str): Examples for reflection.
            prompt (str): The prompt template for generation.
            reflect_prompt (str): The prompt template for reflection.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.
            reflect_additional_keys (Dict[str, str]): Additional keys for reflection prompt formatting.

        Returns:
            Tuple[List[Node], LATSGenerateMetrics]: A list of generated child nodes, and the pydantic of corresponding metrics.
        """
        reflections_str = ""
        reflection_metrics: List[PromptInfo] = []
        if self.reflect_condition():
            reflections, reflection_metrics = self.reflect(
                question=question,
                examples=reflect_examples,
                prompt=reflect_prompt,
                additional_keys=reflect_additional_keys,
            )
            for reflection in reflections:
                reflections_str += (
                    _build_reflection_format(
                        trajectory=reflection["trajectory"],
                        reflection=reflection["reflection"],
                    )
                    + "\n\n"
                )

        trajectory = get_node_trajectory_math(node)

        unique_states = set()
        children_nodes, thoughts_metrics, actions_metrics = [], [], []
        for _ in range(self.n_samples):
            trajectory_i, thought, thought_metrics = self.generate_thought(
                question=question,
                examples=examples,
                trajectory=trajectory,
                reflections=reflections_str,
                depth=node.depth,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            trajectory_i, action_type, query, action_metrics = self.generate_action(
                question=question,
                examples=examples,
                trajectory=trajectory_i,
                reflections=reflections_str,
                depth=node.depth,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            unique_key = f"{thought}::{action_type}::{query}"
            if unique_key not in unique_states:
                unique_states.add(unique_key)

                _, reward, obs, done, external_tool_info = self.generate_observation(
                    key=key,
                    action_type=action_type,
                    query=query,
                    trajectory=trajectory_i,
                    depth=node.depth,
                )

                new_node = Node(
                    state=LATSReActStepOutput(
                        thought=thought,
                        action_type=action_type,
                        query=query,
                        observation=obs,
                        answer="" if not done else query,
                        external_tool_info=external_tool_info,
                    ),
                    parent=node,
                    depth=node.depth + 1,
                    is_terminal=reward == 1 or done,
                    reward=reward,
                )

                if new_node.is_terminal and reward == 0:
                    traversed_nodes = get_node_trajectory_math(new_node)
                    self.failed_trajectories.append(
                        {
                            "trajectory": traversed_nodes,
                            "final_answer": query,
                        }
                    )
            else:
                new_node = Node(
                    state=LATSReActStepOutput(
                        thought=thought,
                        action_type=action_type,
                        query=query,
                        observation="",
                        answer="",
                        external_tool_info={},
                    ),
                )

            thoughts_metrics.append(thought_metrics)
            actions_metrics.append(action_metrics)
            children_nodes.append(new_node)

        metrics = LATSGenerateMetrics(
            thoughts_metrics=thoughts_metrics,
            actions_metrics=actions_metrics,
            reflections_metrics=reflection_metrics,
        )

        return children_nodes, metrics

    def generate_action(
        self,
        question: str,
        examples: str,
        trajectory: str,
        reflections: str,
        depth: int,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, PromptInfo]:
        """Generate an action for the current step in the reasoning process.

        Args:
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for action generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            reflections (str): Previous reflections to guide the action generation.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, str, PromptInfo]: A tuple containing the updated trajectory, action type, query, and the metrics.
        """
        trajectory += f"\nAction {depth + 1}: "
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.choices[0].message.content

        action = action.split("Observation")[0].strip()
        action_type, query = parse_math_action(action)
        trajectory += f" {action_type}[\n```python\n{query}\n```\n]"

        return trajectory, action_type, query, get_token_cost_time(out)

    def generate_observation(
        self,
        key: str,
        action_type: str,
        query: str,
        trajectory: str,
        depth: int,
    ) -> Tuple[str, int, str, bool, Dict[str, Any]]:
        """Generate an observation based on the current action.

        Args:
            key (str): The answer key for evaluation.
            action_type (str): The type of action taken.
            query (str): The query associated with the action.
            trajectory (str): The current trajectory or history of thoughts and actions.
            depth (int): The current depth in the search tree.

        Returns:
            Tuple[str, int, str, bool, Dict[str, str]]: A tuple containing the updated trajectory,
            reward, observation, done flag, and external tool information.
        """
        external_tool_info = {"execution_status": "", "code_answer": ""}
        code_answer, execution_status = safe_execute(query)

        reward, done = 0, False
        trajectory += f"\nObservation {depth + 1}: "
        if action_type.lower() == "finish":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            if EM(code_answer[0], key, normalize=False):
                obs = "Answer is CORRECT"
                reward = int(EM(code_answer[0], key, normalize=False))
            else:
                obs = "Answer is INCORRECT"
            done = True
        elif action_type.lower() == "calculate":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            obs = f"\n```python\n{query}\n```\nExecution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
        else:
            obs = (
                "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
            )
        trajectory += obs

        return trajectory, reward, obs, done, external_tool_info

    def evaluate_node(
        self,
        node: Node,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], LATSEvaluateMetrics]:
        """Evaluate the given node and its children.

        Args:
            node (Node): The node to be evaluated.
            question (str): The main question or task.
            examples (str): Examples for context in evaluation.
            prompt (str): The prompt template for evaluation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[List[Dict[str, Any]], LATSEvaluateMetrics]: A list of dictionaries containing evaluation results for each child node and their metrics.
        """
        values, values_metrics = [], []
        child_trajectory_cache = {}
        for idx, child in enumerate(node.children):
            if not child.is_terminal:
                trajectory = get_node_trajectory_math(child)
                if trajectory in child_trajectory_cache:
                    value = 0
                    explanation = ""
                    value_response = None
                else:
                    failed_trajectories = ""
                    if len(self.reflection_map) > 0:
                        for trajectory_reflection in self.reflection_map:
                            failed_trajectories += (
                                _build_failed_trajectory_format(
                                    question=question,
                                    trajectory=trajectory_reflection["trajectory"],
                                    reflection=trajectory_reflection["reflection"],
                                )
                                + "\n\n"
                            )
                        failed_trajectories = failed_trajectories.rstrip("\n\n")

                    unique_key = f"{trajectory}::{failed_trajectories}"
                    if self.cache_values and unique_key in self.value_cache:
                        value_str = self.value_cache[unique_key]
                        value_response = None
                    else:
                        value_str_out = _prompt_value(
                            llm=self.llm,
                            question=question,
                            examples=examples,
                            trajectory=trajectory,
                            failed_trajectories=failed_trajectories,
                            prompt=prompt,
                            additional_keys=additional_keys,
                        )
                        value_response = value_str_out
                        value_str = value_str_out.choices[0].message.content

                        if self.cache_values:
                            self.value_cache[unique_key] = value_str

                    explanation, value = parse_value(value_str)  # type: ignore
                    value = value / 10.0  # type: ignore
                    node.children[idx].value = value

                    child_trajectory_cache[trajectory] = value

                values_metrics.append(
                    get_token_cost_time(value_response) if value_response else None
                )
                values.append({"explanation": explanation, "value": value})
            else:
                values_metrics.append(None)
                values.append({"explanation": "", "value": -1e10})

        return values, LATSEvaluateMetrics(values_metrics=values_metrics)

    def simulate_node(
        self,
        node: Node,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        value_examples: str,
        prompt: str,
        reflect_prompt: str,
        value_prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        value_additional_keys: Dict[str, str],
    ) -> Tuple[
        float,
        Node,
        List[Node],
        List[List[Node]],
        List[List[Dict[str, Any]]],
        LATSSimulationMetrics,
    ]:
        """Simulate the node to estimate its value and collect information about the simulation process.

        Args:
            node (Node): The node to simulate.
            question (str): The main question or task.
            key (str): The answer key for evaluation.
            examples (str): Examples for context in simulation.
            reflect_examples (str): Examples for reflection during simulation.
            value_examples (str): Examples for value estimation.
            prompt (str): The prompt template for simulation.
            reflect_prompt (str): The prompt template for reflection during simulation.
            value_prompt (str): The prompt template for value estimation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.
            reflect_additional_keys (Dict[str, str]): Additional keys for reflection prompt formatting.
            value_additional_keys (Dict[str, str]): Additional keys for value estimation prompt formatting.

        Returns:
            Tuple[float, Node, List[Node], List[List[Node]], List[List[Dict[str, Any]]], LATSSimulationMetrics]:
                - The estimated value of the node
                - The simulation's terminal node
                - Each simulation iteration's children nodes
                - Each simulation iteration's children nodes' values
                - Metrics for the simulation process
        """
        depth = node.depth
        rewards: List[int] = [0]

        simulation_current_nodes: List[Node] = []
        simulation_children_nodes: List[List[Node]] = []
        simulation_values: List[List[Dict[str, Any]]] = []
        simulation_step_metrics: List[LATSSimulationStepMetrics] = []
        while not node.is_terminal and depth < self.depth_limit:
            simulation_current_nodes.append(node)

            values: List[Dict[str, Any]] = []
            values_metrics: List[Optional[PromptInfo]] = []

            children_nodes, generate_metrics = self.generate_children_nodes(
                node=node,
                question=question,
                key=key,
                examples=examples,
                reflect_examples=reflect_examples,
                prompt=prompt,
                reflect_prompt=reflect_prompt,
                additional_keys=additional_keys,
                reflect_additional_keys=reflect_additional_keys,
            )
            simulation_children_nodes.append(children_nodes)

            for node in children_nodes:
                if node.is_terminal and node.parent:
                    simulation_step_metrics.append(
                        LATSSimulationStepMetrics(
                            generate_metrics=generate_metrics,
                            evaluate_metrics=LATSEvaluateMetrics(
                                values_metrics=values_metrics
                            ),
                        )
                    )

                    simulation_metrics = LATSSimulationMetrics(
                        simulation_step_metrics=simulation_step_metrics
                    )

                    return (
                        node.reward,
                        node,
                        simulation_current_nodes,
                        simulation_children_nodes,
                        simulation_values,
                        simulation_metrics,
                    )

            for child in children_nodes:
                if not child.is_terminal and node.parent:
                    child_trajectory = get_node_trajectory_math(child)
                    failed_trajectories = ""
                    if len(self.reflection_map) > 0:
                        for trajectory_reflection in self.reflection_map:
                            failed_trajectories += (
                                _build_failed_trajectory_format(
                                    question=question,
                                    trajectory=trajectory_reflection["trajectory"],
                                    reflection=trajectory_reflection["reflection"],
                                )
                                + "\n\n"
                            )
                        failed_trajectories = failed_trajectories.rstrip("\n\n")

                    value_str_out = _prompt_value(
                        llm=self.llm,
                        question=question,
                        examples=value_examples,
                        trajectory=child_trajectory,
                        failed_trajectories=failed_trajectories,
                        prompt=value_prompt,
                        additional_keys=value_additional_keys,
                    )

                    value_str = value_str_out.choices[0].message.content

                    explanation, value = parse_value(value_str)  # type: ignore
                    values_metrics.append(get_token_cost_time(value_str_out))
                    values.append({"explanation": explanation, "value": value})
                else:
                    values_metrics.append(None)
                    values.append({"explanation": "", "value": -1e10})

            simulation_values.append(values)
            max_value = max(values, key=lambda x: x["value"])  # type: ignore
            max_value_index = values.index(max_value)
            rewards.append(max_value)  # type: ignore
            node = children_nodes[max_value_index]
            depth += 1

            if depth == self.depth_limit:
                rewards = [-1]

            simulation_step_metrics.append(
                LATSSimulationStepMetrics(
                    generate_metrics=generate_metrics,
                    evaluate_metrics=LATSEvaluateMetrics(values_metrics=values_metrics),
                )
            )

        simulation_metrics = LATSSimulationMetrics(
            simulation_step_metrics=simulation_step_metrics
        )

        return (
            sum(rewards) / len(rewards),
            node,
            simulation_current_nodes,
            simulation_children_nodes,
            simulation_values,
            simulation_metrics,
        )


class LATSGSM8KStrategy(LATSMathStrategy):
    """A strategy class for the GSM8K benchmark using the LATS agent."""

    pass


class LATSSVAMPStrategy(LATSMathStrategy):
    """A strategy class for the SVAMP benchmark using the LATS agent."""

    pass


class LATSTabMWPStrategy(LATSMathStrategy):
    """A strategy class for the TabMWP benchmark using the LATS agent."""

    pass
