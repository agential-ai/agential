"""LATS Agent strategies for QA."""

from typing import Any, Dict, List, Optional, Tuple

from langchain_community.docstore.wikipedia import Wikipedia

from agential.cog.lats.functional import (
    _build_failed_trajectory_format,
    _build_reflection_format,
    _prompt_agent,
    _prompt_value,
    get_node_trajectory_qa,
    parse_qa_action,
    parse_qa_value,
)
from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSReActStepOutput, LATSSimulationOutput
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.eval.em import EM
from agential.llm.llm import BaseLLM, ModelResponse
from agential.utils.docstore import DocstoreExplorer
from agential.utils.general import get_token_cost_time
from agential.utils.parse import remove_newline


class LATSQAStrategy(LATSBaseStrategy):
    """A strategy class for QA benchmarks using the LATS agent.

    Attributes:
        llm: The language model to be used for generating responses.
        docstore (DocstoreExplorer): Document store explorer, defaults to Wikipedia.
        n_samples (int): Number of samples to generate, default is 5.
        max_reflections (int): Maximum number of reflections allowed, default is 4.
        depth_limit (int): Maximum depth of the search tree, default is 7.
        max_unique (int): Maximum number of unique samples to consider, default is 5.
        cache_values (bool): Whether to cache values, default is True.

    The strategy uses these parameters to fine-tune its behavior and performance
    in question-answering tasks.
    """

    def __init__(
        self,
        llm: BaseLLM,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
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
        self.docstore = docstore

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
    ) -> List[Node]:
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
            List[Node]: A list of generated child nodes.
        """
        reflections_str = ""
        if self.reflect_condition():
            reflections = self.reflect(
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

        trajectory = get_node_trajectory_qa(node)

        unique_states = set()
        children_nodes = []
        for _ in range(self.n_samples):
            trajectory_i, thought, thought_model_response = self.generate_thought(
                question=question,
                examples=examples,
                trajectory=trajectory,
                reflections=reflections_str,
                depth=node.depth,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            trajectory_i, action_type, query, action_model_response = (
                self.generate_action(
                    question=question,
                    examples=examples,
                    trajectory=trajectory_i,
                    reflections=reflections_str,
                    depth=node.depth,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
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
                        answer="" if not done else query.lower().strip(),
                        external_tool_info=external_tool_info,
                        thought_metrics=get_token_cost_time(thought_model_response),
                        action_metrics=get_token_cost_time(action_model_response),
                    ),
                    parent=node,
                    depth=node.depth + 1,
                    is_terminal=reward == 1 or done,
                    reward=reward,
                )

                if new_node.is_terminal and reward == 0:
                    traversed_nodes = get_node_trajectory_qa(new_node)
                    self.failed_trajectories.append(
                        {
                            "trajectory": traversed_nodes,
                            "final_answer": query.lower().strip(),
                        }
                    )

                children_nodes.append(new_node)

        return children_nodes

    def generate_action(
        self,
        question: str,
        examples: str,
        trajectory: str,
        reflections: str,
        depth: int,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, ModelResponse]:
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
            Tuple[str, str, str, ModelResponse]: A tuple containing the updated trajectory, action type, query, and model response.
        """
        trajectory += f"\nAction {depth + 1}:"
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

        action = remove_newline(action).split("Observation")[0]
        trajectory += " " + action
        action_type, query = parse_qa_action(action)

        return trajectory, action_type, query, out

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
        external_tool_info = {"search_result": "", "lookup_result": ""}

        reward, done = 0, False
        trajectory += f"\nObservation {depth + 1}: "
        if action_type.lower() == "finish":
            if EM(query, key):
                obs = "Answer is CORRECT"
                reward = int(EM(query, key))
            else:
                obs = "Answer is INCORRECT"
            done = True
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)
            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
        trajectory += obs

        return trajectory, reward, obs, done, external_tool_info

    def evaluate_node(
        self,
        node: Node,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Evaluate the given node and its children.

        Args:
            node (Node): The node to be evaluated.
            question (str): The main question or task.
            examples (str): Examples for context in evaluation.
            prompt (str): The prompt template for evaluation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing evaluation results for each child node.
        """
        children_trajectories = [
            {"child_trajectory": get_node_trajectory_qa(child), "idx": idx}
            for idx, child in enumerate(node.children)
            if not child.is_terminal
        ]

        values = []
        child_trajectory_cache = {}
        for child_trajectory in children_trajectories:
            trajectory: str = child_trajectory["child_trajectory"]  # type: ignore
            idx: int = child_trajectory["idx"]  # type: ignore
            if trajectory in child_trajectory_cache:
                value = 0
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

                    value_str = value_str_out.choices[0].message.content

                    if self.cache_values:
                        self.value_cache[unique_key] = value_str

                explanation, value = parse_qa_value(value_str)  # type: ignore
                value = value / 10.0  # type: ignore
                node.children[idx].value = value

                child_trajectory_cache[trajectory] = value
            values.append({"node_idx": idx, "explanation": explanation, "value": value})

        return values

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
    ) -> Tuple[float, Node, List[Dict[str, Any]]]:
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
            Tuple[float, Node, List[Dict[str, Any]]]: A tuple containing:
                - The estimated value of the node (float)
                - The final node reached in the simulation (Node)
                - A list of dictionaries, representing the states of nodes explored during simulation
        """
        depth = node.depth
        rewards: List[int] = [0]
        results: List[Dict[str, Any]] = []
        while not node.is_terminal and depth < self.depth_limit:
            result = {
                "current_node": node,
                "children_nodes": [],
                "values": [],
            }

            values: List[Dict[str, Any]] = []
            children_nodes = self.generate_children_nodes(
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

            result["children_nodes"] = children_nodes

            for node in children_nodes:
                if node.is_terminal:
                    return node.reward, node, results

            for idx, child in enumerate(children_nodes):
                if not child.is_terminal:
                    child_trajectory = get_node_trajectory_qa(child)
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

                    explanation, value = parse_qa_value(value_str)  # type: ignore
                    values.append(
                        {"node_idx": idx, "explanation": explanation, "value": value}
                    )

            max_value = max(values, key=lambda x: x["value"])  # type: ignore
            max_value_index = values.index(max_value)
            rewards.append(max_value)  # type: ignore
            node = children_nodes[max_value_index]
            depth += 1

            if depth == self.depth_limit:
                rewards = [-1]

            result["best_child_node"] = node
            result["values"] = values

            results.append(result)

        return sum(rewards) / len(rewards), node, results


class LATSHotQAStrategy(LATSQAStrategy):
    """A strategy class for the HotpotQA benchmark using the LATS agent."""

    pass


class LATSTriviaQAStrategy(LATSQAStrategy):
    """A strategy class for the TriviaQA benchmark using the LATS agent."""

    pass


class LATSAmbigNQStrategy(LATSQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the LATS agent."""

    pass


class LATSFEVERStrategy(LATSQAStrategy):
    """A strategy class for the FEVER benchmark using the LATS agent."""

    pass
