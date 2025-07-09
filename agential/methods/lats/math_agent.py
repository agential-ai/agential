"""
LATS Math Agent for mathematical reasoning benchmarks (proper LATS implementation).
"""

from typing import Dict, Any, List, Optional, Tuple
import time
from agential.core.llm import BaseLLM, Response
from agential.methods.base import BaseMethod
from agential.methods.lats.prompts import *
from agential.methods.lats.utils import (
    parse_value,
)
from agential.methods.lats.node import Node
from agential.methods.lats.utils import (
    _build_reflection_format,
    _build_failed_trajectory_format,
    _prompt_value,
    get_node_trajectory,
    parse_math_action,
    log_llm_io,
    clean_llm_output,
)
from agential.eval.classification import EM
from agential.utils.general import safe_execute
from agential.utils.parse import remove_newline


class LATSMath(BaseMethod):
    """LATS Math Agent that implements proper tree search with UCT selection."""

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        n_samples: int = 5,
        max_reflections: int = 4,
        max_iterations: int = 30,
        depth_limit: int = 7,
        max_unique: int = 5,
        cache_values: bool = True,
        truncate_length: int = -1,
        verbose: bool = False,
        config: dict = {},
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.n_samples = n_samples
        self.max_reflections = max_reflections
        self.max_iterations = max_iterations
        self.depth_limit = depth_limit
        self.max_unique = max_unique
        self.cache_values = cache_values
        self.truncate_length = truncate_length
        self.verbose = verbose

        # State for tree search
        self.failed_trajectories: List[Dict[str, str]] = []
        self.reflection_map: List[Dict[str, str]] = []
        self.value_cache: Dict[str, str] = {}
        self.root: Optional[Node] = None

    def generate(
        self,
        question: str,
        key: str = "",
        additional_keys: dict = {},
        reflect_additional_keys: dict = {},
        value_additional_keys: dict = {},
        max_llm_retries: int = 3,
        prompt: Optional[str] = None,
        fewshot: Optional[str] = None,
        reflect_prompt: Optional[str] = None,
        reflect_fewshot: Optional[str] = None,
        value_prompt: Optional[str] = None,
        value_fewshot: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        scratchpad, answer, steps, step_metrics = "", "", [], []
        all_responses = []  # Collect all responses for token/cost tracking

        # Use provided parameters or fall back to config
        prompt = prompt or self.config["prompt"]
        fewshot = fewshot or self.config["fewshot"]
        reflect_prompt = reflect_prompt or self.config["reflect_prompt"]
        reflect_fewshot = reflect_fewshot or self.config["reflect_fewshot"]
        value_prompt = value_prompt or self.config["value_prompt"]
        value_fewshot = value_fewshot or self.config["value_fewshot"]

        # Initialize root node
        self.root = Node(
            state={
                "thought": "",
                "action_type": "",
                "query": "",
                "observation": "",
                "answer": "",
                "external_tool_info": {},
            },
            depth=0,
            is_terminal=False,
            reward=0,
        )

        # Main LATS tree search loop
        iteration = 0
        terminal_node = None

        while iteration < self.max_iterations:
            iteration += 1
            step_start = time.time()

            # Step 1: Select node using UCT
            selected_node = self._select_node(self.root)

            # Step 2: Expand node
            children_nodes, generate_metrics = self._expand_node(
                node=selected_node,
                question=question,
                key=key,
                examples=fewshot,
                reflect_examples=reflect_fewshot,
                reflect_prompt=reflect_prompt,
                prompt=prompt,
                additional_keys=additional_keys,
                reflect_additional_keys=reflect_additional_keys,
            )

            # Collect responses from generation
            for response_list in [
                generate_metrics.get("thoughts_response", []),
                generate_metrics.get("actions_response", []),
                generate_metrics.get("reflections_response", []),
            ]:
                all_responses.extend([r for r in response_list if r])

            # Check if any child is terminal with reward 1
            terminal_children = [
                child
                for child in children_nodes
                if child.is_terminal and child.reward == 1
            ]
            if terminal_children:
                terminal_node = terminal_children[0]
                break

            # Step 3: Evaluate children if not terminal
            if children_nodes and not any(
                child.is_terminal for child in children_nodes
            ):
                values, evaluate_metrics = self._evaluate_node(
                    node=selected_node,
                    question=question,
                    examples=value_fewshot,
                    prompt=value_prompt,
                    additional_keys=value_additional_keys,
                    context="Node Evaluation",
                )

                # Collect responses from evaluation
                for response in evaluate_metrics.get("values_response", []):
                    if response:
                        all_responses.append(response)

                # Step 4: Simulate from best child
                if values:
                    best_child_idx = max(
                        range(len(values)), key=lambda i: values[i]["value"]
                    )
                    best_child = children_nodes[best_child_idx]

                    simulation_reward, simulation_terminal, simulation_responses = (
                        self._simulate_node(
                            node=best_child,
                            question=question,
                            key=key,
                            examples=fewshot,
                            reflect_examples=reflect_fewshot,
                            value_examples=value_fewshot,
                            prompt=prompt,
                            reflect_prompt=reflect_prompt,
                            value_prompt=value_prompt,
                            additional_keys=additional_keys,
                            reflect_additional_keys=reflect_additional_keys,
                            value_additional_keys=value_additional_keys,
                        )
                    )

                    # Collect responses from simulation
                    all_responses.extend(simulation_responses)

                    # Step 5: Backpropagate
                    self._backpropagate_node(simulation_terminal, simulation_reward)

                    # Check if simulation reached terminal
                    if (
                        simulation_terminal.is_terminal
                        and simulation_terminal.reward == 1
                    ):
                        terminal_node = simulation_terminal
                        break
                else:
                    # If no values, just pick the first child for simulation
                    if children_nodes:
                        simulation_reward, simulation_terminal, simulation_responses = (
                            self._simulate_node(
                                node=children_nodes[0],
                                question=question,
                                key=key,
                                examples=fewshot,
                                reflect_examples=reflect_fewshot,
                                value_examples=value_fewshot,
                                prompt=prompt,
                                reflect_prompt=reflect_prompt,
                                value_prompt=value_prompt,
                                additional_keys=additional_keys,
                                reflect_additional_keys=reflect_additional_keys,
                                value_additional_keys=value_additional_keys,
                            )
                        )

                        # Collect responses from simulation
                        all_responses.extend(simulation_responses)

                        # Backpropagate
                        self._backpropagate_node(simulation_terminal, simulation_reward)

                        # Check if simulation reached terminal
                        if (
                            simulation_terminal.is_terminal
                            and simulation_terminal.reward == 1
                        ):
                            terminal_node = simulation_terminal
                            break
            else:
                # If any child is terminal, select the first terminal one
                terminal_children = [
                    child for child in children_nodes if child.is_terminal
                ]
                if terminal_children:
                    terminal_node = terminal_children[0]
                    break

            # Update step metrics
            step_time = time.time() - step_start
            step_metrics.append(
                {
                    "step": iteration,
                    "total_step_time": step_time,
                }
            )

        # Calculate total tokens and cost from all responses
        total_tokens = sum(getattr(r, "total_tokens", 0) for r in all_responses)
        total_cost = sum(getattr(r, "total_cost", 0) for r in all_responses)

        # Extract final answer and trajectory
        if terminal_node and terminal_node.state:
            answer = terminal_node.state.get("answer", "")
            scratchpad = get_node_trajectory(terminal_node) if terminal_node else ""
        elif self.root:
            # If no terminal node found, use the best child of root
            if self.root.children:
                best_child = max(self.root.children, key=lambda c: c.value)
                answer = best_child.state.get("answer", "")
                scratchpad = get_node_trajectory(best_child) if best_child else ""

        # Build steps from trajectory
        if scratchpad:
            lines = scratchpad.split("\n")
            current_step = {}
            for line in lines:
                if line.startswith("Thought"):
                    if current_step:
                        steps.append(current_step)
                    current_step = {
                        "thought": line.split(":", 1)[1].strip() if ":" in line else ""
                    }
                elif line.startswith("Action"):
                    if ":" in line:
                        action_part = line.split(":", 1)[1].strip()
                        action_type, query = parse_math_action(action_part)
                        current_step["action_type"] = action_type
                        current_step["query"] = query
                elif line.startswith("Observation"):
                    if ":" in line:
                        current_step["observation"] = line.split(":", 1)[1].strip()
                        current_step["answer"] = current_step.get("query", "")
            if current_step:
                steps.append(current_step)

        total_time = time.time() - start_time

        return {
            "answer": answer,
            "steps": steps,
            "scratchpad": scratchpad,
            "reflections": self._get_reflections_string(),
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "step_metrics": step_metrics,
            },
        }

    def _select_node(self, node: Node) -> Node:
        """Select the most promising node using UCT."""
        while node and node.children:
            # Filter out terminal children
            non_terminal_children = [
                child for child in node.children if not child.is_terminal
            ]

            # If all children are terminal, move up to parent
            if not non_terminal_children:
                if node.parent:
                    node.parent.children.remove(node)
                    node = node.parent
                else:
                    break
            else:
                # Select child with highest UCT value
                node = max(non_terminal_children, key=lambda child: child.uct())

        return node

    def _expand_node(
        self,
        node: Node,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        reflect_prompt: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        context: str = "Tree Expansion",
    ) -> Tuple[List[Node], Dict[str, Any]]:
        """Expand the given node by generating its children."""
        if node.depth >= self.depth_limit:
            node.is_terminal = True
            return [], {
                "thoughts_response": [],
                "actions_response": [],
                "reflections_response": [],
            }

        return self._generate_children_nodes(
            node=node,
            question=question,
            key=key,
            examples=examples,
            reflect_examples=reflect_examples,
            reflect_prompt=reflect_prompt,
            prompt=prompt,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
            context=context,
        )

    def _generate_children_nodes(
        self,
        node: Node,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        reflect_prompt: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        context: str = "Tree Expansion",
    ) -> Tuple[List[Node], Dict[str, Any]]:
        """Generate child nodes for the given node."""
        reflections_str = ""
        reflection_response: List[Response] = []

        # Generate reflections if needed
        if self._reflect_condition():
            reflections, reflection_response = self._reflect(
                question=question,
                examples=reflect_examples,
                prompt=reflect_prompt,
                additional_keys=reflect_additional_keys,
                context=context + " Reflection",
            )
            for reflection in reflections:
                reflections_str += (
                    _build_reflection_format(
                        trajectory=reflection["trajectory"],
                        reflection=reflection["reflection"],
                    )
                    + "\n\n"
                )

        trajectory = get_node_trajectory(node)
        unique_states = set()
        children_nodes, thoughts_response, actions_response = [], [], []

        for sample_idx in range(self.n_samples):
            trajectory_i, thought, thought_response = self._generate_thought(
                question=question,
                examples=examples,
                trajectory=trajectory,
                reflections=reflections_str,
                depth=node.depth,
                prompt=prompt,
                additional_keys=additional_keys,
                context=context,
                node_index=sample_idx,
            )

            trajectory_i, action_type, query, action_response = self._generate_action(
                question=question,
                examples=examples,
                trajectory=trajectory_i,
                reflections=reflections_str,
                depth=node.depth,
                prompt=prompt,
                additional_keys=additional_keys,
                context=context,
                node_index=sample_idx,
            )

            unique_key = f"{thought}::{action_type}::{query}"
            if unique_key not in unique_states:
                unique_states.add(unique_key)

                # Generate observation
                _, reward, obs, done, external_tool_info = self._generate_observation(
                    key=key,
                    action_type=action_type,
                    query=query,
                    trajectory=trajectory_i,
                    depth=node.depth,
                )

                new_node = Node(
                    state={
                        "thought": thought,
                        "action_type": action_type,
                        "query": query,
                        "observation": obs,
                        "answer": "" if not done else query,
                        "external_tool_info": external_tool_info,
                    },
                    parent=node,
                    depth=node.depth + 1,
                    is_terminal=reward == 1 or done,
                    reward=reward,
                )

                if new_node.is_terminal and reward == 0:
                    trajectory = get_node_trajectory(new_node)
                    self.failed_trajectories.append(
                        {
                            "trajectory": trajectory,
                            "final_answer": query,
                        }
                    )
            else:
                new_node = Node(
                    state={
                        "thought": thought,
                        "action_type": action_type,
                        "query": query,
                        "observation": "",
                        "answer": "",
                        "external_tool_info": {},
                    },
                )

            thoughts_response.append(thought_response)
            actions_response.append(action_response)
            children_nodes.append(new_node)

        metrics = {
            "thoughts_response": thoughts_response,
            "actions_response": actions_response,
            "reflections_response": reflection_response,
        }

        return children_nodes, metrics

    def _generate_thought(
        self,
        question: str,
        examples: str,
        trajectory: str,
        reflections: str,
        depth: int,
        prompt: str,
        additional_keys: Dict[str, str],
        context: str = "Tree Expansion",
        node_index: Optional[int] = None,
    ) -> Tuple[str, str, Response]:
        """Generate a thought for the current step."""
        trajectory += f"\nThought {depth + 1}: "

        # Build prompt
        prompt_kwargs = dict(
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
        )
        prompt_kwargs.update(additional_keys)
        full_prompt = prompt.format(**prompt_kwargs)

        # Retry mechanism for LLM calls
        for r in range(3):  # max_llm_retries
            out = self.llm(full_prompt)
            thought = out.output_text
            thought = remove_newline(thought).split("Action")[0]
            # Clean up any step prefixes like "Thought 1:"
            thought = clean_llm_output(thought)
            if thought.strip():  # Check if we got a valid thought
                log_llm_io(
                    out,
                    f"{context} - Thought",
                    self.verbose,
                    self.truncate_length,
                    parsed_output=thought.strip(),
                    depth=depth,
                    node_index=node_index,
                    retry=r,
                )
                break

        trajectory += thought
        return trajectory, thought, out

    def _generate_action(
        self,
        question: str,
        examples: str,
        trajectory: str,
        reflections: str,
        depth: int,
        prompt: str,
        additional_keys: Dict[str, str],
        context: str = "Tree Expansion",
        node_index: Optional[int] = None,
    ) -> Tuple[str, str, str, Response]:
        """Generate an action for the current step."""
        trajectory += f"\nAction {depth + 1}: "

        # Build prompt
        prompt_kwargs = dict(
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
        )
        prompt_kwargs.update(additional_keys)
        full_prompt = prompt.format(**prompt_kwargs)

        # Retry mechanism for LLM calls
        for r in range(3):  # max_llm_retries
            out = self.llm(full_prompt)
            action = out.output_text
            # Don't use remove_newline for actions as they need to preserve multiline structure
            action = action.split("Observation")[0].strip()
            # Clean up any step prefixes like "Action 5:"
            action = clean_llm_output(action)
            action_type, query = parse_math_action(action)
            if action_type and query:  # Check if we got a valid action
                # Check if query already contains code blocks to avoid double wrapping
                if "```python" in query:
                    formatted_query = query
                    parsed_action = f"{action_type}[{formatted_query}]"
                else:
                    # Format the action with code blocks like the strategy
                    formatted_query = f"\n```python\n{query}\n```\n"
                    parsed_action = f"{action_type}[{formatted_query}]"
                log_llm_io(
                    out,
                    f"{context} - Action",
                    self.verbose,
                    self.truncate_length,
                    parsed_output=parsed_action,
                    depth=depth,
                    node_index=node_index,
                    retry=r,
                )
                break

        # Format trajectory like the strategy, but avoid double wrapping
        if "```python" in query:
            formatted_query = query
        else:
            formatted_query = f"\n```python\n{query}\n```\n"
        trajectory += f" {action_type}[{formatted_query}]"
        return trajectory, action_type, formatted_query, out

    def _generate_observation(
        self,
        key: str,
        action_type: str,
        query: str,
        trajectory: str,
        depth: int,
    ) -> Tuple[str, int, str, bool, Dict[str, Any]]:
        """Generate an observation based on the current action."""
        external_tool_info = {"execution_status": "", "code_answer": ""}
        reward, done = 0, False
        trajectory += f"\nObservation {depth + 1}: "

        # Extract code from query like the strategy
        query = query.split("```python")[-1].split("```")[0].strip()
        # Add typing import for better compatibility with type hints
        code_with_imports = f"from typing import *\n\n{query}"
        code_answer, execution_status = safe_execute(code_with_imports)

        if action_type.lower() == "finish":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            if EM(str(code_answer[0]), key, is_numeric=True):
                obs = "Answer is CORRECT"
                reward = int(EM(str(code_answer[0]), key, is_numeric=True))
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

    def _evaluate_node(
        self,
        node: Node,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        context: str = "Node Evaluation",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate the given node and its children."""
        values, values_response = [], []
        child_trajectory_cache = {}

        for idx, child in enumerate(node.children):
            if not child.is_terminal:
                trajectory = get_node_trajectory(child)
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

                        # Log LLM I/O for value estimation
                        log_llm_io(
                            value_str_out,
                            f"{context} - Child {idx + 1}",
                            self.verbose,
                            self.truncate_length,
                        )

                        value_response = value_str_out
                        value_str = value_str_out.output_text

                        if self.cache_values:
                            self.value_cache[unique_key] = value_str

                    explanation, value = parse_value(value_str)
                    value = value / 10.0
                    node.children[idx].value = value
                    child_trajectory_cache[trajectory] = value

                values_response.append(value_response if value_response else None)
                values.append({"explanation": explanation, "value": value})
            else:
                values_response.append(None)
                values.append({"explanation": "", "value": -1e10})

        return values, {"values_response": values_response}

    def _simulate_node(
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
    ) -> Tuple[float, Node, List[Response]]:
        """Simulate from the given node to estimate its value."""
        depth = node.depth
        rewards: List[float] = [0.0]
        current_node = node
        simulation_responses = []

        while not current_node.is_terminal and depth < self.depth_limit:
            # Generate children for simulation
            children_nodes, generate_metrics = self._generate_children_nodes(
                node=current_node,
                question=question,
                key=key,
                examples=examples,
                reflect_examples=reflect_examples,
                reflect_prompt=reflect_prompt,
                prompt=prompt,
                additional_keys=additional_keys,
                reflect_additional_keys=reflect_additional_keys,
                context="Tree Simulation",
            )

            # Collect responses from generation
            for response_list in [
                generate_metrics.get("thoughts_response", []),
                generate_metrics.get("actions_response", []),
                generate_metrics.get("reflections_response", []),
            ]:
                simulation_responses.extend([r for r in response_list if r])

            # Check if any child is terminal
            terminal_children = [child for child in children_nodes if child.is_terminal]
            if terminal_children:
                current_node = terminal_children[0]
                rewards.append(float(current_node.reward))
                break

            # Evaluate children and select best
            values, evaluate_metrics = self._evaluate_node(
                node=current_node,
                question=question,
                examples=value_examples,
                prompt=value_prompt,
                additional_keys=value_additional_keys,
                context="Simulation Evaluation",
            )

            # Collect responses from evaluation
            for response in evaluate_metrics.get("values_response", []):
                if response:
                    simulation_responses.append(response)

            if values:
                max_value = max(values, key=lambda x: x["value"])
                max_value_index = values.index(max_value)
                rewards.append(max_value["value"])
                current_node = children_nodes[max_value_index]
            else:
                current_node = children_nodes[0] if children_nodes else current_node

            depth += 1

            if depth == self.depth_limit:
                rewards = [-1.0]

        return sum(rewards) / len(rewards), current_node, simulation_responses

    def _backpropagate_node(self, node: Node, value: float) -> None:
        """Backpropagate the estimated value through the tree."""
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if current_node.is_terminal:
                if current_node.reward == 0:
                    current_node.value = (
                        current_node.value * (current_node.visits - 1) + (-1)
                    ) / current_node.visits
                else:
                    current_node.value = (
                        current_node.value * (current_node.visits - 1) + value
                    ) / current_node.visits
            else:
                current_node.value = (
                    current_node.value * (current_node.visits - 1) + value
                ) / current_node.visits

            current_node = current_node.parent

    def _reflect_condition(self) -> bool:
        """Check if reflection should be performed."""
        return (
            len(self.failed_trajectories) > 0
            and len(self.reflection_map) < self.max_reflections
        )

    def _reflect(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        context: str = "Trajectory Reflection",
    ) -> Tuple[List[Dict[str, str]], List[Response]]:
        """Generate reflections on failed trajectories."""
        reflections = []
        reflection_responses = []

        for failed_trajectory in self.failed_trajectories[-self.max_reflections :]:
            reflection_kwargs = dict(
                question=question,
                examples=examples,
                trajectory=failed_trajectory["trajectory"],
            )
            reflection_kwargs.update(additional_keys)
            full_reflection_prompt = prompt.format(**reflection_kwargs)

            response = self.llm(full_reflection_prompt)

            # Log LLM I/O for reflection
            log_llm_io(
                response,
                f"{context} {len(reflections) + 1}",
                self.verbose,
                self.truncate_length,
            )

            reflection_responses.append(response)

            reflection = response.output_text.strip()
            reflections.append(
                {
                    "trajectory": failed_trajectory["trajectory"],
                    "reflection": reflection,
                }
            )

            self.reflection_map.append(
                {
                    "trajectory": failed_trajectory["trajectory"],
                    "reflection": reflection,
                }
            )

        return reflections, reflection_responses

    def _get_reflections_string(self) -> str:
        """Get reflections as a formatted string."""
        reflections_str = ""
        for reflection in self.reflection_map:
            reflections_str += (
                _build_reflection_format(
                    trajectory=reflection["trajectory"],
                    reflection=reflection["reflection"],
                )
                + "\n\n"
            )
        return reflections_str.strip()
