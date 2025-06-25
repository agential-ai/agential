"""
LATS Math Agent for mathematical problem-solving benchmarks (full functionality version).
"""

from typing import Dict, Any, List, Optional, Tuple
import time
from agential.core.llm import BaseLLM, Response
from agential.agents.base import BaseAgent
from agential.agents.lats.prompts import *
from agential.agents.lats.utils import (
    parse_value,
)
from agential.agents.lats.node import Node
from agential.agents.lats.output import LATSReActStepOutput
from agential.agents.lats.functional import (
    _build_failed_trajectory_format,
    _build_reflection_format,
    _prompt_agent,
    _prompt_value,
    get_node_trajectory,
    parse_math_action,
)
from agential.eval.classification import EM
from agential.utils.general import safe_execute


class LATSMath(BaseAgent):
    """Full LATS Math Agent that implements complete tree search functionality."""
    
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        n_samples: int = 5,
        max_reflections: int = 4,
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
        max_iterations: int = 30,
    ) -> Dict[str, Any]:
        start_time = time.time()
        total_tokens = total_cost = 0
        scratchpad, answer, steps, step_metrics = "", "", [], []
        
        # Get prompts and examples
        examples = self.config.get("fewshot", "")
        prompt = self.config.get("prompt", "")
        reflect_prompt = self.config.get("reflect_prompt", "")
        value_prompt = self.config.get("value_prompt", "")
        
        # Initialize root node
        self.root = Node(
            state=LATSReActStepOutput(
                thought="",
                action_type="",
                query="",
                observation="",
                answer="",
                external_tool_info={},
            ),
            depth=0,
            is_terminal=False,
            reward=0,
        )
        
        # Main tree search loop
        current_node = self.root
        iteration = 0
        
        while not current_node.is_terminal and iteration < max_iterations:
            iteration += 1
            step_start = time.time()
            
            # Generate children nodes
            children_nodes, generate_metrics = self._generate_children_nodes(
                node=current_node,
                question=question,
                key=key,
                examples=examples,
                reflect_prompt=reflect_prompt,
                prompt=prompt,
                additional_keys=additional_keys,
                reflect_additional_keys=reflect_additional_keys,
            )
            
            # Add children to current node
            current_node.children = children_nodes
            
            # Evaluate children if not terminal
            if children_nodes and not any(child.is_terminal for child in children_nodes):
                values, evaluate_metrics = self._evaluate_node(
                    node=current_node,
                    question=question,
                    examples=examples,
                    prompt=value_prompt,
                    additional_keys=value_additional_keys,
                )
                
                # Select best child based on value
                best_child_idx = max(range(len(values)), key=lambda i: values[i]["value"])
                current_node = children_nodes[best_child_idx]
            else:
                # If any child is terminal, select the first terminal one
                terminal_children = [child for child in children_nodes if child.is_terminal]
                if terminal_children:
                    current_node = terminal_children[0]
                    break
                elif children_nodes:
                    current_node = children_nodes[0]  # Select first child if no terminal
                else:
                    break
            
            # Update metrics
            step_time = time.time() - step_start
            step_metrics.append({
                "step": iteration,
                "total_step_time": step_time,
                "total_step_tokens": sum(getattr(r, "total_tokens", 0) for r in generate_metrics.get("thoughts_response", [])),
                "total_step_cost": sum(getattr(r, "total_cost", 0) for r in generate_metrics.get("thoughts_response", [])),
            })
            
            # Update total tokens and cost
            for response_list in [generate_metrics.get("thoughts_response", []), 
                                generate_metrics.get("actions_response", []),
                                generate_metrics.get("reflections_response", [])]:
                for response in response_list:
                    if response:
                        total_tokens += getattr(response, "total_tokens", 0)
                        total_cost += getattr(response, "total_cost", 0)
        
        # Extract final answer and trajectory
        if current_node and current_node.state:
            answer = current_node.state.answer
            scratchpad = get_node_trajectory(current_node) if current_node else ""
        
        # Build steps from trajectory
        if scratchpad:
            lines = scratchpad.split('\n')
            current_step = {}
            for line in lines:
                if line.startswith('Thought'):
                    if current_step:
                        steps.append(current_step)
                    current_step = {"thought": line.split(':', 1)[1].strip() if ':' in line else ""}
                elif line.startswith('Action'):
                    if ':' in line:
                        action_part = line.split(':', 1)[1].strip()
                        action_type, query = parse_math_action(action_part)
                        current_step["action_type"] = action_type
                        current_step["query"] = query
                elif line.startswith('Observation'):
                    if ':' in line:
                        current_step["observation"] = line.split(':', 1)[1].strip()
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

    def _generate_children_nodes(
        self,
        node: Node,
        question: str,
        key: str,
        examples: str,
        reflect_prompt: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
    ) -> Tuple[List[Node], Dict[str, Any]]:
        """Generate child nodes for the given node."""
        reflections_str = ""
        reflection_response: List[Response] = []
        
        # Generate reflections if needed
        if self._reflect_condition():
            reflections, reflection_response = self._reflect(
                question=question,
                examples=self.config.get("reflect_examples", ""),
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

        trajectory = get_node_trajectory(node)
        unique_states = set()
        children_nodes, thoughts_response, actions_response = [], [], []
        
        for _ in range(self.n_samples):
            # Generate thought
            trajectory_i, thought, thought_response = self._generate_thought(
                question=question,
                examples=examples,
                trajectory=trajectory,
                reflections=reflections_str,
                depth=node.depth,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            
            # Generate action
            trajectory_i, action_type, query, action_response = self._generate_action(
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

                # Generate observation
                _, reward, obs, done, external_tool_info = self._generate_observation(
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
                    trajectory = get_node_trajectory(new_node)
                    self.failed_trajectories.append(
                        {
                            "trajectory": trajectory,
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
    ) -> Tuple[str, str, Response]:
        """Generate a thought for the current step."""
        trajectory += f"\nThought {depth + 1}: "
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = out.output_text
        thought = thought.split("Action")[0].strip()
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
    ) -> Tuple[str, str, str, Response]:
        """Generate an action for the current step."""
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
        action = out.output_text
        action = action.split("Observation")[0].strip()
        action_type, query = parse_math_action(action)
        trajectory += f" {action_type}[\n```python\n{query}\n```\n]"

        return trajectory, action_type, f"\n```python\n{query}\n```\n", out

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
        query = query.split("```python")[-1].split("```")[0].strip()
        code_answer, execution_status = safe_execute(query)

        reward, done = 0, False
        trajectory += f"\nObservation {depth + 1}: "
        
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
            obs = "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
        
        trajectory += obs
        return trajectory, reward, obs, done, external_tool_info

    def _evaluate_node(
        self,
        node: Node,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
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

    def _reflect_condition(self) -> bool:
        """Check if reflection should be performed."""
        return len(self.failed_trajectories) > 0 and len(self.reflection_map) < self.max_reflections

    def _reflect(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[Dict[str, str]], List[Response]]:
        """Generate reflections on failed trajectories."""
        reflections = []
        reflection_responses = []
        
        for failed_trajectory in self.failed_trajectories[-self.max_reflections:]:
            reflection_kwargs = dict(
                question=question,
                examples=examples,
                trajectory=failed_trajectory["trajectory"],
            )
            reflection_kwargs.update(additional_keys)
            full_reflection_prompt = prompt.format(**reflection_kwargs)
            
            response = self.llm(full_reflection_prompt)
            reflection_responses.append(response)
            
            reflection = response.output_text.strip()
            reflections.append({
                "trajectory": failed_trajectory["trajectory"],
                "reflection": reflection,
            })
            
            self.reflection_map.append({
                "trajectory": failed_trajectory["trajectory"],
                "reflection": reflection,
            })
        
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