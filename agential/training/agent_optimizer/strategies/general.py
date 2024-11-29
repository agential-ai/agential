"""General strategy for the Agent Optimizer."""

# what constitutes a function? (E2 in paper)
# what's the initial function set for react? (guess is defaults)
# what are register_for_llm, register_for_exector? why does step return these two? how does react agent actually use these?

# no execute_code

# general strategy
# with qa

# factor qa out


import copy
import json
import re
import time

from typing import Any, Dict, List, Optional, Tuple
import tiktoken

from tiktoken.core import Encoding

from agential.agents.react.functional import (
    _is_halted,
    _prompt_agent,
    accumulate_metrics,
)
from agential.agents.react.output import ReActOutput, ReActStepOutput
from agential.core.llm import BaseLLM, Response
from agential.training.agent_optimizer.functional import _build_training_step_prompt
from agential.training.agent_optimizer.output import PromptOptimizerOutput, PromptOptimizerStepOutput
from agential.training.agent_optimizer.prompts import ADD_FUNC, FAILURE_EXPERIENCE_P, IMPROVE_CODE_PROMPT, IMPROVE_FUNCTION_PROMPT, REMOVE_FUNC, REVISE_FUNC, STATISTIC_P
from agential.training.agent_optimizer.strategies.base import PromptOptimizerBaseStrategy
from agential.utils.parse import remove_newline


class PromptOptimizerGeneralStrategy(PromptOptimizerBaseStrategy):
    """A general strategy class using the Agent Optimizer.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
        testing (bool): Whether the agent is in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_actions_per_step: int,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
        optimizer_model: Optional[str] = "gpt-3.5-turbo",
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(
            llm=llm,
            max_actions_per_step=max_actions_per_step,
            max_steps=max_steps,
            max_tokens=max_tokens,
            enc=enc,
            testing=testing,
            optimizer_model=optimizer_model,
        )

        self.max_actions_per_step = max_actions_per_step
        self._max_trials = 3
        self.optimizer_model = optimizer_model

        self._trial_conversations_history = []
        self._trial_conversations_performance = []
        self._trial_functions = []

        self._best_conversations_history = []
        self._best_conversations_performance = []
        self._best_functions = []
        self._failure_functions_performance = []
        self._best_performance = -1

        self.llm = llm
        self.llm.config = copy.deepcopy(llm.config)

    def step(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reset: bool,
    ) -> ReActOutput:
        """Generate a ReAct output by iteratively thinking, acting, and observing.
        Args  :
            question (str): The question being answered.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the thought.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.
            reset (bool): Whether to reset the agent's state before generating.
            Returns:
                ReActOutput: The generated output.
        """
        if reset:
            self.reset()

        performance = 2  # what is performance

        if performance < self._best_performance:
            self._failure_functions_performance.append(
                {"functions": self._trial_functions, "performance": performance}
            )
            self._failure_functions_performance = sorted(
                self._failure_functions_performance, key=lambda x: x["performance"]
            )
        else:
            self._failure_functions_performance.clear()
            self._best_performance = performance
            self._best_functions = copy.deepcopy(self._trial_functions)
            self._best_conversations_history = copy.deepcopy(
                self._trial_conversations_history
            )
            self._best_conversations_performance = copy.deepcopy(
                self._trial_conversations_performance
            )

    def reset(self):
        """Reset the agent's state."""
        self._trial_conversations_history = []
        self._trial_conversations_performance = []
        self._trial_functions = []

    def execute_function(
        self,
        name: str,
        packages: str,
        code: str,
        args: str,
    ) -> Tuple[str, str]:
        """Execute a function and return the result and the function's name."""


    def execute_func(name, packages, code, **args):
        """
        The wrapper for generated functions.
        """
        pip_install = (
            f"""print("Installing package: {packages}")\nsubprocess.run(["pip", "-qq", "install", "{packages}"])"""
            if packages
            else ""
        )
        str = f"""
    import subprocess
    {pip_install}
    print("Result of {name} function execution:")
    {code}
    args={args}
    result={name}(**args)
    if result is not None: print(result)
    """
        print(f"execute_code:\n{str}")
        result = execute_code(str, use_docker="shaokun529/evoagent:v1")
        if result[0] != 0:
            raise Exception("Error in executing function:" + result[1])
        print(f"Result: {result[1]}")
        return result[1]


    def generate(
        self,
        objective: str,
        constraints: Optional[str],
        context: str,
        additional_keys: Dict[str, str],
        reset: bool,
    ) -> PromptOptimizerOutput:
        """Generate an optimized solution by iteratively generating, evaluating, and refining steps.

        Args:
            objective (str): The optimization goal or target.
            constraints (Optional[str]): Constraints or limits for the optimization.
            context (str): The context or background information for the task.
            additional_keys (Dict[str, str]): Additional parameters for the language model.
            reset (bool): Whether to reset the optimizer's state before generating.

        Returns:
            OptimizerOutput: The final optimized solution, metrics, and intermediate steps.
        """
        start = time.time()

        if reset:
            self.reset()

        scratchpad = ""
        solution = ""
        finished = False
        step_idx = 1
        steps = []

        while not self.halting_condition(
            finished=finished,
            step_idx=step_idx,
            objective=objective,
            scratchpad=scratchpad,
            context=context,
            constraints=constraints,
            additional_keys=additional_keys,
        ):

            # generate thought
            scratchpad, hypothesis, hypothesis_response = self.generate_thought(
                step_idx=step_idx,
                scratchpad=scratchpad,
                objective=objective,
                context=context,
                constraints=constraints,
                additional_keys=additional_keys,
            )

            # generate action 
            scratchpad, evaluation, evaluation_response = self.generate_action(
                step_idx=step_idx,
                scratchpad=scratchpad,
                hypothesis=hypothesis,
                context=context,
                additional_keys=additional_keys,
            )

            # generate observation
            scratchpad, solution, refinement_response, finished = self.generate_observation(
                step_idx=step_idx,
                scratchpad=scratchpad,
                evaluation=evaluation,
                context=context,
                objective=objective,
                additional_keys=additional_keys,
            )

            steps.append(
                PromptOptimizerStepOutput(
                    hypothesis=hypothesis,
                    evaluation=evaluation,
                    refined_solution=solution,
                    hypothesis_response=hypothesis_response,
                    evaluation_response=evaluation_response,
                    refinement_response=refinement_response,
                )
            )

            step_idx += 1

        total_time = time.time() - start
        total_metrics = accumulate_metrics(steps)
        output = PromptOptimizerOutput(
            final_solution=solution,
            total_prompt_tokens=total_metrics["total_prompt_tokens"],
            total_completion_tokens=total_metrics["total_completion_tokens"],
            total_tokens=total_metrics["total_tokens"],
            total_prompt_cost=total_metrics["total_prompt_cost"],
            total_completion_cost=total_metrics["total_completion_cost"],
            total_cost=total_metrics["total_cost"],
            total_time=total_time if not self.testing else 0.5,
            steps=steps,
        )

        return output



    def generate_action(
        self,
        step_idx: int,
        scratchpad: str,
        hypothesis: str,
        context: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str]:
        """Generate an action based on the current state and context."""


    def generate_observation(
        self,
        step_idx: int,
        scratchpad: str,
        hypothesis: str,
        context: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str]:
        """Generate an action based on the current state and context."""


    def generate_thought(
        self,
        step_idx: int,
        scratchpad: str,
        hypothesis: str,
        context: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str]:
        """Generate an action based on the current state and context."""

    
    def step(self):
        """Perform a single step in the optimization process by editing prompts instead of modifying functions."""
        
        performance = self._calculate_performance()
        best_prompts = []
        failure_prompts = []

        if self._is_improved_performance(performance):
            self._update_best(performance)
            best_prompts.append(f"New best performance achieved: {performance}")
        else:
            self._record_failure(performance)
            failure_prompts.append(f"Performance did not improve. Current score: {performance}")

        self._reset_trial_data()

        best_functions, incumbent_functions = 0  # set this

        failure_experience_prompt, statistic_prompt = 0  # set this

        statistic_prompts = [f"Step statistics: Current performance: {performance}"]
        experience_prompts = failure_prompts if failure_prompts else best_prompts

        for action_index in range(self.max_actions_per_step):
            actions = self._generate_actions(
                action_index,
                best_functions,
                incumbent_functions,
                failure_experience_prompt,
                statistic_prompt,
            )

        for action_index in range(self.max_actions_per_step):

            action_prompts = [
                f"Action {action_index}: Generate a strategy based on the current performance.",
                f"Best prompts so far: {', '.join(best_prompts)}",
                f"Experience prompts: {', '.join(experience_prompts)}",
                f"Statistics prompts: {', '.join(statistic_prompts)}",
            ]

            combined_prompt = "\n".join(action_prompts)

            # call llm o tgenerate new actions
            actions = self.language_model.generate(combined_prompt)

            if actions and self._validate_actions(actions):
                # update trial data?
                self._update_trial(actions, performance)
            else:
                failure_prompts.append(f"Action {action_index} failed validation.")

        out = {
            "best_prompts": best_prompts,
            "failure_prompts": failure_prompts,
            "statistic_prompts": statistic_prompts,
            "actions": actions,
        }

        return out

    def _calculate_performance(self):
        """Calculate average performance for current trial conversations."""
        return sum(
            sum(d.values()) for d in self._trial_conversations_performance
        ) / len(self._trial_conversations_performance)

    def _is_improved_performance(self, performance):
        """Check if the current performance is an improvement."""
        return performance > self._best_performance

    def _update_best(self, performance):
        """Update best performance and related records."""

    def _record_failure(self, performance):
        raise NotImplementedError

    def _reset_trial_data(self):  # or just reset data in general?
        """Reset trial data for a new trial."""
        self._trial_conversations = []
        self._trial_conversations_performance = []
        self._current_prompts = []


    def _validate_actions(self, actions):
        """Validate the generated actions."""
        raise NotImplementedError

    def _generate_actions(
        self,
        action_index,
        best_functions,
        incumbent_functions,
        fail_prompt,
        statistic_prompt,
    ):

        # most likely generate actions
        # possibly no thought, observations

        # observation could be the type of action user gives? or performance

        raise NotImplementedError

    def generate_action(
        self,
        action_index,
        best_functions,
        incumbent_functions,
        failure_experience_prompt,
        statistic_prompt,
    ):
        """Generates and validates actions based on the current training prompt.

        Args:
            action_index (int): The index for the current action iteration.
            best_functions (List[Dict]): The current best functions.
            incumbent_functions (List[Dict]): The functions to be updated based on actions.
            failure_experience_prompt (str): Experience gained from previous failures.
            statistic_prompt (str): Statistical information to guide action generation.

        Returns:
            List[Dict]: Updated incumbent functions based on validated actions.
        """
        prompt = _build_training_step_prompt(
            best_conversations_history=self.best_conversations_history,
            actions_num=action_index,
            best_functions=best_functions,
            incumbent_functions=incumbent_functions,
            accumulated_experience=failure_experience_prompt,
            statistic_information=statistic_prompt,
        )
        messages = [{"role": "user", "content": prompt}]

        for i in range(self._max_trials):
            response = self._client.create(
                messages=messages,
                tools=[ADD_FUNC, REVISE_FUNC, REMOVE_FUNC],
                tool_choice="auto",
            )
            actions = response.choices[0].message.tool_calls
            if self._validate_actions(actions, incumbent_functions):
                return self._update_function_call(incumbent_functions, actions)

        # no valid, incumbent_functions ret
        return incumbent_functions

    def call_llm(
        llm: BaseLLM,
        messages: list,
        tools: list,
        tool_choice: str = "auto",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """Calls the LLM to generate a response based on the input messages and tools.

        Args:
            client: The LLM client instance for making requests.
            messages (list): List of input messages for the LLM.
            tools (list): List of tool options (e.g., ADD_FUNC, REVISE_FUNC, REMOVE_FUNC).
            tool_choice (str): The strategy for tool selection. Defaults to "auto".
            max_tokens (int): The maximum number of tokens for the response. Defaults to 256.
            temperature (float): Sampling temperature for randomness in the response. Defaults to 0.7.

        Returns:
            Response: The LLM's response, including the tool calls.
        """
        return llm.create(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def improve_code(
        llm,
        function_code: str,
        function_description: str,
        improvement_goals: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Improves the given function's code using an LLM.

        Args:
            client: The LLM client instance for making requests.
            function_code (str): The code of the function to be improved.
            function_description (str): A description of the function's purpose and expected behavior.
            improvement_goals (list): Goals for improvement (e.g., efficiency, readability).
            max_tokens (int): The maximum tokens for the response. Defaults to 512.
            temperature (float): Sampling temperature for randomness. Defaults to 0.7.

        Returns:
            str: The improved code as suggested by the LLM.
        """
        # what should prompt be?
        prompt = ""

        # call llm
        response = llm.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # better code (assuming it HAS been iproved )
        return response.choices[0].message.content.strip()

    def _update_function_call(self, incumbent_functions, actions):
        """Updates the function call based on the validated actions."""

    def update_agent_functions(
        existing_functions: List[Dict[str, Any]], 
        actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Update the list of agent functions based on provided actions.

        Args:
            existing_functions (List[Dict[str, Any]]): The current list of agent functions,
                where each function is a dictionary containing attributes like `name`, `description`, etc.
            actions (List[Dict[str, Any]]): A list of action dictionaries specifying how to update the functions.
                Each action includes details like `action_name`, `name`, and other optional attributes.

        Returns:
            List[Dict[str, Any]]: The updated list of agent functions.
        """
        formatted_actions = []

        for action in actions:
            try:
                func_data = json.loads(action["function"]["arguments"].strip('"'))
                func_data["action_name"] = action["function"]["name"]

                if func_data.get("action_name") == "remove_function":
                    formatted_actions.append(
                        {
                            "action_name": func_data.get("action_name"),
                            "name": func_data.get("name"),
                        }
                    )
                else:
                    formatted_actions.append(
                        {
                            "action_name": func_data.get("action_name"),
                            "name": func_data.get("name"),
                            "description": func_data.get("description"),
                            "arguments": json.loads(
                                func_data.get("arguments").strip('"')
                            ),
                            "packages": func_data.get("packages"),
                            "code": func_data.get("code"),
                        }
                    )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Failed to process action: {action}. Error: {e}")
                continue

        for action in formatted_actions:
            action_name = action.get("action_name")
            if action_name == "remove_function":
                existing_functions = [
                    func
                    for func in existing_functions
                    if func["name"] != action["name"]
                ]
            else:
                existing_functions = [
                    func
                    for func in existing_functions
                    if func["name"] != action["name"]
                ]
                existing_functions.append(
                    {
                        "name": action["name"],
                        "description": action.get("description"),
                        "arguments": action.get("arguments"),
                        "packages": action.get("packages"),
                        "code": action.get("code"),
                    }
                )

        return existing_functions

    def _update_prompt_call(self, incumbent_prompts, actions):
        """
        Updates the prompt calls based on the provided actions.

        Args:
            incumbent_prompts (List[Dict[str, Any]]): The current list of prompts.
            actions (List[Dict[str, Any]]): A list of action dictionaries specifying how to update the prompts.
                Each action includes details like `action_name`, `name`, and other optional attributes.

        Returns:
            List[Dict[str, Any]]: The updated list of prompts.
        """
        formatted_actions = []

        for action in actions:
            prompt_data = json.loads(action.function.arguments.strip('"'))
            prompt_data["action_name"] = action.function.nae

            if prompt_data.get("action_name") == "remove_prompt":
                item = {
                    "action_name": prompt_data.get("action_name"),
                    "name": prompt_data.get("name"),
                }
            else:
                item = {
                    "action_name": prompt_data.get("action_name"),
                    "name": prompt_data.get("name"),
                    "content": prompt_data.get("content"),
                    "context": json.loads(prompt_data.get("context").strip('"')),
                    "metadata": prompt_data.get("metadata"),
                }
            formatted_actions.append(item)

        actions = formatted_actions

        for action in actions:
            name, content, context, metadata, action_name = (
                action.get("name"),
                action.get("content"),
                action.get("context"),
                action.get("metadata"),
                action.get("action_name"),
            )
            if action_name == "remove_prompt":
                incumbent_prompts = [
                    item for item in incumbent_prompts if item["name"] != name
                ]
            else:
                incumbent_prompts = [
                    item for item in incumbent_prompts if item["name"] != name
                ]
                incumbent_prompts.append(
                    {
                        "name": name,
                        "content": content,
                        "context": context,
                        "metadata": metadata,
                    }
                )

        return incumbent_prompts


    def generate_code(
            pattern: str = "pattern"
        ) -> Tuple[str, float]:
        """
        Generate code using Agential's LLM calls.

        Args:
            pattern (str): The regular expression pattern for extracting code blocks.
            config (dict): Configuration for the LLM API call.

        Returns:
            Tuple[str, float]: Generated code and the cost of the operation.
        """
        response = call_llm(XXX)
        cost = response.get("cost", 0.0)
        text_output = response.get("content", "")
        generated_code = extract_code(text_output, pattern)
        return generated_code, cost

    def improve_function(
        file_name: str, 
        func_name: str, 
        objective: str, 
    ) -> Tuple[str, float]:
        """
        Improve the specified function in the given file to achieve a defined objective.

        Args:
            file_name (str): Path to the file containing the function.
            func_name (str): Name of the function to be improved.
            objective (str): The objective to achieve with the function improvement.
            config (dict): Configuration for the LLM API call.

        Returns:
            Tuple[str, float]: Improved function code and the cost of the operation.
        """
        with open(file_name, "r") as f:
            file_string = f.read()

        prompt = IMPROVE_FUNCTION_PROMPT

        params = {"prompt": prompt, "model": "gpt-3.5-turbo", "timeout": 600}
        response = call_llm(**params)
        cost = response.get("cost", 0.0)
        improved_function = response.get("content", "")
        return improved_function, cost


    def construct_intermediate_prompt(failure_functions_performance, best_conversations_performance):
        """
        Constructs intermediate prompts to provide performance feedback and statistical context.
        """

        if failure_functions_performance:
            failure_experience_prompt = FAILURE_EXPERIENCE_P
            for item in failure_functions_performance:
                failure_experience_prompt += f"Function:\n{item['functions']}\n"
                failure_experience_prompt += f"Performance:\n{item['performance']}\n"
        else:
            failure_experience_prompt = ""

        if best_conversations_performance:
            statistic_prompt = STATISTIC_P
            for item in best_conversations_performance:
                statistic_prompt += f"{item}\n"
        else:
            statistic_prompt = ""

        return failure_experience_prompt, statistic_prompt


    def improve_code(
        files: List[str], objective: str, suggest_only: bool = True, 
    ) -> Tuple[str, float]:
        """
        Improve the code in multiple files or provide suggestions for improvement.

        Args:
            files (List[str]): List of file paths containing the source code.
            objective (str): The objective to achieve with the code improvements.
            suggest_only (bool): Whether to return only suggestions (True) or include improved code (False).
            config (dict): Configuration for the LLM API call.

        Returns:
            Tuple[str, float]: Suggestions or improved code and the cost of the operation.
        """
        code = ""
        for file_name in files:
            with open(file_name, "r") as f:
                file_string = f.read()
            code += f"""{file_name}:
    {file_string}

    """

        followup = "" if suggest_only else " followed by the improved code"
        prompt = IMPROVE_CODE_PROMPT

        params = {"prompt": prompt, "model": "gpt-3.5-turbo", "timeout": 900}
        response = call_llm(**params)
        cost = response.get("cost", 0.0)
        result = response.get("content", "")
        return result, cost


#IMPORTANT

#CODE SNIPPET FORMAT: r"```(.*?)```"

    def extract_code(
            content: str, 
            pattern: str
        ) -> str:
        """
        Extract code blocks from the provided content using a given pattern.

        Args:
            content (str): The content to search for code blocks.
            pattern (str): The regular expression pattern for extracting code.

        Returns:
            str: Extracted code blocks as a single string.
        """
        
        matches = re.findall(pattern, content, re.DOTALL)
        return "\n\n".join(matches)



    def halting_condition(
        self,
        finished: bool,
        idx: int,
        question: str,
        scratchpad: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determines whether the current iteration of the task should be halted based on various conditions.

        Args:
            finished (bool): Whether the task has been completed.
            idx (int): The current index of the iteration.
            question (str): The question being answered.
            scratchpad (str): The current state of the scratchpad.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the action.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.

        Returns:
            bool: True if the task should be halted, False otherwise.
        """
        return _is_halted(
            finished=finished,
            idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reset(self) -> None:
        """Resets the internal state."""
        pass




    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reset: bool,
    ) -> ReActOutput:
        """Generate a ReAct output by iteratively thinking, acting, and observing.

        Args:
            question (str): The question being answered.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the thought.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.
            reset (bool): Whether to reset the agent's state before generating.

        Returns:
            ReActOutput: The generated output, including the final answer, metrics, and step-by-step details.
        """
        start = time.time()

        if reset:
            self.reset()

        scratchpad = ""
        answer = ""
        finished = False
        idx = 1
        steps = []
        while not self.halting_condition(
            finished=finished,
            idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        ):
            # Think.
            scratchpad, thought, thought_response = self.generate_thought(
                idx=idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act.
            scratchpad, action_type, query, action_response = self.generate_action(
                idx=idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe.
            scratchpad, answer, obs, finished, external_tool_info = (
                self.generate_observation(
                    idx=idx, scratchpad=scratchpad, action_type=action_type, query=query
                )
            )

            steps.append(
                ReActStepOutput(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    observation=obs,
                    answer=answer,
                    external_tool_info=external_tool_info,
                    thought_response=thought_response,
                    action_response=action_response,
                )
            )

            idx += 1

        total_time = time.time() - start
        total_metrics = accumulate_metrics(steps)
        out = ReActOutput(
            answer=answer,
            total_prompt_tokens=total_metrics["total_prompt_tokens"],
            total_completion_tokens=total_metrics["total_completion_tokens"],
            total_tokens=total_metrics["total_tokens"],
            total_prompt_cost=total_metrics["total_prompt_cost"],
            total_completion_cost=total_metrics["total_completion_cost"],
            total_cost=total_metrics["total_cost"],
            total_prompt_time=total_metrics["total_prompt_time"],
            total_time=total_time if not self.testing else 0.5,
            additional_info=steps,
        )

        return out

    def generate_thought(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, Response]:
        """Generate a thought based on the given inputs.

        Args:
            idx (int): The current index of the thought.
            scratchpad (str): The current state of the scratchpad.
            question (str): The question being answered.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the thought.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.

        Returns:
            Tuple[str, str, Response]: The updated scratchpad, the generated thought, and the metrics for the thought.
        """
        scratchpad += f"\nThought {idx}: "

        out = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = remove_newline(out.output_text).split("Action")[0].strip()
        scratchpad += thought

        return scratchpad, thought, out

    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generate an action based on the given inputs.

        Args:
            idx (int): The current index of the action.
            scratchpad (str): The current state of the scratchpad.
            question (str): The question being answered.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the action.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.

        Returns:
            Tuple[str, str, str, Response]: The updated scratchpad, the generated action, the action type, and the metrics for the action.
        """
        raise NotImplementedError

    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str
    ) -> Tuple[str, str, str, bool, Dict[str, Any]]:
        """Generate an observation based on the given inputs.

        Args:
            idx (int): The current index of the observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action performed.
            query (str): The query or action to observe.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: A tuple containing:
                - The updated scratchpad.
                - The generated observation.
                - The observation type.
                - A boolean indicating if the task is finished.
                - A dictionary with additional information.
        """
        raise NotImplementedError

    def halting_condition(
        self,
        finished: bool,
        idx: int,
        question: str,
        scratchpad: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determines whether the current iteration of the task should be halted based on various conditions.

        Args:
            finished (bool): Whether the task has been completed.
            idx (int): The current index of the iteration.
            question (str): The question being answered.
            scratchpad (str): The current state of the scratchpad.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the action.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.

        Returns:
            bool: True if the task should be halted, False otherwise.
        """
        return _is_halted(
            finished=finished,
            idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reset(self) -> None:
        """Resets the internal state."""
        pass