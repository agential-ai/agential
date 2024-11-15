"""General strategy for the ReAct Agent."""

# what constitutes a function? (E2 in paper)
# what's the initial function set for react? (guess is defaults)
# what are register_for_llm, register_for_exector? why does step return these two? how does react agent actually use these?

#no execute_code

#general strategy 
#with qa 

# factor qa out 


import copy
import time

from typing import Any, Dict, Optional, Tuple
import tiktoken

from tiktoken.core import Encoding

from agential.agents.react.functional import (
    _is_halted,
    _prompt_agent,
    accumulate_metrics,
)
from agential.agents.react.output import ReActOutput, ReActStepOutput
from agential.core.llm import BaseLLM, Response
from agential.training.agent_optimizer.strategies.base import AgentOptimizerBaseStrategy
from agential.utils.parse import remove_newline


class AgentOptimizerGeneralStrategy(AgentOptimizerBaseStrategy):
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
            optimizer_model: Optional[str] = "gpt-3.5-turbo",
            testing: bool = False,
        ) -> None:
            """Initialization."""
            super().__init__(
                llm=llm,
                max_actions_per_step=max_actions_per_step,
                max_steps=max_steps,
                max_tokens=max_tokens,
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
            
        performance = 2 #what is performance

        if performance < self._best_performance:
            self._failure_functions_performance.append({
                "functions": self._trial_functions,
                "performance": performance
            })
            self._failure_functions_performance = sorted(
                self._failure_functions_performance, key=lambda x: x["performance"]
            )
        else:
            self._failure_functions_performance.clear()
            self._best_performance = performance
            self._best_functions = copy.deepcopy(self._trial_functions)
            self._best_conversations_history = copy.deepcopy(self._trial_conversations_history)
            self._best_conversations_performance = copy.deepcopy(self._trial_conversations_performance)
        


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



class ReActGeneralStrategy(ReActBaseStrategy):
    """A general strategy class using the ReAct agent.

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
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(
            llm=llm,
            max_steps=max_steps,
            max_tokens=max_tokens,
            enc=enc,
            testing=testing,
        )

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
        self, 
        idx: int, 
        scratchpad: str, action_type: str, query: str
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

    def step(self):
        performance = self._calculate_performance()
        
        if self._is_improved_performance(performance):
            self._update_best(performance)
        else:
            self._record_failure(performance)

        self._reset_trial_data()
        best_functions, incumbent_functions = 0 #set this

        failure_experience_prompt, statistic_prompt = 0 #set this

        for action_index in range(self.max_actions_per_step):
            actions = self._generate_actions(action_index, best_functions, incumbent_functions, failure_experience_prompt, statistic_prompt)
            


            #if actions and validating true 
            #    we update function call 

        return out
    
    def _calculate_performance(self):
        """Calculate average performance for current trial conversations."""
        return sum(sum(d.values()) for d in self._trial_conversations_performance) / len(self._trial_conversations_performance)

    def _is_improved_performance(self, performance):
        """Check if the current performance is an improvement."""
        return performance > self._best_performance

    def _update_best(self, performance):
        """Update best performance and related records."""
        
    def _record_failure(self, performance):
        raise NotImplementedError


    def _reset_trial_data(self): # or just reset data in general?
        """Reset trial data for a new trial."""
        self._trial_conversations_performance = []
        
        raise NotImplementedError


    def _validate_actions(self, actions):
        """Validate the generated actions."""
        raise NotImplementedError


    def _generate_actions(
            self, 
            action_index, 
            best_functions, 
            incumbent_functions, 
            fail_prompt, 
            statistic_prompt):
        
        #most likely generate actions 
        # possibly no thought, observations 

        #observation could be the type of action user gives? or performance

        raise NotImplementedError

    def generate_action(self, action_index, best_functions, incumbent_functions, failure_experience_prompt, statistic_prompt):
        """Generates and validates actions based on the current prompt configuration.

        Args:
            action_index (int): The index for the current action iteration.
            best_functions (List[Dict]): The current best functions.
            incumbent_functions (List[Dict]): The functions to be updated based on actions.
            failure_experience_prompt (str): Experience gained from previous failures.
            statistic_prompt (str): Statistical information to guide action generation.

        Returns:
            List[Dict]: Updated incumbent functions based on validated actions.
        """
        prompt = OPT_PROMPT.format(
            best_conversations_history=self._best_conversations_history,
            best_conversations_num=len(self._best_conversations_history),
            actions_num=action_index,
            best_functions=best_functions,
            incumbent_functions=incumbent_functions,
            accumulated_experience=failure_experience_prompt,
            statistic_informations=statistic_prompt,
        )#should be the prompt agent thing
        messages = [{"role": "user", "content": prompt}]

        for i in range(self._max_trials):
            #replace this section somehow
            response = self._client.create(
                messages=messages, tools=[ADD_FUNC, REVISE_FUNC, REMOVE_FUNC], tool_choice="auto"
            )
            actions = response.choices[0].message.tool_calls
            if self._validate_actions(actions, incumbent_functions):
                return self._update_function_call(incumbent_functions, actions)
        
        #no valid actions, incumbent unchanged
        return incumbent_functions


    def _update_function_call(self, incumbent_functions, actions):
        """Updates the function call based on the validated actions."""



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
