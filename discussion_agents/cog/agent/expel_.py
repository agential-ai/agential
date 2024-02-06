"""ExpeL Agent.

Original Paper: https://arxiv.org/abs/2308.10144
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

class ReflectAgent(ReactAgent):
    """
    A Generic Reflection Agent.
    """
    def __init__(
        self,
        # REACT #
        name: str,
        system_instruction: Union[str, Dict[str, str]],
        human_instruction: Callable,
        fewshots: Union[List[str], Dict[str, List[str]]],
        system_prompt: Callable,
        env: BaseEnv,
        llm: str,
        llm_builder: Callable,
        openai_api_key: str,
        tasks: List[Dict[str, Any]],
        max_steps: int,
        llm_parser: Callable,
        observation_formatter: Callable,
        testing: bool = False,
        task_idx: int = 0,
        benchmark_name=None,
        # REFLECT #
        reflection_fewshots: List[str],
        reflection_task_prompt: Callable,
        reflection_system_instruction: str,
        max_relfection_depth: int,
        message_splitter: Callable,
        identifier: Callable,
        message_step_splitter: Callable,
        reflection_prefix: str,
        previous_trials_formatter: Callable,
        # EXPEL #
        system_critique_instructions: Dict[str, str],
        human_critiques: Dict[str, PromptTemplate],
        rule_template: PromptTemplate,
        max_num_rules: Union[int, str],
        truncate_strategy: str,
        embedder: Callable,
        embedder_path: str,
        step_stripper: Callable,
        retriever_cls: Callable,
        success_critique_num: int,
        fewshot_strategy: str,
        critique_truncate_strategy: str,
        critique_summary_suffix: str,
        max_fewshot_tokens: int,
        reranker: str,
        buffer_retrieve_ratio: int,
    ) -> None:
        ############### EXPPEL ################
        self.benchmark_name = benchmark_name
        self.system_critique_instructions = system_critique_instructions
        self.human_critiques = human_critiques
        self.max_num_rules = max_num_rules
        self.rule_template = rule_template
        self.truncate_strategy = truncate_strategy
        self.critique_truncate_strategy = critique_truncate_strategy
        self.embedder = embedder(model_name=embedder_path)
        self.fewshot_strategy = fewshot_strategy
        self.retriever_cls = retriever_cls
        self.step_stripper = step_stripper
        self.success_critique_num = success_critique_num
        self.reranker = reranker
        self.buffer_retrieve_ratio = buffer_retrieve_ratio
        self.failed_training_task_idx = []
        self.critique_summary_suffix = critique_summary_suffix
        self.max_fewshot_tokens = max_fewshot_tokens
        self.eval_successes = []
        self.succeeded_trial_history: Dict[str, Trajectory] = {}
        self.failed_trial_history: Dict[str, Trajectory] = {}
        self.critiques = {}
        self.all_success_critiques = {}
        self.past_reflections = {}
        self.rule_items = []
        self.rule_items_with_count = []
        self.cache_rules = {}
        self._train = True
        ############### REFLECT ################
        self.reflection_counter = Count(max_relfection_depth)
        self.reflection_fewshots = reflection_fewshots
        self.reflection_task_prompt = reflection_task_prompt
        self.message_splitter = message_splitter
        self.identifier = identifier
        self.message_step_splitter = message_step_splitter
        self.reflection_prefix = reflection_prefix
        self.format_reflections = previous_trials_formatter
        self.reflection_prompt_history = []
        self.reflections = []
        self.previous_trial = []
        self.formatted_reflection = None
        self.perform_reflection = False
        self.increment_task = False
        ai_name = 'an advanced reasoning agent that can improve based on self refection'
        self.reflection_system_kwargs = dict(instruction=reflection_system_instruction, ai_name=ai_name)
        
        ############### REACT ################
        self.name = name
        self.tasks = tasks
        self.task_idx = task_idx
        self.all_system_instruction = system_instruction
        self.human_instruction = human_instruction
        self.human_instruction_kwargs = {'max_steps': max_steps}
        self.all_fewshots = fewshots
        self.system_prompt = system_prompt
        self.prompt_history = []
        self.testing = testing
        self.max_steps = max_steps
        self.llm_parser = llm_parser
        self.observation_formatter = observation_formatter
        self._last_observation_history = None

        self.env = env(**self.tasks[self.task_idx]['env_kwargs'], max_steps=self.max_steps)
        self.env.reset()
        self.task = self.tasks[self.task_idx]['task']
        self.reset()
        self.truncated, self.reward, self.terminated = False, False, False
        self.print_message = partial(print_message, testing=testing)

        self.success, self.fail, self.halted = 0, 0, 0

        self.llm = llm_builder(llm_name=llm, openai_api_key=openai_api_key, long_ver=False)
        self.long_context_llm = llm_builder(llm_name=llm, openai_api_key=openai_api_key, long_ver=True)
        del openai_api_key
        self.token_counter = partial(token_counter, llm=llm, tokenizer=getattr(self.llm, 'tokenizer', None))

        # build base prompt
        self._build_agent_prompt()
        self.update_dynamic_prompt_components()
        
        self.long_pass = None

        self.idx2task = {idx: task['task'] for idx, task in enumerate(self.tasks)}
        self.task2idx = {task['task']: idx for idx, task in enumerate(self.tasks)}

    def is_terminated(self) -> bool:
        return self.env.is_terminated()

    def log_history(self, include_task: bool = True, include_all: bool = False) -> str:
        all_history = '\n'.join([prompt.content for prompt in self.prompt_history])
        if include_all:
            return all_history

        # only log the task prompt and the agent's response
        reflection_pattern = r'{}'.format(self.format_reflections(self.reflections, include_prefix=False))
        match = re.search(re.escape(reflection_pattern), all_history)
        if not match or match.group() == '' or not include_task:
            task_text_list = human_task_message_prompt.format_messages(task=self.remove_task_suffix(self.task))[0].content.split('\n')
            task_text = '\n'.join(task_text_list)
            pattern = r'{}'.format(re.escape(task_text.strip()) + '.*')
            match = re.search(pattern, all_history)
        if include_task:
            return match.group().lstrip("Now it's your turn!\n") + match.string[match.end():]
        return match.string[match.end():].strip()

    def is_truncated(self) -> bool:
        return self.env.is_truncated() or (self.token_counter(self.log_history(include_all=True)) > 15800)


    def run(self, mode: str, eval_idx: int = None, reset: bool = True):
        # normal training step
        if mode == 'train':
            if self.perform_reflection and not self.is_success():
                self.reflect()

            ################################
            if reset:
                self.env.reset()
                self.reset()

            while not (self.is_truncated() or self.is_terminated()):
                self.step()
            ################################
                
            if self.reflection_counter.is_maximum() or self.is_success():
                self.increment_task = True

    def is_success(self) -> bool:
        return EM(self.answer, self.key)
    
    def prompt_agent(self) -> str:
        self.prompt_history = self.collapse_prompts(self.prompt_history)
        self.update_dynamic_prompt_components()
        prompt_history = self.collapse_prompts(self.prompt_history)
        if self.testing:
            print('###################################')
            for prompt in prompt_history:
                self.print_message(prompt, self.token_counter)
            return input()
        try:
            return self.llm(prompt_history, stop=['\n', '\n\n'])
        except InvalidRequestError:
            while self.long_pass is None:
                res = input('Changing to long context LLM. Press Enter to continue.\n')
                if res == 'pass':
                    self.long_pass = True
                elif res != '':
                    continue
                break

            return self.long_context_llm(prompt_history, stop=['\n', '\n\n'])

    def step(self) -> None:
        message, message_type, others = self.llm_parser(self.prompt_agent(), self.curr_step, False)
        self.prompt_history.append(message)
        self.print_message(message)

        thought_num = 1
        # loops while in thinking mode
        while message_type == 'thought':
            thought_num += 1
            message, message_type, others = self.llm_parser(self.prompt_agent(), self.curr_step, False)
            self.prompt_history.append(message)
            self.print_message(message)

            if thought_num > 2:
                if message_type == 'thought':
                    others['action'] = 'N/A'
                break

        # Observe
        observation, self.reward, self.terminated, self.truncated, _ = self.env.step(others['action'])
        if others['action'] == 'N/A' and thought_num > 2:
            observation = "You are thinking too many times without taking action."
        observation_history, operation = self.observation_formatter(observation, step=self.curr_step)
        if operation == 'append':
            self.prompt_history.append(observation_history)
        elif operation == 'replace':
            for message in self.prompt_history:
                if self._last_observation_history.content in message.content:
                    message.content = message.content.replace(self._last_observation_history.content, observation_history.content)
                    break
            self._last_observation_history = deepcopy(observation_history)        
        self.print_message(observation_history)

        BaseAgent.after_step(self)

        self.prompt_history = self.collapse_prompts(self.prompt_history)

        self.curr_step += 1

        trial = self.prompt_history[self.history_index].content.split(self.remove_task_suffix(self.task), 1)[-1].strip()
        steps = self.message_step_splitter(
            lines=trial,
            cycler=self.message_splitter,
            step_identifier=self.identifier)
        self.previous_trial.append(HumanMessage(content=steps[-1]))

    @property
    def history_index(self) -> int:
        return -1

    def after_step(self) -> None:
        pass

    def _format_reflection_scratchpad(self) -> str:
        lines = [ref.content for ref in self.reflection_prompt_history[self.reflect_interaction_idx:]]
        lines_by_tokens = sorted(lines, key=lambda x: self.token_counter(x))
        while self.token_counter(''.join(lines)) > 12000:
            ind = lines.index(lines_by_tokens.pop(-1))
            line = lines[ind]
            lines[ind]  = line.split(':')[0] + ': ...'
        combined_message = HumanMessage(content='\n'.join(lines))
        self.reflection_prompt_history = self.reflection_prompt_history[:self.reflect_interaction_idx]
        self.reflection_prompt_history.append(combined_message)

    def reflect(self) -> None:
        self._format_reflection_scratchpad()
        self.reflection_prompt_history.append(HumanMessage(content=self.reflection_prefix))
        reflection = self.prompt_reflection()
        self.reflections.append(reflection)
        self.formatted_reflection = self.format_reflections(self.reflections)
        print(self.formatted_reflection)
        # wipe the history for a new round
        self.previous_trial = []


    def prompt_reflection(self) -> str:
        self.reflection_prompt_history = self.collapse_prompts(self.reflection_prompt_history)
        if self.benchmark_name == 'webshop':
            # match the last "Observation:"
            pattern = r"\nObservation: (.*[\n]+)+Next plan:.*"
            matches = re.findall(pattern, self.reflection_prompt_history[-1].content)
            if 'Ran out of steps' in matches[-1]:
                add_text = "\nObservation: Ran out of steps! TASK FAILED\n\nNext plan:\n"
            elif 'Repeated action' in matches[-1]:
                add_text = "\nObservation: Repeated action! TASK FAILED\n\nNext plan:\n"
            else:
                add_text = "\nObservation: Wrong item! TASK FAILED\n\nNext plan:\n"

            new_history = self.reflection_prompt_history[-1].content.split(matches[-1])
            new_history = ''.join(new_history[:-1]) + add_text

            self.reflection_prompt_history[-1].content = new_history

        if self.testing:
            print('###################################')
            for prompt in self.reflection_prompt_history:
                self.print_message(prompt, self.token_counter)
            return input()
        try:
            return self.llm(self.reflection_prompt_history, stop=['\n', '\n\n'])
        except InvalidRequestError:
            return self.long_context_llm(self.reflection_prompt_history, stop=['\n', '\n\n'])

    def reset(self, *args, **kwargs) -> None:
        self.prompt_history = []
        self.update_dynamic_prompt_components(reset=True)
        self.curr_step = 1
        self._build_agent_prompt()
        self.reflection_prompt_history = []
        self._build_reflection_prompt()
        if self.increment_task:
            self.reflections = []
            self.reflection_counter.reset()
            self.formatted_reflection = None
            self.previous_trial = []

    def update_dynamic_prompt_components(self):
        #####################
        # Updating fewshots #
        #####################
        if isinstance(self.all_fewshots, dict):
            self.fewshots = self.all_fewshots[self.env.env_name]
        elif isinstance(self.all_fewshots, list):
            self.fewshots = self.all_fewshots

        #########################
        # Updating instructions #
        #########################
        if isinstance(self.all_system_instruction, str):
            self.system_instruction = self.all_system_instruction
        elif isinstance(self.all_system_instruction, dict):
            self.system_instruction = self.all_system_instruction[self.env.env_name]
        # if system gives instruction, then human instruction is empty
        self.human_instruction_kwargs['instruction'] = ''
        self.num_fewshots = len(self.fewshots)

    def collapse_prompts(self, prompt_history: List[ChatMessage]) -> List[ChatMessage]:
        """Courtesy of GPT4"""
        if not prompt_history:
            return []

        new_prompt_history = []
        scratch_pad = prompt_history[0].content
        last_message_type = type(prompt_history[0])

        for message in prompt_history[1:]:
            current_message_type = type(message)
            if current_message_type == last_message_type:
                scratch_pad += '\n' + message.content
            else:
                new_prompt_history.append(last_message_type(content=scratch_pad))
                scratch_pad = message.content
                last_message_type = current_message_type

        # Handle the last accumulated message
        new_prompt_history.append(last_message_type(content=scratch_pad))

        return new_prompt_history
    
    def insert_before_task_prompt(self) -> None:
        if self.formatted_reflection is not None:
            self.prompt_history.append(HumanMessage(content=self.formatted_reflection))

    def insert_after_task_prompt(self) -> None:
        return
    
    def remove_task_suffix(self, task: str) -> str:
        if self.benchmark_name == 'alfworld':
            return task.split('___')[0]
        return task

    def _build_agent_prompt(self) -> None:
        system_prompt = self.system_prompt.format_messages(
            instruction=self.system_instruction, ai_name=self.name
        )
        self.prompt_history.extend(system_prompt)
        self._build_fewshot_prompt(
            fewshots=self.fewshots, prompt_history=self.prompt_history,
            instruction_prompt=self.human_instruction,
            instruction_prompt_kwargs=self.human_instruction_kwargs,
            prompt_type='react_type',
        )
        self.prompt_history = self.collapse_prompts(self.prompt_history)
        self.log_idx = len(self.prompt_history)
        self.insert_before_task_prompt()

        self.prompt_history.append(human_task_message_prompt.format_messages(task=self.remove_task_suffix(self.task))[0])
        self.insert_after_task_prompt()
        self.prompt_history = self.collapse_prompts(self.prompt_history)
        self.pretask_idx = len(self.prompt_history)
        return self.prompt_history



    def _build_reflection_prompt(self) -> None:
        # avoid building reflection prompt if it already exists
        if self.reflection_prompt_history != []:
            return
        system_prompt = self.system_prompt.format_messages(**self.reflection_system_kwargs)
        self.reflection_prompt_history.extend(system_prompt)
        self._build_fewshot_prompt(
            fewshots=self.reflection_fewshots,
            prompt_history=self.reflection_prompt_history,
            instruction_prompt=self.reflection_task_prompt,
            instruction_prompt_kwargs={},
            prompt_type='reflect_type',
        )
        self.reflection_prompt_history.append(HumanMessage(content=f'Previous trial:\n{self.remove_task_suffix(self.task)}'))
        self.reflect_interaction_idx = len(self.reflection_prompt_history)
        for message in self.previous_trial:
            self.reflection_prompt_history.append(message)


    def _build_fewshot_prompt(
        self,
        fewshots: List[str],
        prompt_history: List[ChatMessage],
        instruction_prompt: PromptTemplate,
        instruction_prompt_kwargs: Dict[str, Any],
        prompt_type: str,
    ) -> str:
        if human_instruction_fewshot_message_prompt is not None and instruction_prompt is not None:
            prompt_history.append(
                human_instruction_fewshot_message_prompt('message_style_kwargs').format_messages(
                    instruction=instruction_prompt.format_messages(
                        **instruction_prompt_kwargs)[0].content,
                    fewshots='\n\n'.join(fewshots)
                )[0]
            )

    def update_stats(self) -> None:
        # only count when finished trying for this task
        if self.increment_task:
            if not self.is_success() and self.is_truncated():
                self.halted += 1
            else:
                if self.reward:
                    self.success += 1
                else:
                    self.fail += 1


    def next_task(self) -> bool:
        # storing reflections
        if self.task not in self.past_reflections:
            self.past_reflections[self.task] = []
        if self.reflections != []:
            self.past_reflections[self.task].append(self.reflections[-1])

        # only reflect on the task if the task is training task
        if self.training:
            # record the tasks
            history = self.log_history(include_task=False)
            trajectory = Trajectory(
                task=self.remove_task_suffix(self.task),
                trajectory=history,
                reflections=self.reflections,
                splitter=self.message_splitter,
                identifier=self.identifier,
                step_splitter=self.message_step_splitter,
            )
            self.succeeded_trial_history = deepcopy(self.succeeded_trial_history)
            self.failed_trial_history = deepcopy(self.failed_trial_history)

            # first time doing the task
            if self.task not in self.failed_trial_history:
                self.succeeded_trial_history[self.task] = []
                self.failed_trial_history[self.task] = []
            # if changing task, reflect accordingly
            if self.increment_task:
                if self.is_success():
                    self.succeeded_trial_history[self.task].append(trajectory)
                else:
                    self.failed_trial_history[self.task].append(trajectory)
                    # record the task index that failed
                    self.failed_training_task_idx.append(self.task_idx)
            else:
                self.failed_trial_history[self.task].append(trajectory)
        
        # increment task if reflection counter is at max OR if the agent is successful
        if self.increment_task:
            self.task_idx += 1
            if self.job_not_done():
                self.task = self.tasks[self.task_idx]['task']
                self.set_env(self.tasks[self.task_idx]['env_kwargs'], max_steps=self.max_steps)
                self.perform_reflection = False
                # wipe the history for a new task
                self.previous_trial = []
        # if there are more tasks, perform reflection
        if self.job_not_done() and not self.increment_task:
            self.perform_reflection = True
            self.reflection_counter.increment()
        self.reset()
        self.env.reset()
        if self.increment_task:
            self.increment_task = False
            return True
        return False
    
    def job_not_done(self) -> bool:
        return self.task_idx < len(self.tasks)
    
    def set_env(self, task_kwargs: Dict[str, Any], max_steps: int):
        self.env.__init__(**task_kwargs, max_steps=max_steps)