"""Generative Agent implementation from LangChain.

Note: The following classes are versions of LangChain's Generative Agent
implementations with my improvements.

Original Paper: https://arxiv.org/abs/2304.03442
LangChain: https://github.com/langchain-ai/langchain
LangChain Generative Agents:
https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/generative_agents
LangChain Generative Agents Doc Page:
https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/characters
"""
import re
import string
import random

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.pydantic_v1 import BaseModel, Field

from discussion_agents.memory.base_memory import GenerativeAgentMemory


class GenerativeAgent(BaseModel):
    """An Agent as a character with memory and innate characteristics.

    This class represents a character or agent with attributes such as name, age, traits,
    lifestyle, memory, and an underlying language model. It combines innate characteristics
    with the ability to store and retrieve memories and interact with a language model.

    Attributes:
        name (str): The character's name.
        age (int): The optional age of the character. Defaults to None.
        innate_traits (str): Permanent traits ascribed to the character.
        learned_traits (str): Learned traits.
        lifestyle (str): Lifestyle traits of the character that should not change.
        status (str): Current status of the character (what they are currently doing).
        daily_req (str): daily requireements for the agent.
        daily_plan_req (str): Daily plan requirements for the agent.
        memory (GenerativeAgentMemory): The memory object that combines relevance, recency,
            and 'importance' for storing and retrieving memories.
        llm (BaseLanguageModel): The underlying language model used for text generation.
        verbose (bool, optional): A flag to enable verbose mode for debugging and logging.
            Defaults to False.
        summary (str): A stateful self-summary generated via reflection on the character's
            memory. (Private attribute)
        summary_refresh_seconds (int): How frequently to re-generate the summary. (Private attribute)
        last_refreshed (datetime): The last time the character's summary was regenerated.
            (Private attribute)
        daily_summaries (List[str]): Summary of the events in the plan that the agent took.
            (Private attribute)
    """

    name: str
    age: int
    innate_traits: str
    learned_traits: str
    lifestyle: str
    status: str
    daily_req: List[str]
    daily_plan_req: str
    f_daily_schedule: List[str] = []

    memory: GenerativeAgentMemory
    llm: BaseLanguageModel
    verbose: bool = False
    summary: str = ""  #: :meta private:
    summary_refresh_seconds: int = 3600  #: :meta private:
    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def chain(self, prompt: PromptTemplate, llm_kwargs: Dict[str, Any] = {}) -> LLMChain:
        """Create a Large Language Model (LLM) chain for text generation.

        This method creates a Large Language Model (LLM) chain for text generation
        using the specified prompt template. It allows you to chain together multiple
        prompts and interactions with the LLM for more complex text generation tasks.

        Args:
            prompt (PromptTemplate): The prompt template to be used for text generation.

        Returns:
            LLMChain: An instance of the LangChain LLM chain configured with the provided
                prompt template.

        Example usage:
            agent = GenerativeAgent(...)
            prompt_template = PromptTemplate.from_template("Generate a creative story.")
            llm_chain = agent.chain(prompt_template)
            generated_text = llm_chain.run()
            # 'generated_text' contains the text generated using the specified prompt.
        """
        return LLMChain(
            llm=self.llm, llm_kwargs=llm_kwargs, prompt=prompt, verbose=self.verbose, memory=self.memory
        )

    # LLM-related methods
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        r"""Parse a newline-separated string into a list of strings.

        This static method takes a string that contains multiple lines separated by
        newline characters and parses it into a list of strings. It removes any empty
        lines and also removes any leading numbers followed by a period (commonly used
        in numbered lists).

        Args:
            text (str): The input string containing newline-separated lines.

        Returns:
            List[str]: A list of strings parsed from the input text.

        Example usage:
            input_text = "1. Item 1\n2. Item 2\n3. Item 3\n\n4. Item 4"
            parsed_list = GenerativeAgent._parse_list(input_text)
            # 'parsed_list' contains ["Item 1", "Item 2", "Item 3", "Item 4"]

        Note:
            - This method is useful for parsing structured text into a list of items.
            - It removes leading numbers and periods often used in numbered lists.
        """
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def generate_daily_req(
        self, current_day: datetime, wake_up_hour: Optional[int] = 8
    ) -> List[str]:
        prompt = PromptTemplate.from_template(
            "{summary}\n"
            + "In general, {name}'s lifestyle: {lifestyle}\n"
            + "Today is {current_day}. "
            + "Here is {name}'s plan today in broad-strokes "
            + "(with the time of the day. e.g., have lunch at 12:00 pm, watch TV from 7 to 8 pm): "
            + "1) wake up and complete the morning routine at {wake_up_hour}, "
            + "2)\n"
            + "Provide each step on a new line.\n"
        )
        kwargs = dict(
            summary=self.get_summary(),
            lifestyle=self.lifestyle,
            name=self.name,
            current_day=current_day.strftime("%A %B %d"),
            wake_up_hour=wake_up_hour,
        )
        result = self._parse_list(self.chain(prompt).run(**kwargs).strip())
        result = (
            [f"1) wake up and complete the morning routine at {wake_up_hour}:00 am"] + 
            result
        )
        result = [s.split(")")[-1].rstrip(",.").strip() for s in result]

        return result

    def update_status_and_daily_plan_req(self, current_day: datetime) -> None:
        current_day_str = current_day.strftime("%A %B %d, %Y")
        focal_points = [
            f"{self.name}'s plan for {current_day_str}.",
            f"Important recent events for {self.name}'s life."
        ]

        relevant_context = []
        for focal_point in focal_points:
            fetched_memories = self.memory.fetch_memories(focal_point)
            relevant_context.append(
                self.memory.format_memories_detail(fetched_memories)
            )
        relevant_context = "\n".join(relevant_context)

        plan_prompt = PromptTemplate.from_template(
            relevant_context + "\n"
            + "Given the statements above, "
            + "is there anything that {name} should remember as they plan for"
            + " *{current_day_str}*? "
            + "If there is any scheduling information, be as specific as possible "
            + "(include date, time, and location if stated in the statement)\n\n"
            + "Write the response from {name}'s perspective."
        )
        plan_kwargs = dict(
            name=self.name,
            current_day_str=current_day_str
        )
        plan_result = self.chain(prompt=plan_prompt).run(**plan_kwargs).strip()

        thought_prompt = PromptTemplate.from_template(
            relevant_context + "\n"
            + "Given the statements above, how might we summarize "
            + "{name}'s feelings about their days up to now?\n\n"
            + "Write the response from {name}'s perspective."
        )
        thought_kwargs = dict(
            name=self.name
        )
        thought_result = self.chain(prompt=thought_prompt).run(**thought_kwargs).strip()

        status_prompt = PromptTemplate.from_template(
            "{name}'s status from "
            + "{previous_day}:\n"
            + "{status}\n\n"
            + "{name}'s thoughts at the end of "
            + "{previous_day}:\n" 
            + (plan_result + " " + thought_result).replace('\n', '') + "\n\n"
            + "It is now {current_day}. "
            + "Given the above, write {name}'s status for "
            + "{current_day} that reflects {name}'s "
            + "thoughts at the end of {previous_day}. "
            + "Write this in third-person talking about {name}."
            + "If there is any scheduling information, be as specific as possible "
            + "(include date, time, and location if stated in the statement).\n\n"
            + "Follow this format below:\nStatus: <new status>"
        )
        status_kwargs = dict(
            name=self.name,
            previous_day=(current_day - timedelta(days=1)).strftime('%A %B %d, %Y'),
            status=self.status,
            current_day=current_day.strftime('%A %B %d, %Y')
        )
        self.status = self.chain(prompt=status_prompt).run(**status_kwargs).strip()

        daily_plan_req_prompt = PromptTemplate.from_template(
            self.get_summary() + "\n"
            + "Today is {current_day}. "
            + "Here is {name}'s plan today in broad-strokes "
            + "(with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).\n\n"
            + "Follow this format (the list should have 4~6 items but no more):\n"
            + "1. wake up and complete the morning routine at <time>, 2. ..."
        )
        daily_plan_req_kwargs = dict(
            current_day=current_day.strftime('%A %B %d, %Y'),
            name=self.name
        )
        self.daily_plan_req = (
            self.chain(prompt=daily_plan_req_prompt)
            .run(**daily_plan_req_kwargs)
            .strip()
            .replace("\n", " ")
        )

    def rand_id(self, i=6, j=6):
        k = random.randint(i, j)
        hash = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
        return hash

    def generate_hourly_schedule(
            self, n_m1_activity: List[str], curr_hour_str: str, current_day: datetime
        ) -> str:
        hour_str = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", 
                    "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", 
                    "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", 
                    "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
                    "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]
        
        current_day = current_day.strftime("%A %B %d, %Y")
        schedule_format = ""
        for i in hour_str: 
            schedule_format += f"[{current_day} -- {i}]"
            schedule_format += f" Activity: [Fill in]\n"
        schedule_format = schedule_format[:-1]

        intermission_str = f"Here's the originally intended hourly breakdown of"
        intermission_str += f" {self.name}'s schedule today: "
        for count, i in enumerate(self.daily_req): 
            intermission_str += f"{str(count+1)}) {i}, "
        intermission_str = intermission_str[:-2]

        prior_schedule = "\n"
        for count, i in enumerate(n_m1_activity): 
            prior_schedule += f"[(ID:{self.rand_id()})" 
            prior_schedule += f" {current_day} --"
            prior_schedule += f" {hour_str[count]}] Activity:"
            prior_schedule += f" {self.name}"
            prior_schedule += f" is {i}\n"

        prompt = PromptTemplate.from_template(
            schedule_format
            + "==="
            + self.get_summary() + "\n"
            + prior_schedule + "\n"
            + intermission_str + "\n"
            + f"[(ID:{self.rand_id()})"
            + " {current_day}"
            + " -- {curr_hour_str}] Activity:"
            + " {name} is"
        )
        prompt_kwargs = dict(
            current_day=current_day,
            curr_hour_str=curr_hour_str,
            name=self.name
        )
        llm_kwargs = dict(
            max_tokens=50, 
            temperature=0.5,
        )
        result = self.chain(prompt, llm_kwargs=llm_kwargs).run(**prompt_kwargs, stop="\n").rstrip(".").strip()
        return result

    def generate_hourly_schedule_top_3(
            self, current_day: datetime, wake_up_hour: Optional[int] = 8
        ) -> List[str]:
        hour_str = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", 
                    "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", 
                    "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", 
                    "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
                    "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]

        n_m1_activity = []
        diversity_repeat_count = 3
        for i in range(diversity_repeat_count): 
            if len(set(n_m1_activity)) < 5:  # Number of unique activities < 5.
                n_m1_activity = []
                for curr_hour_str in hour_str: 
                    if wake_up_hour > 0: 
                        n_m1_activity += ["sleeping"]
                        wake_up_hour -= 1
                    else: 
                        n_m1_activity += [self.generate_hourly_schedule(n_m1_activity, curr_hour_str, current_day)]

        # Step 1. Compressing the hourly schedule to the following format: 
        # The integer indicates the number of hours. They should add up to 24. 
        # [['sleeping', 6], ['waking up and starting her morning routine', 1], 
        # ['eating breakfast', 1], ['getting ready for the day', 1], 
        # ['working on her painting', 2], ['taking a break', 1], 
        # ['having lunch', 1], ['working on her painting', 3], 
        # ['taking a break', 2], ['working on her painting', 2], 
        # ['relaxing and watching TV', 1], ['going to bed', 1], ['sleeping', 2]]
        _n_m1_hourly_compressed = []
        prev, prev_count = None, 0 
        for i in n_m1_activity: 
            if i != prev:
                prev_count = 1 
                _n_m1_hourly_compressed += [[i, prev_count]]
                prev = i
            elif _n_m1_hourly_compressed: 
                    _n_m1_hourly_compressed[-1][1] += 1

        # Step 2. Expand to min scale (from hour scale)
        # [['sleeping', 360], ['waking up and starting her morning routine', 60], 
        # ['eating breakfast', 60],..
        n_m1_hourly_compressed = []
        for task, duration in _n_m1_hourly_compressed: 
            n_m1_hourly_compressed += [[task, duration*60]]

        return n_m1_hourly_compressed

    def _long_term_planning(self, new_day: str, current_day: datetime, wake_up_hour: int = 8):
        # When it is a new day, we start by creating the daily_req of the persona.
        # Note that the daily_req is a list of strings that describe the persona's
        # day in broad strokes.
        if new_day == "First day": 
            # Bootstrapping the daily plan for the start of then generation:
            # if this is the start of generation (so there is no previous day's 
            # daily requirement, or if we are on a new day, we want to create a new
            # set of daily requirements.
            self.daily_req = self.generate_daily_req(current_day=current_day, wake_up_hour=wake_up_hour)
        elif new_day == "New day":
            self.update_status_and_daily_plan_req(current_day=current_day)

        # Based on the daily_req, we create an hourly schedule for the persona, 
        # which is a list of todo items with a time duration (in minutes) that 
        # add up to 24 hours.
        self.f_daily_schedule = self.generate_hourly_schedule_top_3(current_day=current_day, wake_up_hour=wake_up_hour)

        # Added March 4 -- adding plan to the memory.
        thought = f"This is {self.name}'s plan for {current_day.strftime('%A %B %d, %Y')}:"
        for i in self.daily_req: 
            thought += f" {i},"
        thought = thought[:-1] + "."

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: thought,
                self.memory.now_key: current_day,
            },
        )

    def get_entity_from_observation(self, observation: str) -> str:
        """Extract the observed entity from a given observation text.

        Args:
            observation (str): The observation text from which to extract the entity.

        Returns:
            str: The extracted entity name.

        This method uses a prompt to extract and identify the entity mentioned in the
        observation text.

        Example:
            observation = "A cat is chasing a mouse."
            agent = GenerativeAgent(...)
            entity = agent.get_entity_from_observation(observation)
            print(entity)
        """
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        return self.chain(prompt).run(observation=observation).strip()

    def get_entity_action(self, observation: str, entity_name: str) -> str:
        """Determine the action performed by the specified entity in an observation.

        Args:
            observation (str): The observation text containing the entity's action.
            entity_name (str): The name of the entity whose action to determine.

        Returns:
            str: The action performed by the specified entity.

        This method uses a prompt to identify and describe the action performed by the
        specified entity in the given observation.

        Example:
            observation = "A dog is barking loudly."
            entity = "dog"
            agent = GenerativeAgent(...)
            action = agent.get_entity_action(observation, entity)
            print(action)
        """
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    def summarize_related_memories(self, observation: str) -> str:
        """Generate a summary of memories most relevant to an observation.

        Args:
            observation (str): The observation for which to summarize related memories.

        Returns:
            str: A summary of the most relevant memories in the context of the observation.

        This method generates a summary by posing questions about the relationship between
        the character represented by this `GenerativeAgent` and the entity mentioned in
        the observation. It then incorporates relevant context from memory to provide a
        coherent summary.

        Example:
            observation = "Alice met a friendly cat in the park."
            agent = GenerativeAgent(...)
            summary = agent.summarize_related_memories(observation)
            print(summary)
        """
        prompt = PromptTemplate.from_template(
            "{q1}?\n"
            + "Context from memory:\n"
            + "{relevant_memories}\n"
            + "Relevant context:\n"
        )
        entity_name = self.get_entity_from_observation(observation)
        entity_action = self.get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()

    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """Generate a reaction or response to a given observation or dialogue act.

        Args:
            observation (str): The observation or dialogue act to react to.
            suffix (str): The suffix to append to the generated reaction; a call-to-action.
            now (Optional[datetime], optional): The timestamp for the current time.
                Defaults to None.

        Returns:
            str: The generated reaction or response.

        This helper method generates a reaction or response by providing contextual information
        about the agent, including a summary, current time, relevant memories, and
        recent observations. It then adds the provided `suffix` to create a complete
        reaction or response.

        Example:
            observation = "Alice: Hello, how are you?"
            suffix = "GenerativeAgent: Hello, Alice! I'm doing well, thank you."
            agent = GenerativeAgent(...)
            reaction = agent._generate_reaction(observation, suffix)
            print(reaction)
        """
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s lifestyle: {lifestyle}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            lifestyle=self.lifestyle,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def _clean_response(self, text: str) -> str:
        """Clean the response text by removing the agent's name prefix.

        Args:
            text (str): The response text to clean.

        Returns:
            str: The cleaned response text with the agent's name prefix removed.

        This method is used to remove the agent's name prefix from the response text
        if it exists. This can be useful when presenting the response in a dialogue
        format where the agent's name prefix is not needed.

        Example:
            response = "GenerativeAgent: Hello! How can I help you?"
            agent = GenerativeAgent(...)
            cleaned_response = agent._clean_response(response)
            print(cleaned_response)
        """
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

    def generate_reaction(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """Generate a reaction to a given observation.

        Args:
            observation (str): The observation or input to react to.
            now (Optional[datetime], optional): The current timestamp for the reaction.
                Defaults to None.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean value indicating whether the
            agent should respond (True) or not (False), and the generated reaction or
            response text.

        This method generates a reaction or response to a given observation. It uses a
        template that instructs the agent on how to react to the observation. The
        agent can choose to react, say something, or do nothing based on the
        instructions in the template.

        Example:
            observation = "A user asked a question."
            agent = GenerativeAgent(...)
            should_respond, reaction = agent.generate_reaction(observation)
            if should_respond:
                print(f"{self.name} said: {reaction}")
            else:
                print(f"{self.name} chose not to respond.")
        """
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """Generate a dialogue response to a given observation.

        Args:
            observation (str): The observation or input to respond to.
            now (Optional[datetime], optional): The current timestamp for the response.
                Defaults to None.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean value indicating whether the
            agent should continue the dialogue (True) or end it (False), and the
            generated dialogue response text.

        This method generates a dialogue response to a given observation. It uses a
        template that instructs the agent on how to respond to the observation, either
        by ending the conversation or continuing it. The agent can choose to say
        something in response or end the conversation based on the instructions in the
        template.

        Example:
            observation = "User: Hello, how are you?"
            agent = GenerativeAgent(...)
            should_continue, response = agent.generate_dialogue_response(observation)
            if should_continue:
                print(f"{self.name} said: {response}")
            else:
                print(f"{self.name} said: {response} (End of conversation)")
        """
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {farewell}",
                    self.memory.now_key: now,
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {response_text}",
                    self.memory.now_key: now,
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    ######################################################
    # Agent stateful' summary methods.                   #
    # Each dialog or response prompt includes a header   #
    # summarizing the agent's self-description. This is  #
    # updated periodically through probing its memories  #
    ######################################################
    def compute_agent_summary(self) -> str:
        """Compute a summary of the agent's core characteristics based on relevant memories.

        Returns:
            str: A summary of the agent's core characteristics.

        This method generates a summary of the agent's core characteristics based on
        relevant memories stored in the agent's memory. It uses a template to instruct
        the agent to think about and summarize its core characteristics, focusing on
        what has been learned from previous observations and experiences.

        Example:
            agent = GenerativeAgent(...)
            summary = agent.compute_agent_summary()
            print(f"{agent.name}'s Core Characteristics: {summary}")
        """
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )

    def get_summary(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a descriptive summary of the agent.

        Args:
            force_refresh (bool, optional): If True, force a refresh of the summary
                even if it was recently computed. Defaults to False.
            now (datetime, optional): The current datetime to use for refreshing
                the summary. Defaults to None, which uses the current system time.

        Returns:
            str: A descriptive summary of the agent.

        This method returns a descriptive summary of the agent, including the agent's
        name, age (if available), innate traits, and a summary of the agent's core
        characteristics based on relevant memories. The summary is refreshed
        periodically, but it can be forced to refresh by setting `force_refresh` to True.

        Example:
            agent = GenerativeAgent(...)
            summary = agent.get_summary()
            print(summary)
        """
        current_time = datetime.now() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self.compute_agent_summary()
            self.last_refreshed = current_time
        return (
            f"Name: {self.name}\n"
            + f"Age: {self.age}\n"
            + f"Innate traits: {self.innate_traits}\n"
            + f"Learned traits: {self.learned_traits}\n"
            + f"Status: {self.status}\n"
            + f"Lifestyle: {self.lifestyle}\n"
            + f"Daily plan requirement: {self.daily_plan_req}\n"
            + f"Current Date: {current_time.strftime('%A %B %d')}\n"
            + f"{self.summary}\n"
        )

    def get_full_header(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a full header of the agent's lifestyle, summary, and current time.

        Args:
            force_refresh (bool, optional): If True, force a refresh of the summary
                even if it was recently computed. Defaults to False.
            now (datetime, optional): The current datetime to use for refreshing
                the summary. Defaults to None, which uses the current system time.

        Returns:
            str: A full header including the agent's lifestyle, summary, and current time.

        This method returns a full header that includes the agent's lifestyle, a descriptive
        summary of the agent (which can be refreshed using `force_refresh`), and the
        current time in a formatted string.

        Example:
            agent = GenerativeAgent(...)
            header = agent.get_full_header()
            print(header)
        """
        now = datetime.now() if now is None else now
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return f"{summary}\nIt is {current_time_str}.\n{self.name}'s lifestyle: {self.lifestyle}"
