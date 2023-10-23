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
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic.v1 import BaseModel, Field

from discussion_agents.core.memory import BaseCoreWithMemory
from discussion_agents.memory.generative_agents import GenerativeAgentMemory
from discussion_agents.planning.generative_agents import (
    generate_broad_plan,
    update_status,
    generate_refined_plan_step,
)
from discussion_agents.utils.parse import remove_name


class GenerativeAgent(BaseModel):
    """A Generative Agent with memory, innate characteristics, and language model interaction.

    This class serves as a representation of an agent. It includes attributes
    such as the character's name, age, permanent traits, lifestyle characteristics, current
    status, and an agent core with memory capabilities. The core enables interactions with a
    language model, empowering the GenerativeAgent to plan, reflect, summarize, and store memories.

    Attributes:
        name (str): The agent's name.
        age (int): The agent's age.
        traits (str): Permanent traits associated with the agent.
        lifestyle (str): Lifestyle characteristics that remain relatively stable.
        core (BaseCoreWithMemory): The core component for language model interaction and memory management.
    """

    name: str
    age: int
    traits: str
    lifestyle: str
    core: BaseCoreWithMemory

    status: str = ""  #: :meta private:
    summary: str = ""  #: :meta private:
    summary_refresh_seconds: int = 3600  #: :meta private:
    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    plan_req: Dict[str, str] = Field(default_factory=dict)  # : :meta private:

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def generate_broad_plan(
        self,
        instruction: str,
    ) -> List[str]:
        broad_plan = generate_broad_plan(
            instruction=instruction,
            summary=self.summary if self.summary else self.get_summary(),
            core=self.core
        )
        self.plan_req["broad"] = broad_plan

        return broad_plan
    
    def update_status(
        self,
        instruction: str,
        previous_steps: List[str],
        plan_step: str,
    ) -> str:
        new_status = update_status(
            instruction=instruction,
            previous_steps=previous_steps,
            plan_step=plan_step,
            summary=self.summary if self.summary else self.get_summary(),
            status=self.status
        )
        self.status = new_status

        return new_status
    
    def generate_refined_plan_step(
        self,
        instruction: str,
        previous_steps: List[str],
        plan_step: str,
        k: int = 1,
    ) -> List[str]:
        
        refined_plan_step = generate_refined_plan_step(
            instruction=instruction,
            previous_steps=previous_steps,
            plan_step=plan_step,
            summary=self.summary if self.summary else self.get_summary(),
            core=self.core,
            k=k
        )

        self.plan_req["refined_step"] = refined_plan_step

        return refined_plan_step

    def clear_plan(self) -> bool:
        self.plan_req = {}
        return True

        # thought = (
        #     f"This is {self.name}'s plan for {current_day.strftime('%A %B %d, %Y')}:"
        # )
        # for i in self.daily_req:
        #     thought += f" {i},"
        # thought = thought[:-1] + "."

        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: thought,
        #         self.memory.now_key: current_day,
        #     },
        # )

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
            "What is the observed entity in the following observation? {observation}\n"
            + "Entity="
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)
        return chain.run(observation=observation).strip()

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
            "What is the {entity} doing in the following observation? {observation}\n"
            + "The {entity} is"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)
        return chain.run(entity=entity_name, observation=observation).strip()

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
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)
        return chain.run(q1=q1, queries=[q1, q2]).strip()

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
            "{agent_summary_description}\n"
            + "It is {current_time}.\n"
            + "{agent_name}'s lifestyle: {lifestyle}\n"
            + "Summary of relevant context from {agent_name}'s memory:\n"
            + "{relevant_memories}\n"
            + "Most recent observations: {most_recent_memories}\n"
            + "Observation: {observation}\n\n"
            + "{suffix}"
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
            agent_name=self.name,
            lifestyle=self.lifestyle,
            relevant_memories=relevant_memories_str,
            observation=observation,
            suffix=suffix,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

        return chain.run(**kwargs).strip()

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
        return remove_name(text, self.name)

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
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

        # The agent seeks to think about their core characteristics.
        return chain.run(
            name=self.name, queries=[f"{self.name}'s core characteristics"]
        ).strip()

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

        if "broad_schedule" not in self.plan_req:
            self.plan(current_day=current_time)
        return (
            f"Name: {self.name}\n"
            + f"Age: {self.age}\n"
            + f"Innate traits: {self.traits}\n"
            + f"Status: {self.status}\n"
            + f"Lifestyle: {self.lifestyle}\n"
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
