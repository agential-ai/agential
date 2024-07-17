"""ExpeL Agent.

Original Paper: https://arxiv.org/pdf/2308.10144.pdf
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.agent import BaseAgent
from agential.cog.expel.factory import EXPEL_BENCHMARK_FEWSHOTS, ExpeLFactory
from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.reflexion.agent import ReflexionReActAgent


class ExpeLAgent(BaseAgent):
    """Implements ExpeL, a reflective, experiential learning agent.

    Attributes:
        llm (BaseChatModel): Primary language model for general tasks.
        benchmark (str): The benchmark name.
        reflexion_react_strategy_kwargs (Dict[str, Any]): Configuration options for the ReflexionReAct agent.
            Defaults max_steps=7 and max_trials=3 for the ReflexionReActAgent.
        reflexion_react_agent (Optional[ReflexionReActAgent]): The ReflexionReAct agent. Optional.
        experience_memory (Optional[ExpeLExperienceMemory]): Memory module for storing experiences.
        insight_memory (Optional[ExpeLInsightMemory]): Memory module for storing insights derived from experiences.
        success_batch_size (int): Batch size for processing success experiences in generating insights.

    Methods:
        generate(question, key): Generates a response based on a given question and key, potentially extracting insights and applying self-reflection in the process.
        reset(): Resets the agent's state for a new problem-solving session, clearing memory modules and the ReAct agent's state.
        gather_experience(questions, keys): Collects experiences from interactions, storing them for future reference and insight extraction.
        extract_insights(experiences): Analyzes stored experiences to extract and store insights for improving future interactions.
        update_insights(operations): Updates the stored insights based on the analysis of new experiences.
        retrieve(): Retrieves the current state of the agent's memories, including both experiences and insights.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        benchmark: str,
        reflexion_react_agent: Optional[ReflexionReActAgent] = None,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        reflexion_react_strategy_kwargs: Dict[str, Any] = {
            "max_steps": 7,
            "max_trials": 3,
        },
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm
        self.benchmark = benchmark
        reflexion_react_agent = reflexion_react_agent or ReflexionReActAgent(
            llm=llm, benchmark=benchmark, **reflexion_react_strategy_kwargs
        )

        self.strategy = ExpeLFactory().get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            reflexion_react_agent=reflexion_react_agent,
            experience_memory=experience_memory,
            insight_memory=insight_memory,
            **strategy_kwargs,
        )

    def generate(
        self,
        question: str,
        key: str,
        prompt: str = "",
        examples: str = "",
        reflect_examples: str = "",
        reflect_prompt: str = "",
        reflect_strategy: str = "reflexion",
        additional_keys: Dict[str, str] = {},
        reflect_additional_keys: Dict[str, str] = {},
        use_dynamic_examples: bool = True,
        extract_insights: bool = True,
        patience: int = 3,
        k_docs: int = 24,
        num_fewshots: int = 6,
        max_fewshot_tokens: int = 1500,
        reranker_strategy: Optional[str] = None,
        reset_reflexion: bool = True,
        reset: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Collects and stores experiences from interactions based on specified questions and strategies.

        This method invokes the ReflexionReAct agent to process a set of questions with corresponding keys,
        using the provided strategy, prompts, and examples. It captures the trajectories of the agent's reasoning
        and reflection process, storing them for future analysis and insight extraction.

        Parameters:
            questions (List[str]): A list of questions for the agent to process.
            keys (List[str]): Corresponding keys to the questions, used for internal tracking and analysis.
            prompt (str): The initial prompt or instruction to guide the ReflexionReAct agent's process. Defaults to "".
            examples (str): Examples to provide context or guidance for the ReflexionReAct agent. Defaults to "".
            reflect_examples (str): Examples specifically for the reflection phase of processing. Defaults to "".
            reflect_prompt (str): The prompt or instruction guiding the reflection process. Defaults to "".
            reflect_strategy (Optional[str]): The strategy to use for processing questions. Defaults to "reflexion".
            additional_keys (Dict[str, str]): The additional keys. Defaults to {}.
            reflect_additional_keys (Dict[str, str]): Additional keys for the reflection phase. Defaults to {}.
            use_dynamic_examples (bool): A boolean specifying whether or not to use dynamic examples from ExpeL's memory. Defaults to True.
            extract_insights (bool): Whether to extract insights from the experiences. Defaults to True.
            patience (int): The number of times to retry the agent's process if it fails. Defaults to 3.
            k_docs (int): The number of documents to retrieve for the fewshot. Defaults to 24.
            num_fewshots (int): The number of examples to use for the fewshot. Defaults to 6.
            max_fewshot_tokens (int): The maximum number of tokens to use for the fewshot. Defaults to 1500.
            reranker_strategy (Optional[str]): The strategy to use for re-ranking the retrieved. Defaults to None.
            reset_reflexion (bool): Whether to reset the ReflexionReAct agent. Defaults to True.
            reset (bool): Whether to reset the agent's state for a new problem-solving session. Defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the collected experiences, including questions, keys, trajectories,
            and reflections.
        """
        if reset_reflexion:
            self.strategy.reset(only_reflexion=True)

        if reset:
            self.reset()

        # User has ability to override examples.
        if use_dynamic_examples:
            examples, additional_keys = self.strategy.get_dynamic_examples(
                question=question,
                examples=examples,
                k_docs=k_docs,
                num_fewshots=num_fewshots,
                max_fewshot_tokens=max_fewshot_tokens,
                reranker_strategy=reranker_strategy,
                additional_keys=additional_keys,
            )

        experience = self.strategy.generate(
            question=question,
            key=key,
            examples=examples,
            prompt=prompt,
            reflect_examples=reflect_examples,
            reflect_prompt=reflect_prompt,
            reflect_strategy=reflect_strategy,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
            patience=patience,
            **kwargs,
        )

        if extract_insights:
            self.strategy.extract_insights(experience)

        return experience

    def reset(self) -> None:
        """Resets the agent's state.

        This method clears the memory modules and resets the state of the ReflexionReAct agent,
        the experience memory, and the insight memory.
        """
        self.strategy.reset()
