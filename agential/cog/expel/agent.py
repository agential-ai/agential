"""ExpeL Agent.

Original Paper: https://arxiv.org/pdf/2308.10144.pdf
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.agent import BaseAgent
from agential.cog.reflexion.agent import ReflexionReActAgent
from agential.cog.expel.functional import (
    categorize_experiences,
    gather_experience,
    get_folds,
    get_operations_compare,
    get_operations_success,
    retrieve_insight_index,
)
from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.expel.prompts import (
    END_OF_EXAMPLES_DELIMITER,
    EXPEL_REFLEXION_REACT_INSTRUCTION,
    RULE_PREFIX,
)
from agential.cog.reflexion.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.prompts.benchmark.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.utils.general import shuffle_chunk_list


class ExpeLAgent(BaseAgent):
    """Implements ExpeL, a reflective, experiential learning agent.

    Attributes:
        llm (BaseChatModel): Primary language model for general tasks.
        reflexion_react_kwargs (Optional[Dict[str, Any]]): Configuration options for the ReflexionReAct agent.
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
        mode: Dict[str, str] = {},
        reflexion_react_strategy_kwargs: Dict[str, Any] = {
            "max_steps": 7,
            "max_trials": 3,
        },
        reflexion_react_agent: Optional[ReflexionReActAgent] = None,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        success_batch_size: int = 8,
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm

        if not reflexion_react_agent:
            self.reflexion_react_agent = ReflexionReActAgent(
                llm=llm,
                mode=mode,
                **reflexion_react_strategy_kwargs,
            )
        else:
            self.reflexion_react_agent = reflexion_react_agent

        self.success_batch_size = success_batch_size

        if not insight_memory:
            self.insight_memory = ExpeLInsightMemory()
        else:
            self.insight_memory = insight_memory

        if not experience_memory:
            self.experience_memory = ExpeLExperienceMemory()
        else:
            self.experience_memory = experience_memory
            self.extract_insights(self.experience_memory.experiences)

    def generate(
        self,
        question: str,
        key: str,
        should_extract_insights: bool = True,
        prompt: str = EXPEL_REFLEXION_REACT_INSTRUCTION,
        examples: Optional[str] = None,
        reflect_examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        reflect_prompt: str = REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy: str = "reflexion",
        additional_keys: Union[List[Dict[str, str]], Dict[str, str]] = {},
        reflect_additional_keys: Union[List[Dict[str, str]], Dict[str, str]] = {},
        patience: int = 1,
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
            should_extract_insights (bool): Whether to extract insights from the experiences. Defaults to True.
            prompt (str): The initial prompt or instruction to guide the ReflexionReAct agent's process.
            examples (Optional[str]): Examples to provide context or guidance for the ReflexionReAct agent.
            reflect_examples (str): Examples specifically for the reflection phase of processing.
            reflect_prompt (str): The prompt or instruction guiding the reflection process.
            reflect_strategy (Optional[str]): The strategy to use for processing questions. Defaults to "reflexion".
            additional_keys (Union[List[Dict[str, str]], Dict[str, str]]): The additional keys. Defaults to {}.
            reflect_additional_keys (Union[List[Dict[str, str]], Dict[str, str]]): Additional keys for the reflection phase. Defaults to {}.
            patience (int): The number of times to retry the agent's process if it fails. Defaults to 1.
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
            self.reflexion_react_agent.reset()

        if reset:
            self.reset()

        # User has ability to override examples.
        if not examples:
            # Dynamically load in relevant past successful trajectories as fewshot examples.
            examples = self.experience_memory.load_memories(
                query=question,
                k_docs=k_docs,
                num_fewshots=num_fewshots,
                max_fewshot_tokens=max_fewshot_tokens,
                reranker_strategy=reranker_strategy,
            )["fewshots"]
            examples = (
                examples if examples else [HOTPOTQA_FEWSHOT_EXAMPLES_REACT]  # type: ignore
            )
            examples = "\n\n".join(examples + [END_OF_EXAMPLES_DELIMITER]) + "\n"  # type: ignore

            # Dynamically load in all insights.
            examples += RULE_PREFIX
            insights = self.insight_memory.load_memories()["insights"]
            insights = "".join(
                [f"{i}. {insight['insight']}\n" for i, insight in enumerate(insights)]
            )
            examples += insights

        experience = self.gather_experience(
            questions=[question],
            keys=[key],
            examples=examples,  # type: ignore
            prompt=prompt,
            reflect_examples=reflect_examples,
            reflect_prompt=reflect_prompt,
            reflect_strategy=reflect_strategy,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
            patience=patience,
            **kwargs,
        )

        if should_extract_insights:
            self.extract_insights(experience)

        return experience

    def reset(self) -> None:
        """Resets the agent's state.

        This method clears the memory modules and resets the state of the ReflexionReAct agent,
        the experience memory, and the insight memory.
        """
        self.reflexion_react_agent.reset()
        self.experience_memory.clear()
        self.insight_memory.clear()

    def gather_experience(
        self,
        questions: List[str],
        keys: List[str],
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt: str = REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        reflect_examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        reflect_prompt: str = REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        reflect_strategy: str = "reflexion",
        additional_keys: Union[List[Dict[str, str]], Dict[str, str]] = {},
        reflect_additional_keys: Union[List[Dict[str, str]], Dict[str, str]] = {},
        patience: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Collects and stores experiences from interactions based on specified questions and keys.

        This method invokes the ReflexionReAct agent to process a set of questions with corresponding keys,
        using the provided strategy, prompts, and examples. It captures the trajectories of the agent's reasoning
        and reflection process, storing them for future analysis and insight extraction.

        Parameters:
            questions (List[str]): A list of questions for the agent to process.
            keys (List[str]): Corresponding keys to the questions, used for internal tracking and analysis.
            examples (str, optional): Fewshot examples. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_REACT.
            prompt (str, optional): Prompt template string. Defaults to REFLEXION_REACT_INSTRUCTION_HOTPOTQA.
            reflect_examples (str, optional): Reflection fewshot examples. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT.
            reflect_prompt (str, optional): Reflect prompt template string. Defaults to REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA.
            reflect_strategy (str, optional): The reflection strategy. Can be of 3 types. Defaults to None.
                - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
                - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
                - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
            additional_keys (List[Dict[str, str]], Dict[str, str]): Additional keys for the prompt. Defaults to {}.
            reflect_additional_keys (List[Dict[str, str]], Dict[str, str]): Additional keys for the reflect prompt. Defaults to {}.
            patience (int, optional): The patience for the agent. Defaults to 1.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the collected experiences, including questions, keys, trajectories,
            and reflections.
        """
        # Gather experience.
        experiences = gather_experience(
            reflexion_react_agent=self.reflexion_react_agent,
            questions=questions,
            keys=keys,
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
        self.reflexion_react_agent.reset()

        self.experience_memory.add_memories(
            questions=experiences["questions"],
            keys=experiences["keys"],
            trajectories=experiences["trajectories"],
            reflections=experiences["reflections"],
        )

        return experiences

    def extract_insights(self, experiences: Dict[str, Any]) -> None:
        """Analyzes stored experiences to extract and store insights.

        This method categorizes experiences and applies different strategies to extract actionable insights from them.
        Insights are derived from the comparison of successful and unsuccessful trajectories, as well as
        successful trajectories.

        Parameters:
            experiences (Dict[str, Any]): A dictionary containing the experiences to be analyzed, structured
            with keys: idxs, questions, keys, trajectories, and reflections.
        """
        # Extract insights.
        categories = categorize_experiences(experiences)
        folds = get_folds(categories, len(experiences["idxs"]))

        for train_idxs in folds.values():
            train_category_idxs = {
                category: list(set(train_idxs).intersection(set(category_idxs)))  # type: ignore
                for category, category_idxs in categories.items()
            }

            # Compare.
            for train_idx in train_category_idxs["compare"]:
                question = experiences["questions"][train_idx]
                trajectory = experiences["trajectories"][
                    train_idx
                ]  # List[Dict[str, Any]].

                # Compare the successful trial with all previous failed trials.
                success_trial = "".join(
                    f"Thought: {step.thought}\nAction: {step.action_type}[{step.query}]\nObservation: {step.observation}\n"
                    for step in trajectory[-1].react_output
                )
                for failed_trial in trajectory[:-1]:
                    failed_trial = "".join(
                        f"Thought: {step.thought}\nAction: {step.action_type}[{step.query}]\nObservation: {step.observation}\n"
                        for step in failed_trial.react_output
                    )
                    insights = self.insight_memory.load_memories()["insights"]

                    operations = get_operations_compare(
                        llm=self.llm,
                        insights=insights,
                        question=question,
                        success_trial=success_trial,
                        failed_trial=failed_trial,
                        is_full=self.insight_memory.max_num_insights < len(insights),
                    )
                    self.update_insights(operations=operations)

            # Success.
            if train_category_idxs["success"]:
                batched_success_trajs_idxs = shuffle_chunk_list(
                    train_category_idxs["success"], self.success_batch_size
                )
                for success_idxs in batched_success_trajs_idxs:
                    insights = self.insight_memory.load_memories()["insights"]

                    # Concatenate batched successful trajectories.
                    concat_success_trajs = [
                        f"{experiences['questions'][idx]}\n"
                        + "".join(
                            f"Thought: {step.thought}\nAction: {step.action_type}[{step.query}]\nObservation: {step.observation}\n"
                            for step in experiences["trajectories"][idx][0].react_output
                        )
                        for idx in success_idxs
                    ]

                    success_trials = "\n\n".join(concat_success_trajs)

                    operations = get_operations_success(
                        llm=self.llm,
                        success_trials=success_trials,
                        insights=insights,
                        is_full=self.insight_memory.max_num_insights < len(insights),
                    )
                    self.update_insights(operations=operations)

    def update_insights(self, operations: List[Tuple[str, str]]) -> None:
        """Updates the agent's stored insights.

        This method processes a list of operations (e.g., ADD, REMOVE, EDIT, AGREE) on the insights derived from comparing
        new experiences against the stored ones. It allows the agent to refine its understanding and strategies
        for future interactions.

        Parameters:
            operations (List[Tuple[str, str]]): A list of tuples, each containing an operation type (ADD, REMOVE, EDIT, AGREE)
            and the insight or modification related to that operation.
        """
        # Update rules with comparison insights.
        for i in range(len(operations)):
            insights = self.insight_memory.load_memories()["insights"]
            operation, operation_insight = operations[i]
            operation_type = operation.split(" ")[0]

            if operation_type == "REMOVE":
                insight_idx = retrieve_insight_index(insights, operation_insight)
                if insight_idx != -1:
                    self.insight_memory.delete_memories(insight_idx)
            elif operation_type == "AGREE":
                insight_idx = retrieve_insight_index(insights, operation_insight)
                if insight_idx != -1:
                    self.insight_memory.update_memories(
                        idx=insight_idx, update_type="AGREE"
                    )
            elif operation_type == "EDIT":
                insight_idx = int(operation.split(" ")[1])
                self.insight_memory.update_memories(
                    idx=insight_idx,
                    update_type="EDIT",
                    insight=operation_insight,
                )
            elif operation_type == "ADD":
                self.insight_memory.add_memories(
                    [{"insight": operation_insight, "score": 2}]
                )

    def retrieve(self) -> Dict[str, Any]:
        """Retrieves the current state of the agent's memories: experiences and insights.

        Returns:
            Dict[str, Any]: A dictionary containing five keys, 'experiences', 'success_traj_docs',
                'vectorstore', and 'insights'.
        """
        return {
            **self.experience_memory.show_memories(),
            **self.insight_memory.show_memories(),
        }
