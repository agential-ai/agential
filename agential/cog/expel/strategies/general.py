"""ExpeL Agent strategies for QA."""

from typing import Any, Dict, List, Optional, Tuple

from copy import deepcopy
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
from agential.cog.expel.strategies.base import ExpeLBaseStrategy
from agential.cog.reflexion.agent import ReflexionReActAgent
from agential.llm.llm import BaseLLM
from agential.utils.general import shuffle_chunk_list


class ExpeLStrategy(ExpeLBaseStrategy):
    """A general strategy class for the ExpeL agent.

    Attributes:
    llm (BaseLLM): The language model used for generating answers and critiques.
    reflexion_react_agent (ReflexionReActAgent): The ReflexionReAct agent.
    experience_memory (ExpeLExperienceMemory): Memory module for storing experiences. Default is None.
    insight_memory (ExpeLInsightMemory): Memory module for storing insights derived from experiences. Default is None.
    success_batch_size (int): Batch size for processing success experiences in generating insights. Default is 8.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflexion_react_agent: ReflexionReActAgent,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        success_batch_size: int = 8,
    ) -> None:
        """Initialization."""
        experience_memory = experience_memory or ExpeLExperienceMemory()
        insight_memory = insight_memory or ExpeLInsightMemory()
        super().__init__(
            llm,
            reflexion_react_agent,
            experience_memory,
            insight_memory,
            success_batch_size,
        )

        if experience_memory:
            self.extract_insights(self.experience_memory.experiences)

    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        prompt: str,
        reflect_examples: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: Dict[str, Any],
        reflect_additional_keys: Dict[str, Any],
        patience: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generates a response based on the provided question, key, examples, prompt, reflect_examples, reflect_prompt, reflect_strategy, additional_keys, reflect_additional_keys, and patience.

        Args:
            question (str): The question to generate a response for.
            key (str): The key associated with the question.
            examples (str): The examples to use for the generation.
            prompt (str): The prompt to use for the generation.
            reflect_examples (str): The examples to use for the reflection.
            reflect_prompt (str): The prompt to use for the reflection.
            reflect_strategy (str): The strategy to use for the reflection.
            additional_keys (Dict[str, Any]): Additional keys to include in the response.
            reflect_additional_keys (Dict[str, Any]): Additional keys to include in the reflection.
            patience (int): The number of attempts to make before giving up.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: The generated response.
        """
        experiences = self.gather_experience(
            questions=[question],
            keys=[key],
            examples=examples,
            prompt=prompt,
            reflect_examples=reflect_examples,
            reflect_prompt=reflect_prompt,
            reflect_strategy=reflect_strategy,
            additional_keys=[additional_keys],
            reflect_additional_keys=[reflect_additional_keys],
            patience=patience,
            **kwargs,
        )

        return experiences

    def get_dynamic_examples(
        self,
        question: str,
        examples: str,
        k_docs: int,
        num_fewshots: int,
        max_fewshot_tokens: int,
        reranker_strategy: Optional[str],
        additional_keys: Dict[str, Any],
    ) -> Tuple[str, Dict[str, str]]:
        """Dynamically loads relevant past successful trajectories as few-shot examples and insights from the experience and insight memories, and returns the updated examples and additional keys.

        Args:
            question (str): The question to use for loading the relevant past successful trajectories.
            examples (str): The examples to use as a fallback if no dynamic examples are found.
            k_docs (int): The number of relevant past successful trajectories to load.
            num_fewshots (int): The number of few-shot examples to include.
            max_fewshot_tokens (int): The maximum number of tokens to include in the few-shot examples.
            reranker_strategy (Optional[str]): The reranker strategy to use for loading the relevant past successful trajectories.
            additional_keys (Dict[str, Any]): Additional keys to update with the loaded insights.

        Returns:
            Tuple[str, Dict[str, str]]: The updated examples and additional keys.
        """
        additional_keys = additional_keys.copy()

        # Dynamically load in relevant past successful trajectories as fewshot examples.
        dynamic_examples = self.experience_memory.load_memories(
            query=question,
            k_docs=k_docs,
            num_fewshots=num_fewshots,
            max_fewshot_tokens=max_fewshot_tokens,
            reranker_strategy=reranker_strategy,
        )["fewshots"]
        examples = "\n\n---\n\n".join(
            dynamic_examples if dynamic_examples else [examples]  # type: ignore
        )

        # Dynamically load in all insights.
        insights = self.insight_memory.load_memories()["insights"]
        insights = "".join(
            [f"{i}. {insight['insight']}\n" for i, insight in enumerate(insights)]
        )
        additional_keys.update({"insights": insights})

        return examples, additional_keys

    def gather_experience(
        self,
        questions: List[str],
        keys: List[str],
        examples: str,
        prompt: str,
        reflect_examples: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: List[Dict[str, str]],
        reflect_additional_keys: List[Dict[str, str]],
        patience: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Gathers experience data for the Reflexion React agent, including questions, keys, examples, prompts, and additional keys. The gathered experience is added to the experience memory and returned as a dictionary.

        Args:
            questions (List[str]): A list of questions to gather experience for.
            keys (List[str]): A list of keys to associate with the gathered experience.
            examples (str): The examples to use for the experience.
            prompt (str): The prompt to use for the experience.
            reflect_examples (str): The examples to use for the reflection experience.
            reflect_prompt (str): The prompt to use for the reflection experience.
            reflect_strategy (str): The reflection strategy to use.
            additional_keys (List[Dict[str, str]]): Additional keys to associate with the gathered experience.
            reflect_additional_keys (List[Dict[str, str]]): Additional keys to associate with the reflection experience.
            patience (int): The patience to use for the experience gathering.
            **kwargs (Any): Additional keyword arguments to pass to the `gather_experience` function.

        Returns:
            List[Dict[str, Any]]: A list of experience outputs.
        """
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
            questions=[exp["question"] for exp in experiences],
            keys=[exp["key"] for exp in experiences],
            trajectories=[exp["trajectory"] for exp in experiences],
            reflections=[exp["reflections"] for exp in experiences],
        )
        return experiences

    def extract_insights(self, experiences: List[Dict[str, Any]]) -> None:
        """Extracts insights from the provided experiences and updates the `InsightMemory` accordingly.

        This method is responsible for analyzing the successful and failed trials in the provided experiences, comparing them, and generating insights that are then stored in the `InsightMemory`. The insights are generated using the `get_operations_compare` and `get_operations_success` functions, and the `update_insights` method is used to apply the generated operations to the `InsightMemory`.
        The method first categorizes the experiences into "compare" and "success" categories, and then processes the experiences in batches. For the "compare" category, it compares the successful trial with all previous failed trials and generates insights using the `get_operations_compare` function. For the "success" category, it concatenates the successful trials and generates insights using the `get_operations_success` function.

        Args:
            experiences (List[Dict[str, Any]]): A dictionary containing the experiences to be processed, including questions, trajectories, and other relevant data.
        """
        # Extract insights.
        categories = categorize_experiences(experiences)
        folds = get_folds(categories, len(experiences))

        for train_idxs in folds.values():
            train_category_idxs = {
                category: list(set(train_idxs).intersection(set(category_idxs)))  # type: ignore
                for category, category_idxs in categories.items()
            }

            # Compare.
            for train_idx in train_category_idxs["compare"]:
                question = experiences[train_idx]["question"]
                trajectory = experiences[train_idx]["trajectory"]

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
                        f"{experiences[idx]['question']}\n"
                        + "".join(
                            f"Thought: {step.thought}\nAction: {step.action_type}[{step.query}]\nObservation: {step.observation}\n"
                            for step in experiences[idx]["trajectory"][0].react_output
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
        """Updates the insights in the `InsightMemory` based on the provided operations.

        The `operations` parameter is a list of tuples, where each tuple contains an operation type and an insight. The supported operation types are:
        - "REMOVE": Removes the insight from the `InsightMemory`.
        - "AGREE": Increases the score of the insight in the `InsightMemory`.
        - "EDIT": Updates the insight in the `InsightMemory` with the provided insight.
        - "ADD": Adds a new insight to the `InsightMemory` with a score of 2.

        This method is responsible for applying the various operations to the insights stored in the `InsightMemory`.

        Args:
            operations (List[Tuple[str, str]]): A list of tuples, where each tuple contains an operation type and an insight.
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

    def create_output_dict(
        self,
        examples: str,
        additional_keys: Dict[str, str],
        experience: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Creates and returns an output dictionary containing the current state of the agent.

        Args:
            examples (str): The examples to be included in the output.
            additional_keys (Dict[str, str]): Additional key-value pairs to be included in the output.
            experience (List[Dict[str, Any]]): The current experience to be included in the output.

        Returns:
            Dict[str, Any]: A dictionary containing the current state of the agent, including examples, additional keys, and experience.
        """
        output_dict = {
            "examples": examples,
            "insights": additional_keys.get("insights", ""),
            "experience": {
                k: v for k, v in experience[0].items() if k not in ["question", "key"]
            },
            "experience_memory": deepcopy(self.experience_memory.show_memories()),
            "insight_memory": deepcopy(self.insight_memory.show_memories()),
        }
        return output_dict

    def reset(self, only_reflexion: bool = False) -> None:
        """Resets the state of the `ReflexionReactAgent` and clears the `ExperienceMemory` and `InsightMemory` if `only_reflexion` is `False`.

        Args:
            only_reflexion (bool, optional): If `True`, only the `ReflexionReactAgent` is reset. If `False`, the `ExperienceMemory` and `InsightMemory` are also cleared. Defaults to `False`.
        """
        if only_reflexion:
            self.reflexion_react_agent.reset()
        else:
            self.reflexion_react_agent.reset()
            self.experience_memory.clear()
            self.insight_memory.clear()
