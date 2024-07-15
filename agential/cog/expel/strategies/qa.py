"""ExpeL Agent strategies for QA."""

from typing import Optional, Dict, Any, Tuple, List, Union

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.expel.strategies.base import ExpeLBaseStrategy
from agential.cog.reflexion.agent import ReflexionReActAgent
from agential.cog.expel.functional import (
    gather_experience,
    categorize_experiences,
    get_folds,
    get_operations_compare,
    get_operations_success,
    retrieve_insight_index,
)
from agential.utils.general import shuffle_chunk_list


class ExpeLQAStrategy(ExpeLBaseStrategy):
    def __init__(
        self,
        llm: BaseChatModel,
        reflexion_react_agent: ReflexionReActAgent,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        success_batch_size: int = 8,
    ) -> None:
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

    def generate(self) -> str:
        pass

    def get_dynamic_examples(
        self,
        question: str,
        examples: str,
        k_docs: int,
        num_fewshots: int,
        max_fewshot_tokens: int,
        reranker_strategy: str,
        additional_keys: Dict[str, Any],
    ) -> Tuple[str, Dict[str, str]]:
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
        additional_keys: Union[List[Dict[str, str]], Dict[str, str]],
        reflect_additional_keys: Union[List[Dict[str, str]], Dict[str, str]],
        patience: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
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

    def reset(self, only_reflexion: bool) -> None:
        if only_reflexion:
            self.reflexion_react_agent.reset()
        self.experience_memory.clear()
        self.insight_memory.clear()


class ExpeLHotQAStrategy(ExpeLQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ExpeL agent."""

    pass


class ExpeLTriviaQAStrategy(ExpeLQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ExpeL agent."""

    pass


class ExpeLAmbigNQStrategy(ExpeLQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ExpeL agent."""

    pass


class ExpeLFEVERStrategy(ExpeLQAStrategy):
    """A strategy class for the FEVER benchmark using the ExpeL agent."""

    pass
