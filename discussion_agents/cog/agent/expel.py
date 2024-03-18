from typing import Optional, Dict, Any, List

from langchain_core.language_models.chat_models import BaseChatModel

from discussion_agents.cog.modules.memory.expel import ExpeLExperienceMemory, ExpeLInsightMemory
from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.agent.reflexion import ReflexionReActAgent

from discussion_agents.cog.functional.expel import (
    gather_experience,
    categorize_experiences,
    get_folds,
    _prompt_compare_critique,
    parse_insights,
    remove_err_operations,
    get_operations_compare, 
    get_operations_success,
)


class ExpeLAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        self_reflect_llm: BaseChatModel, 
        action_llm: BaseChatModel,
        reflexion_react_kwargs: Optional[Dict[str, Any]] = None,
        reflexion_react_agent: Optional[ReflexionReActAgent] = None,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        max_num_insights: int = 20,
    ) -> None:
        super().__init__()

        self.llm = llm

        if not reflexion_react_agent:
            self.reflexion_react_agent = ReflexionReActAgent(
                self_reflect_llm=self_reflect_llm,
                action_llm=action_llm,
                **reflexion_react_kwargs
            )
        else:
            self.reflexion_react_agent = reflexion_react_agent

        if not experience_memory:
            self.experience_memory = ExpeLExperienceMemory()
        else:
            self.experience_memory = experience_memory

        if not insight_memory:
            self.insight_memory = ExpeLInsightMemory()
        else:
            self.insight_memory = insight_memory

        self.max_num_insights = max_num_insights

    def generate(
        self, 
        question: str, 
        key: str, 
        reflect: bool = True, 
        reset: bool = False,
        reset_reflexion: bool = True,
        strategy: str = "reflexion"
    ):
        if reset_reflexion:
            self.reflexion_react_agent.reset()

        if reset:
            self.reset()

        if reflect:
            self.update_rules()

        # Needs to be changed.
        experience = gather_experience(
            reflexion_react_agent=self.reflexion_react_agent, 
            questions=[question],
            keys=[key],
            strategy=strategy
        )

        self.experience_memory.add_memories(
            questions=experience['questions'],
            keys=experience['keys'],
            trajectories=experience['trajectories'],
            reflections=experience['reflections']
        )

    def update_rules(self) -> None:
        pass

    def reset(self) -> None:
        self.reflexion_react_agent.reset()
        self.experience_memory.clear()
        self.insight_memory.clear()

    def gather_experience(
        self, 
        questions: List[str],
        keys: List[str],
        strategy: Optional[str] = "reflexion"
    ) -> None:
        # Gather experience.
        self.reflexion_react_agent.reset()
        experiences = gather_experience(
            reflexion_react_agent=self.reflexion_react_agent,
            questions=questions,
            keys=keys,
            strategy=strategy
        )
        self.reflexion_react_agent.reset()

        self.experience_memory.add_memories(
            questions=experiences['questions'],
            keys=experiences['keys'],
            trajectories=experiences['trajectories'],
            reflections=experiences['reflections']
        )

        # Extract insights.
        categories = categorize_experiences(self.experience_memory.experiences)
        folds = get_folds(categories, len(self.experience_memory))

        for fold, train_idxs in folds.items():
            # print(fold, train_idxs)
            # rules = create_rules(
            #     llm, 
            #     experiences, 
            #     categories, 
            #     train_idxs, 
            #     rules, 
            #     max_num_rules
            # )

            train_category_idxs = {
                category: list(set(train_idxs).intersection(set(category_idxs)))  # type: ignore
                for category, category_idxs in categories.items()
            }

            # Compare.
            for train_idx in train_category_idxs["compare"]:
                question = experiences["questions"][train_idx]
                trajectory = experiences["trajectories"][
                    train_idx
                ]  # List[Tuple[bool, str, List[Tuple[str, str, str]]]].

                # Compare the successful trial with all previous failed trials.
                success_trial = "\n".join(["\n".join(step) for step in trajectory[-1][-1]])
                for failed_trial in trajectory[:-1]:
                    failed_trial = "\n".join(["\n".join(step) for step in failed_trial[-1]])
                    insights = self.insight_memory.load_memories()['insights']

                    get_operations_compare(
                        llm=self.llm,
                        insights=insights,
                        question=question,
                        success_trial=success_trial,
                        failed_trial=failed_trial,
                        max_num_rules=
                    )

                    # Prompt.
                    out = _prompt_compare_critique(
                        llm=self.llm,
                        insights=insights,
                        question=question,
                        success_trial=success_trial,
                        failed_trial=failed_trial,
                        is_full=self.max_num_insights < len(self.insight_memory),
                    )

                    # Parse.
                    operations = parse_insights(out)

                    # Remove no-ops.
                    operations = remove_err_operations(insights, operations)

                    # Update rules with comparison insights.
