from typing import Any, Dict, Optional, List, Tuple
import time
from copy import deepcopy
from agential.core.llm import BaseLLM, Response
from agential.agents.expel.memory import ExpeLExperienceMemory, ExpeLInsightMemory
from agential.agents.expel.utils import (
    gather_experience,
    accumulate_metrics,
    categorize_experiences,
    get_folds,
    parse_insights,
    remove_err_operations,
    retrieve_insight_index,
    _prompt_compare_critique,
    _prompt_all_success_critique,
)
from agential.utils.general import shuffle_chunk_list

class ExpeLAgent:
    def __init__(
        self,
        llm: BaseLLM,
        reflexion_react_agent,
        config: dict = {},
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        success_batch_size: int = 8,
        extract_init_insights: bool = True,
        testing: bool = False,
        **kwargs
    ):
        self.llm = llm
        self.reflexion_react_agent = reflexion_react_agent
        self.config = config
        self.experience_memory = experience_memory or ExpeLExperienceMemory()
        self.insight_memory = insight_memory or ExpeLInsightMemory()
        self.success_batch_size = success_batch_size
        self.extract_init_insights = extract_init_insights and self.experience_memory.experiences != []
        self.testing = testing
        self.kwargs = kwargs

    def generate(
        self,
        question: str,
        key: str = "",
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
        reset: bool = False,
    ) -> dict:
        start = time.time()
        compares_response: List[List[Any]] = []
        successes_response: List[List[Any]] = []
        examples = self.config["fewshot"]
        reflect_examples = self.config["reflect_fewshot"]
        prompt = self.config["prompt"]
        reflect_prompt = self.config["reflect_prompt"]

        if self.extract_init_insights:
            compare_response, success_response = self.extract_insights(
                self.experience_memory.experiences
            )
            compares_response.append(compare_response)
            successes_response.append(success_response)
            self.extract_init_insights = False

        if reset:
            self.reset()

        if use_dynamic_examples:
            examples, additional_keys = self.get_dynamic_examples(
                question=question,
                examples=examples,
                k_docs=k_docs,
                num_fewshots=num_fewshots,
                max_fewshot_tokens=max_fewshot_tokens,
                reranker_strategy=reranker_strategy,
                additional_keys=additional_keys,
            )
        else:
            additional_keys = additional_keys.copy()
            additional_keys.update({"insights": ""})

        if self.reflexion_react_agent is None:
            raise ValueError("You must provide a ReflexionReAct agent as 'reflexion_react_agent' to ExpeLAgent.")
        experience: List[Dict[str, Any]] = self.gather_experience(
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
        )

        if extract_insights:
            compare_response, success_response = self.extract_insights(experience)
            compares_response.append(compare_response)
            successes_response.append(success_response)

        # Compose output
        total_time = time.time() - start
        total_metrics = accumulate_metrics(
            compares_response=compares_response,
            successes_response=successes_response,
            experiences=experience,
        )
        # Compose answer and steps using new dict-based output
        answer = ""
        if experience and "trajectory" in experience[0]:
            traj = experience[0]["trajectory"]
            # Use the answer from the trajectory dict
            answer = traj.get("answer", "")
        out = {
            "answer": answer,
            "experience": {k: v for k, v in experience[0].items() if k not in ["question", "key"]} if experience else {},
            "experience_memory": deepcopy(self.experience_memory.show_memories()),
            "insight_memory": deepcopy(self.insight_memory.show_memories()),
            "metrics": {
                "total_time": total_time if not self.testing else 0.5,
                "total_prompt_tokens": total_metrics["total_prompt_tokens"],
                "total_completion_tokens": total_metrics["total_completion_tokens"],
                "total_tokens": total_metrics["total_tokens"],
                "total_prompt_cost": total_metrics["total_prompt_cost"],
                "total_completion_cost": total_metrics["total_completion_cost"],
                "total_cost": total_metrics["total_cost"],
                "total_prompt_time": total_metrics["total_prompt_time"],
            },
            "compares_response": compares_response if extract_insights else None,
            "successes_response": successes_response if extract_insights else None,
        }
        return out

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
        additional_keys = additional_keys.copy()
        dynamic_examples = self.experience_memory.load_memories(
            query=question,
            k_docs=k_docs,
            num_fewshots=num_fewshots,
            max_fewshot_tokens=max_fewshot_tokens,
            reranker_strategy=reranker_strategy,
        )["fewshots"]
        examples = "\n\n---\n\n".join(dynamic_examples if dynamic_examples else [examples])
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
    ) -> List[Dict[str, Any]]:
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
        )
        self.experience_memory.add_memories(
            questions=[exp["question"] for exp in experiences],
            keys=[exp["key"] for exp in experiences],
            trajectories=[exp["trajectory"] for exp in experiences],
            reflections=[exp["reflections"] for exp in experiences],
        )
        return experiences

    def extract_insights(
        self, experiences: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[Any]]:
        # Use the new dict-based output structure for trajectory and steps
        categories = categorize_experiences(experiences)
        folds = get_folds(categories, len(experiences))
        compares_response: List[Any] = []
        successes_response: List[Any] = []
        for train_idxs in folds.values():
            train_category_idxs = {
                category: list(set(train_idxs).intersection(set(category_idxs)))
                for category, category_idxs in categories.items()
            }
            # Compare
            for train_idx in train_category_idxs["compare"]:
                question = experiences[train_idx]["question"]
                trajectory = experiences[train_idx]["trajectory"]
                # Use the last trial's steps for the successful trial
                success_trial = ""
                if trajectory["trials"]:
                    last_trial = trajectory["trials"][-1]
                    success_trial = "".join(
                        f"Thought: {step['thought']}\nAction: {step['action_type']}[{step['query']}]\nObservation: {step['observation']}\n"
                        for step in last_trial["steps"]
                    )
                for failed_trial in trajectory["trials"][:-1]:
                    failed_trial_str = "".join(
                        f"Thought: {step['thought']}\nAction: {step['action_type']}[{step['query']}]\nObservation: {step['observation']}\n"
                        for step in failed_trial["steps"]
                    )
                    insights = self.insight_memory.load_memories()["insights"]
                    compare_out = _prompt_compare_critique(
                        llm=self.llm,
                        insights=insights,
                        question=question,
                        success_trial=success_trial,
                        failed_trial=failed_trial_str,
                        is_full=self.insight_memory.max_num_insights < len(insights),
                    )
                    compares_response.append(compare_out)
                    insights_str = compare_out.output_text.strip("\n").strip()
                    operations = parse_insights(insights_str)
                    operations = remove_err_operations(insights, operations)
                    self.update_insights(operations=operations)
            # Success
            if train_category_idxs["success"]:
                batched_success_trajs_idxs = shuffle_chunk_list(
                    train_category_idxs["success"], self.success_batch_size
                )
                for success_idxs in batched_success_trajs_idxs:
                    insights = self.insight_memory.load_memories()["insights"]
                    concat_success_trajs = []
                    for idx in success_idxs:
                        traj = experiences[idx]["trajectory"]
                        if traj["trials"]:
                            trial = traj["trials"][0]
                            steps_str = "".join(
                                f"Thought: {step['thought']}\nAction: {step['action_type']}[{step['query']}]\nObservation: {step['observation']}\n"
                                for step in trial["steps"]
                            )
                            concat_success_trajs.append(f"{experiences[idx]['question']}\n" + steps_str)
                    success_trials = "\n\n".join(concat_success_trajs)
                    success_out = _prompt_all_success_critique(
                        llm=self.llm,
                        insights=insights,
                        success_trajs_str=success_trials,
                        is_full=self.insight_memory.max_num_insights < len(insights),
                    )
                    successes_response.append(success_out)
                    insights_str = success_out.output_text.strip("\n").strip()
                    operations = parse_insights(insights_str)
                    operations = remove_err_operations(insights, operations)
                    self.update_insights(operations=operations)
        return compares_response, successes_response

    def update_insights(self, operations: List[Tuple[str, str]]) -> None:
        for i in range(len(operations)):
            insights = self.insight_memory.load_memories()["insights"]
            operation, operation_insight = operations[i]
            operation_type = operation.split(" ")[0]
            if operation_type == "AGREE":
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
        for i in range(len(operations)):
            insights = self.insight_memory.load_memories()["insights"]
            operation, operation_insight = operations[i]
            operation_type = operation.split(" ")[0]
            if operation_type == "REMOVE":
                insight_idx = retrieve_insight_index(insights, operation_insight)
                if insight_idx != -1:
                    self.insight_memory.delete_memories(insight_idx)

    def reset(self) -> None:
        self.experience_memory.clear()
        self.insight_memory.clear() 