from typing import Any, Dict, Optional, List, Tuple
import time
from copy import deepcopy
from itertools import chain
from agential.core.llm import BaseLLM
from agential.agents.reflexion import Reflexion
from agential.agents.base import BaseAgent
from agential.agents.expel.memory import ExpeLExperienceMemory, ExpeLInsightMemory
from agential.agents.expel.utils import (
    parse_insights,
    remove_err_operations,
    retrieve_insight_index,
    _build_compare_prompt,
    _build_all_success_prompt,
    log_llm_io,
)
from agential.utils.general import shuffle_chunk_list
import random


class ExpeLAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        verbose: bool = False,
        config: dict = {},
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        success_batch_size: int = 8,
        extract_init_insights: bool = True,
        reflexion_kwargs: Dict[str, Any] = {},
        truncate_length: Optional[int] = None,
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.experience_memory = experience_memory or ExpeLExperienceMemory()
        self.insight_memory = insight_memory or ExpeLInsightMemory()
        self.success_batch_size = success_batch_size
        self.extract_init_insights = (
            extract_init_insights and self.experience_memory.experiences != []
        )
        self.truncate_length = truncate_length

        # Create the reflexion_react_agent internally
        self.reflexion_react_agent = Reflexion(
            llm=llm, benchmark=benchmark, **reflexion_kwargs
        )

    def generate(
        self,
        question: str,
        key: str = "",
        reflect_strategy: str = "reflexion",
        additional_keys: Dict[str, str] = {},
        reflect_additional_keys: Dict[str, str] = {},
        use_dynamic_examples: bool = True,
        extract_insights: bool = True,
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

        experience: List[Dict[str, Any]] = self.gather_experience(
            questions=[question],
            keys=[key],
            examples=examples,
            prompt=prompt,
            reflect_examples=reflect_examples,
            reflect_prompt=reflect_prompt,
            additional_keys=[additional_keys],
            reflect_additional_keys=[reflect_additional_keys],
        )

        if extract_insights:
            compare_response, success_response = self.extract_insights(experience)
            compares_response.append(compare_response)
            successes_response.append(success_response)

        # Calculate metrics directly
        total_tokens = 0
        total_cost = 0.0
        total_time = time.time() - start

        # Accumulate from compare responses
        for response_list in compares_response:
            for response in response_list:
                total_tokens += getattr(response, "total_tokens", 0)
                total_cost += getattr(response, "total_cost", 0.0)

        # Accumulate from success responses
        for response_list in successes_response:
            for response in response_list:
                total_tokens += getattr(response, "total_tokens", 0)
                total_cost += getattr(response, "total_cost", 0.0)

        # Accumulate from experiences
        for exp in experience:
            trajectory = exp["trajectory"]
            metrics = trajectory.get("metrics", {})
            total_tokens += metrics.get("total_tokens", 0)
            total_cost += metrics.get("total_cost", 0.0)

        # Compose answer and steps using new dict-based output
        answer = ""
        if experience and "trajectory" in experience[0]:
            traj = experience[0]["trajectory"]
            # Use the answer from the trajectory dict
            answer = traj.get("answer", "")
        out = {
            "answer": answer,
            "experience": {
                k: v for k, v in experience[0].items() if k not in ["question", "key"]
            }
            if experience
            else {},
            "experience_memory": deepcopy(self.experience_memory.show_memories()),
            "insight_memory": deepcopy(self.insight_memory.show_memories()),
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "trials_taken": len(experience) if experience else 0,
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
        examples = "\n\n---\n\n".join(
            dynamic_examples if dynamic_examples else [examples]
        )
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
        additional_keys: List[Dict[str, str]],
        reflect_additional_keys: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Collects and aggregates experiences from a ReflexionReAct by generating trajectories and reflections for a set of questions and keys.

        The function iterates over each question-key pair, generates a trajectory using the specified strategy, and records the reflections generated by the agent. Each trajectory and its corresponding reflections are appended to their respective lists within the 'experiences' dictionary.

        Parameters:
            questions (List[str]): A list of questions to be processed by the agent.
            keys (List[str]): A list of keys that are paired with the questions to guide the agent's generation.
            examples (str, optional): Fewshot examples.
            prompt (str, optional): Prompt template string.
            reflect_examples (str, optional): Reflection fewshot examples.
            reflect_prompt (str, optional): Reflect prompt template string.
            additional_keys (List[Dict[str, str]]): Additional keys for the prompt. Defaults to [].
            reflect_additional_keys (List[Dict[str, str]]): Additional keys for the reflect prompt. Defaults to [].

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the question, key, trajectory, and reflections.
        """
        if not additional_keys:
            additional_keys = [{} for _ in range(len(questions))]

        if not reflect_additional_keys:
            reflect_additional_keys = [{} for _ in range(len(questions))]

        experiences = []
        for question, key, main_keys, reflect_keys in zip(
            questions, keys, additional_keys, reflect_additional_keys
        ):
            trajectory = self.reflexion_react_agent.generate(
                question=question,
                key=key,
                fewshot=examples,
                prompt=prompt,
                reflect_fewshot=reflect_examples,
                reflect_prompt=reflect_prompt,
                additional_keys=main_keys,  # type: ignore
                reflect_additional_keys=reflect_keys,  # type: ignore
            )

            reflections = [
                trial.get("reflections", "")
                for trial in trajectory.get("trials", [])
                if trial.get("reflections")
            ]
            # For Reflexion agents, reflections are stored as a string in the main output
            if trajectory.get("reflections"):
                reflections.append(trajectory["reflections"])
            selected_reflections = list(set(list(chain.from_iterable(reflections))))  # type: ignore
            experience = {
                "question": question,
                "key": key,
                "trajectory": trajectory,
                "reflections": selected_reflections,
            }
            experiences.append(experience)

        self.experience_memory.add_memories(
            questions=[exp["question"] for exp in experiences],
            keys=[exp["key"] for exp in experiences],
            trajectories=[exp["trajectory"] for exp in experiences],
            reflections=[exp["reflections"] for exp in experiences],
        )
        return experiences

    def categorize_experiences(
        self, experiences: List[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Categorizes experiences based on the success of trials in the trajectories.

        This function iterates over each index in the experiences and categorizes them into 'compare', 'success', or 'fail' based on the outcomes of the trials. Each trial is represented by a tuple, with the first element indicating success (True) or failure (False).

        Parameters:
            experiences (List[Dict[str, Any]]): A list of dictionaries, each containing the question, key, trajectory, and reflections.

        Returns:
            Dict[str, List]: A dictionary with the indices of tasks categorized into 'compare', 'success', and 'fail'.

        Raises:
        - ValueError: If a trajectory does not fit into any category, indicating an unhandled scenario.
        """
        count_dict: Dict[str, List] = {"compare": [], "success": [], "fail": []}

        for idx, experience in enumerate(experiences):
            trajectory = experience["trajectory"]
            trials = trajectory["trials"]

            trials_are_correct = [trial["correct"] for trial in trials]

            # Success.
            if (
                all(trials_are_correct) and len(trials_are_correct) == 1
            ):  # If success @ first trial, then stop generation.
                count_dict["success"].append(idx)
            # Compare.
            elif trials_are_correct[
                -1
            ]:  # If fail(s), then succeeds, then only last trial is True.
                count_dict["compare"].append(idx)
            # Fail.
            elif not all(trials_are_correct):  # All trials failed, then fail case.
                count_dict["fail"].append(idx)
            else:
                raise ValueError(f"Unhandled scenario for trajectory at index {idx}.")

        return count_dict

    def get_folds(
        self,
        categories: Dict[str, List],
        n_instances: int,
        n_folds: int = 2,
        seed: int = 42,
    ) -> Dict[int, List]:
        """Distributes indices into a specified number of stratified folds for cross-validation.

        Indices from each category ('compare', 'success', 'fail') are shuffled and then distributed across the folds. Each fold will serve as a validation set once during cross-validation, with the remaining data used for training.

        Parameters:
            categories (Dict[str, List]): A dictionary containing lists of indices for each category.
            n_instances (int): The total number of instances across all categories.
            n_folds (int, optional): The number of folds to create for cross-validation. Default is 2.

        Returns:
            Dict[int, List]: A dictionary where keys are fold indices and values are the lists of indices representing the training set for that fold.
        """
        random.seed(seed)

        folds: Dict[int, List] = {fold: [] for fold in range(n_folds)}

        # Assign labels for 'compare', 'success', and  'fail'.
        for _, indices in categories.items():
            indices = random.sample(indices, len(indices))
            for count, idx in enumerate(indices):
                folds[count % n_folds].append(idx)

        # Each fold is a validation set. Take the difference to get the training set for each fold.
        folds = {
            fold: list(set(list(range(n_instances))).difference(values))
            for fold, values in folds.items()
        }

        return folds

    def extract_insights(
        self, experiences: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[Any]]:
        # Use the new dict-based output structure for trajectory and steps
        categories = self.categorize_experiences(experiences)
        folds = self.get_folds(categories, len(experiences))
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

                    # Build compare prompt using the utility function
                    is_full = self.insight_memory.max_num_insights < len(insights)
                    prompt = _build_compare_prompt(
                        insights=insights,
                        question=question,
                        success_trial=success_trial,
                        failed_trial=failed_trial_str,
                        is_full=is_full,
                    )
                    compare_out = self.llm(prompt)
                    log_llm_io(
                        compare_out,
                        f"Compare Insights - Trial {train_idx}",
                        self.verbose,
                        self.truncate_length,
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
                            concat_success_trajs.append(
                                f"{experiences[idx]['question']}\n" + steps_str
                            )
                    success_trials = "\n\n".join(concat_success_trajs)

                    # Build all success prompt using the utility function
                    is_full = self.insight_memory.max_num_insights < len(insights)
                    prompt = _build_all_success_prompt(
                        insights=insights,
                        success_trajs_str=success_trials,
                        is_full=is_full,
                    )
                    success_out = self.llm(prompt)
                    log_llm_io(
                        success_out,
                        f"Success Insights - Batch {success_idxs}",
                        self.verbose,
                        self.truncate_length,
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
