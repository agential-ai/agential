"""ExpeL's memory implementations.

Original Paper: https://arxiv.org/abs/2308.10144
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy

from langchain.vectorstores import FAISS
from langchain_core.documents.base import Document

from discussion_agents.cog.modules.memory.base import BaseMemory


class ExpeLExperienceMemory(BaseMemory):
    def __init__(
        self,
        experiences: Optional[Dict[str, List]] = {},
        success_traj_idxs: Optional[List[int]] = [],
        fewshot_questions: Optional[List[str]] = [],
        fewshot_keys: Optional[List[str]] = [],
        fewshot_examples: Optional[List[List[Tuple[str, str, str]]]] = [],
    ) -> None:
        super().__init__()

        self.success_traj_idxs = success_traj_idxs
        self.experiences = deepcopy(experiences)

        self.success_traj_docs = []
        for idx in self.success_traj_idxs:
            question = self.experiences["questions"][idx]
            trajectory = self.experiences["trajectories"][idx][
                0
            ]  # Zero-th trial of trajectory.
            is_correct, _, steps = trajectory
            assert is_correct  # Ensure trajectory is successful.

            # Add the task.
            self.success_traj_docs.append(
                Document(
                    page_content=question, metadata={"type": "task", "task_idx": idx}
                )
            )

            # Add all trajectory actions.
            self.success_traj_docs.extend(
                [
                    Document(
                        page_content=action,
                        metadata={"type": "action", "task_idx": idx},
                    )
                    for (_, action, _) in steps
                ]
            )

            # Add all trajectory thoughts.
            self.success_traj_docs.extend(
                [
                    Document(
                        page_content=thought,
                        metadata={"type": "thought", "task_idx": idx},
                    )
                    for (thought, _, _) in steps
                ]
            )

            # Add each step.
            for step in steps:
                self.success_traj_docs.append(
                    Document(
                        page_content="\n".join(step),
                        metadata={"type": "step", "task_idx": idx},
                    )
                )

        # If including fewshot examples in experiences.
        if fewshot_questions and fewshot_keys and fewshot_examples:
            # Update experiences.
            for question, key, steps in zip(fewshot_questions, fewshot_keys, fewshot_examples):
                idx = max(self.experiences['idxs']) + 1

                self.experiences['idxs'].append(idx)
                self.experiences['questions'].append(question)
                self.experiences['keys'].append(key)
                self.experiences['trajectories'].append(
                    [
                        (True, key, steps)
                    ]
                )
                self.experiences['reflections'].append([])

                # Update success_traj_docs.

                # Add the task.
                self.success_traj_docs.append(
                    Document(
                        page_content=question, metadata={
                            "type": "task", 
                            "task_idx": idx
                        }
                    )
                )

                # Add all trajectory actions.
                self.success_traj_docs.extend(
                    [
                        Document(
                            page_content=action,
                            metadata={"type": "action", "task_idx": idx},
                        )
                        for (_, action, _) in steps
                    ]
                )

                # Add all trajectory thoughts.
                self.success_traj_docs.extend(
                    [
                        Document(
                            page_content=thought,
                            metadata={"type": "thought", "task_idx": idx},
                        )
                        for (thought, _, _) in steps
                    ]
                )

                # Add each step.
                for step in steps:
                    self.success_traj_docs.append(
                        Document(
                            page_content="\n".join(step),
                            metadata={"type": "step", "task_idx": idx},
                        )
                    )

    def clear(self) -> None:
        """Clears all stored experiences from the memory.

        Resets the memory to its initial empty state.
        """
        self.experiences = {
            "idxs": [],
            "questions": [],
            "keys": [],
            "trajectories": [],
            "reflections": [],
        }

    def add_memories(
        self,
        questions: List[str],
        keys: List[str],
        trajectories: List[List[Tuple[bool, str, List[Tuple[str, str, str]]]]],
        reflections: Optional[List[List[str]]] = [],
    ) -> None:
        assert len(questions) == len(keys) == len(trajectories)

        self.experiences["idxs"].extend(
            list(
                range(
                    len(self.experiences["idxs"]),
                    len(self.experiences["idxs"]) + len(questions),
                )
            )
        )
        self.experiences["questions"].extend(questions)
        self.experiences["keys"].extend(keys)
        self.experiences["trajectories"].extend(trajectories)

        if reflections:
            assert len(reflections) == len(questions)
        else:
            reflections = [[] for _ in range(len(questions))]
        self.experiences["reflections"].extend(reflections)

    def load_memories(self, idxs: Union[List[int], int]) -> Dict[str, Any]:
        if isinstance(idxs, int):
            idxs = [idxs]

        return {
            "questions": [self.experiences["questions"][idx] for idx in idxs],
            "keys": [self.experiences["keys"][idx] for idx in idxs],
            "trajectories": [self.experiences["trajectories"][idx] for idx in idxs],
            "reflections": [self.experiences["reflections"][idx] for idx in idxs],
        }

    def show_memories(self) -> Dict[str, Any]:
        return self.experiences
