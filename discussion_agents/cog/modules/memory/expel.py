"""ExpeL's memory implementations.

Original Paper: https://arxiv.org/abs/2308.10144
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from typing import Any, Dict, List, Tuple, Optional

from langchain.vectorstores import FAISS

from discussion_agents.cog.modules.memory.base import BaseMemory


from typing import Any, Dict, List, Tuple, Optional, Union
from discussion_agents.cog.modules.memory.base import BaseMemory

class ExpeLExperienceMemory(BaseMemory):
    def __init__(
        self,
        questions: Optional[List[str]] = [],
        keys: Optional[List[str]] = [],
        trajectories: Optional[List[List[Tuple[bool, str, List[Tuple[str, str, str]]]]]] = [],
        reflections: Optional[List[List[str]]] = []
    ) -> None:
        """Initializes the experience memory.

        Initializes the memory with optional lists of questions, keys, trajectories, and reflections. Ensures data
        consistency by asserting that all provided lists have the same length. Initializes reflections to an empty list
        for each question if not provided.

        Args:
            questions (Optional[List[str]]): A list of string questions associated with the experiences. Defaults to an empty list.
            keys (Optional[List[str]]): A list of string keys that uniquely identify each experience. Defaults to an empty list.
            trajectories (Optional[List[List[Tuple[bool, str, List[Tuple[str, str, str]]]]]]): A list of trajectories, where each trajectory
                is a list of steps, and each step is a tuple consisting of a success flag (bool), a description (str),
                and a list of details (List[Tuple[str, str, str]]). Defaults to an empty list.
            reflections (Optional[List[List[str]]]): A list of lists, where each sublist contains string reflections for each experience.
                Defaults to an empty list corresponding to each question if not explicitly provided.

        Raises:
            AssertionError: If the lengths of provided lists (questions, keys, trajectories) do not match, or if reflections are provided
                            but do not match the length of questions.
        """
        super().__init__()

        assert len(questions) == len(keys) == len(trajectories)

        if reflections:
            assert len(reflections) == len(questions)
        else:
            reflections = [[] for _ in range(len(questions))]

        self.experiences = {
            "idxs": list(range(len(questions))),
            "questions": questions,
            "keys": keys,
            "trajectories": trajectories,
            "reflections": reflections
        }

    def clear(self) -> None:
        """Clears all stored experiences from the memory.

        Resets the memory to its initial empty state.
        """
        self.experiences = {
            "idxs": [],
            "questions": [],
            "keys": [],
            "trajectories": [],
            "reflections": []
        }

    def add_memories(
        self, 
        questions: List[str], 
        keys: List[str], 
        trajectories: List[List[Tuple[bool, str, List[Tuple[str, str, str]]]]],
        reflections: Optional[List[List[str]]] = []
    ) -> None:
        """Adds new memories to the existing collection.

        Extends the current memory with new questions, keys, trajectories, and optionally reflections. Checks for
        consistency in the lengths of the provided lists.

        Args:
            questions (List[str]): New questions to add.
            keys (List[str]): New keys corresponding to each question.
            trajectories (List[List[Tuple[bool, str, List[Tuple[str, str, str]]]]]): New trajectories associated with each question and key.
            reflections (Optional[List[List[str]]]): Optional new reflections associated with each question. Defaults to an empty list
                for each new question if not provided.

        Raises:
            AssertionError: If the lengths of questions, keys, and trajectories do not match, or if reflections are provided but their
                            length doesn't match the length of new questions.
        """
        assert len(questions) == len(keys) == len(trajectories)

        self.experiences['idxs'].extend(
            list(
                range(
                    len(self.experiences['idxs']),
                    len(self.experiences['idxs']) + len(questions)
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
        """Retrieves specified memories by their indices.

        Args:
            idxs (Union[List[int], int]): A single index or a list of indices specifying which memories to load.

        Returns:
            Dict[str, Any]: A dictionary containing lists of the requested components ('questions', 'keys', 'trajectories', 'reflections') of the memories.
        """
        if isinstance(idxs, int): 
            idxs = [idxs]            

        return {
            'questions': [self.experiences['questions'][idx] for idx in idxs],
            'keys': [self.experiences['keys'][idx] for idx in idxs],
            'trajectories': [self.experiences['trajectories'][idx] for idx in idxs],
            'reflections': [self.experiences['reflections'][idx] for idx in idxs],
        }

    def show_memories(self) -> Dict[str, Any]:
        """Returns all currently stored memories.

        Returns:
            Dict[str, Any]: A dictionary containing all components of the stored memories ('idxs', 'questions', 'keys', 'trajectories', 'reflections').
        """
        return self.experiences

