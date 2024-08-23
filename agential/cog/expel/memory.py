"""ExpeL's memory implementations.

Original Paper: https://arxiv.org/abs/2308.10144
Paper Repository: https://github.com/LeapLabTHU/ExpeL
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

import tiktoken

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from scipy.spatial.distance import cosine
from tiktoken.core import Encoding

from agential.cog.base.modules.memory import BaseMemory
from agential.cog.reflexion.output import ReflexionReActOutput


class ExpeLExperienceMemory(BaseMemory):
    """ExpeL's experience pool memory.

    Attributes:
        experiences (List[Dict[str, Any]]): A list of experiences. Defaults to [].
        strategy (str): The strategy employed for handling and vectorizing experiences. Defaults to "task".
        embedder (Embeddings): An embedding object used for generating vector embeddings of documents. Defaults to HuggingFaceEmbeddings.
        encoder (Encoding): An encoder object used for token counting within documents. Defaults to gpt-3.5-turbo.
    """

    def __init__(
        self,
        experiences: Optional[List[Dict[str, Any]]] = [],
        strategy: str = "task",
        embedder: Embeddings = HuggingFaceEmbeddings(),
        encoder: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initializes the memory with optional experiences, fewshot examples, and strategies."""
        super().__init__()

        self.experiences = deepcopy(experiences) if experiences else []
        self.strategy = strategy
        self.embedder = embedder
        self.encoder = encoder

        # Collect all successful trajectories.
        success_traj_idxs: List[int] = []
        if len(self.experiences):
            success_traj_idxs = []
            for idx, experience in enumerate(self.experiences):
                trajectory = experience["trajectory"].additional_info
                is_correct = trajectory[0].steps[-1].is_correct  # Success on last step of the zero-th trial of this trajectory.
                if is_correct:
                    success_traj_idxs.append(idx)

        self.success_traj_docs: List[Document] = []
        for idx in success_traj_idxs:
            question = self.experiences[idx]["question"]
            steps = self.experiences[idx]["trajectory"].additional_info[
                0
            ].steps  # Zero-th trial of trajectory.

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
                        page_content=f"Action: {step.action_type}[{step.query}]",
                        metadata={"type": "action", "task_idx": idx},
                    )
                    for step in steps
                ]
            )

            # Add all trajectory thoughts.
            self.success_traj_docs.extend(
                [
                    Document(
                        page_content=f"Thought: {step.thought}",
                        metadata={"type": "thought", "task_idx": idx},
                    )
                    for step in steps
                ]
            )

            # Add each step.
            for step in steps:
                step_string = f"Thought: {step.thought}\nAction: {step.action_type}[{step.query}]\nObservation: {step.observation}\n"
                self.success_traj_docs.append(
                    Document(
                        page_content=step_string,
                        metadata={"type": "step", "task_idx": idx},
                    )
                )

        # Create vectorstore.
        self.vectorstore = None
        if len(self.experiences) and len(self.success_traj_docs):
            self.vectorstore = FAISS.from_documents(
                [
                    doc
                    for doc in self.success_traj_docs
                    if doc.metadata["type"] == self.strategy
                ],
                self.embedder,
            )

    def __len__(self) -> int:
        """Returns length of experiences."""
        return len(self.experiences)

    def clear(self) -> None:
        """Clears all stored experiences from the memory.

        Resets the memory to its initial empty state.
        """
        self.experiences = []
        self.success_traj_docs = []
        self.vectorstore = None

    def add_memories(
        self,
        questions: List[str],
        keys: List[str],
        trajectories: List[ReflexionReActOutput],
        reflections: Optional[List[List[str]]] = [],
    ) -> None:
        """Adds new experiences to the memory, including associated questions, keys, trajectories, and optional reflections.

        Args:
            questions (List[str]): Questions related to the experiences being added.
            keys (List[str]): Answers corresponding to the provided questions.
            trajectories (List[ReflexionReActOutput]): A list of trajectories.
            reflections (Optional[List[List[str]]], default=[]): A list of additional reflective notes on the experiences.
        """
        assert len(questions) == len(keys) == len(trajectories)

        if reflections:
            assert len(reflections) == len(questions)
        else:
            reflections = [[] for _ in range(len(questions))]

        start_idx = len(self.experiences)

        # Update experiences.
        experiences = [
            {
                "question": question,
                "key": key,
                "trajectory": trajectory,
                "reflections": reflection,
            }
            for (question, key, trajectory, reflection) in zip(
                questions, keys, trajectories, reflections
            )
        ]
        self.experiences.extend(experiences)

        # Update success_traj_docs.
        success_traj_idxs = []
        for idx, trajectory in enumerate(trajectories, start_idx):
            is_correct = trajectory.addtional_info[0].steps[-1].is_correct
            if is_correct:
                success_traj_idxs.append(idx)

        for idx in success_traj_idxs:
            question = self.experiences[idx]["question"]
            steps = self.experiences[idx]["trajectory"].addtional_info[
                0
            ].steps  # Zero-th trial of trajectory.

            # Add the task.
            self.success_traj_docs.append(
                Document(
                    page_content=question, metadata={"type": "task", "task_idx": idx}
                )
            )

            # Add all trajectory actions.
            self.success_traj_docs.extend(
                [  # type: ignore
                    Document(
                        page_content=f"Action: {step.action_type}[{step.query}]",  # type: ignore
                        metadata={"type": "action", "task_idx": idx},
                    )
                    for step in steps
                ]
            )

            # Add all trajectory thoughts.
            self.success_traj_docs.extend(
                [  # type: ignore
                    Document(
                        page_content=f"Thought: {step.thought}",  # type: ignore
                        metadata={"type": "thought", "task_idx": idx},
                    )
                    for step in steps
                ]
            )

            # Add each step.
            for step in steps:
                step_string = f"Thought: {step.thought}\nAction: {step.action_type}[{step.query}]\nObservation: {step.observation}\n"
                self.success_traj_docs.append(
                    Document(
                        page_content=step_string,  # type: ignore
                        metadata={"type": "step", "task_idx": idx},
                    )
                )

        if success_traj_idxs:
            # Create vectorstore.
            self.vectorstore = FAISS.from_documents(
                [
                    doc
                    for doc in self.success_traj_docs
                    if doc.metadata["type"] == self.strategy
                ],
                self.embedder,
            )

    def _fewshot_doc_token_count(self, fewshot_doc: Document) -> int:
        """Returns the token count of a given document's successful trajectory.

        Args:
            fewshot_doc (Document): The document containing trajectory data.

        Returns:
            int: The token count of the document's trajectory.
        """
        task_idx = fewshot_doc.metadata["task_idx"]
        trajectory = self.experiences[task_idx]["trajectory"]
        steps = trajectory[0].react_output  # A successful trial.
        steps_str = ""
        for step in steps:
            step = f"Thought: {step.thought}\nAction: {step.action_type}[{step.query}]\nObservation: {step.observation}\n"
            steps_str += step

        return len(self.encoder.encode(steps_str))

    def load_memories(
        self,
        query: str,
        k_docs: int = 24,
        num_fewshots: int = 6,
        max_fewshot_tokens: int = 1500,
        reranker_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieves fewshot documents based on a similarity search, with optional re-ranking strategies.

        Args:
            query (str): The query to perform similarity search against.
            k_docs (int): The number of documents to return from a similarity search.
            num_fewshots (int): The number of fewshot examples to utilize or retrieve.
            max_fewshot_tokens (int): The maximum number of tokens allowed in a single fewshot example. Defaults to 1500.
            reranker_strategy (Optional[str]): The re-ranking strategy to be applied based on similarity measures.

        Returns:
            Dict[str, Any]: A dictionary of retrieved fewshot documents (strings).
        """
        # If empty.
        if (
            not len(self.experiences)
            or not k_docs
            or not num_fewshots
            or not max_fewshot_tokens
            or not self.vectorstore
        ):
            return {"fewshots": []}

        # Query the vectorstore.
        fewshot_docs = self.vectorstore.similarity_search(query, k=k_docs)

        # Post-processing.

        # Re-ranking, optional.
        if not reranker_strategy:
            fewshot_docs = list(fewshot_docs)
        elif reranker_strategy == "length":
            fewshot_docs = list(
                sorted(fewshot_docs, key=self._fewshot_doc_token_count, reverse=True)
            )
        elif reranker_strategy == "thought":
            fewshot_tasks = set([doc.metadata["task_idx"] for doc in fewshot_docs])
            subset_docs = list(
                filter(
                    lambda doc: doc.metadata["type"] == "thought"
                    and doc.metadata["task_idx"] in fewshot_tasks,
                    list(self.success_traj_docs),
                )
            )
            fewshot_docs = sorted(
                subset_docs,
                key=lambda doc: cosine(
                    self.embedder.embed_query(doc.page_content),
                    self.embedder.embed_query(query),
                ),
            )
        elif reranker_strategy == "task":
            fewshot_tasks = set([doc.metadata["task_idx"] for doc in fewshot_docs])
            subset_docs = list(
                filter(
                    lambda doc: doc.metadata["type"] == "thought"
                    and doc.metadata["task_idx"] in fewshot_tasks,
                    list(self.success_traj_docs),
                )
            )
            fewshot_docs = sorted(
                subset_docs,
                key=lambda doc: cosine(
                    self.embedder.embed_query(doc.page_content),
                    self.embedder.embed_query(query),
                ),
            )
        else:
            raise NotImplementedError

        current_tasks = set()
        fewshots = []

        # Filtering.
        # Exclude fewshot documents that exceed the token limit
        # or have already been selected as fewshot examples to avoid redundancy.
        for fewshot_doc in fewshot_docs:
            task_idx = fewshot_doc.metadata["task_idx"]
            question = self.experiences[task_idx]["question"]
            trajectory = self.experiences[task_idx]["trajectory"]
            steps = trajectory[0].react_output  # Zero-th successful trial.
            steps_str = ""
            for step in steps:
                step = f"Thought: {step.thought}\nAction: {step.action_type}[{step.query}]\nObservation: {step.observation}\n"
                steps_str += step

            if (
                len(self.encoder.encode(steps_str)) <= max_fewshot_tokens
                and task_idx not in current_tasks
            ):
                fewshots.append(f"{question}\n{steps_str}")
                current_tasks.add(task_idx)

            if len(fewshots) == num_fewshots:
                break

        return {"fewshots": fewshots}

    def show_memories(
        self,
        experiences_key: str = "experiences",
    ) -> Dict[str, Any]:
        """Displays the current set of stored experiences and vectorstore information.

        Args:
            experiences_key (str, optional): Key for accessing experiences. Defaults to "experiences".

        Returns:
            Dict[str, Any]: A dictionary containing experiences, successful trajectory documents, and vectorstore details.
        """
        return {
            experiences_key: self.experiences,
        }


class ExpeLInsightMemory(BaseMemory):
    """A memory management class for ExpeL insights, handling operations like adding, deleting, and updating insights within a memory storage with a maximum capacity.

    Attributes:
        insights (List[Dict[str, Any]]): A list to store insight dictionaries.
        max_num_insights (int): Maximum number of insights that can be stored.
    """

    def __init__(
        self,
        insights: List[Dict[str, Any]] = [],
        max_num_insights: int = 20,
        leeway: int = 5,
    ) -> None:
        """Initializes the ExpeLInsightMemory with optional insights and a maximum storage limit.

        Args:
            insights (List[Dict[str, Any]]): Initial list of insights to store in memory.
            max_num_insights (int): The maximum number of insights that can be stored.
            leeway (int): Number of memories allowed over max_num_insights before
                delete_memories instantly deletes an indexed memory.
        """
        super().__init__()

        self.insights = deepcopy(insights)
        self.max_num_insights = max_num_insights
        self.leeway = leeway

    def __len__(self) -> int:
        """Returns length of insights."""
        return len(self.insights)

    def clear(self) -> None:
        """Clears all stored insights from the memory."""
        self.insights = []

    def add_memories(self, insights: List[Dict[str, Any]]) -> None:
        """Adds new insights to the memory, up to the maximum storage limit.

        Args:
            insights (List[Dict[str, Any]]): A list of insights to add to the memory.
        """
        self.insights.extend(insights)

    def delete_memories(self, idx: int) -> None:
        """Deletes an insight from memory based on its index.

        Adjusts insight scores before deletion.

        Args:
            idx (int): The index of the insight to delete.
        """
        if idx < len(self.insights):
            if len(self.insights) >= self.max_num_insights + self.leeway:
                _ = self.insights.pop(idx)
            else:
                self.insights[idx]["score"] -= 1
                if self.insights[idx]["score"] <= 0:
                    _ = self.insights.pop(idx)

    def update_memories(
        self, idx: int, update_type: str, insight: Optional[str] = None
    ) -> None:
        """Updates an insight or its score based on the specified update type.

        Args:
            idx (int): The index of the insight to update.
            update_type (str): The type of update ("EDIT", "AGREE").
            insight (str): The new insight text (if applicable).
        """
        if update_type == "EDIT" and insight:
            self.insights[idx]["insight"] = insight
            self.insights[idx]["score"] += 1
        elif update_type == "AGREE":
            self.insights[idx]["score"] += 1
        else:
            raise NotImplementedError

    def load_memories(self, insights_key: str = "insights") -> Dict[str, Any]:
        """Loads and returns stored insights.

        Args:
            insights_key (str): The key name under which insights are returned.

        Returns:
            Dict[str, Any]: A dictionary containing stored insights.
        """
        return {insights_key: self.insights}

    def show_memories(self, insights_key: str = "insights") -> Dict[str, Any]:
        """Returns a dictionary of all stored insights for display or analysis.

        Args:
            insights_key (str): The key name under which insights are returned.

        Returns:
            Dict[str, Any]: A dictionary containing stored insights.
        """
        return {insights_key: self.insights}
