"""WebVoyager data manager."""

import json
import os

from typing import Any, Dict, List, Optional


class WebVoyagerDataManager:
    """WebVoyager data manager to process and read JSONL files."""

    def __init__(self, mode: str = "benchmark", examples_dir: str = "") -> None:
        """Initialize the data manager and load data from a JSONL file.

        Args:
            mode (str): The mode to run the benchmark in. Can be either 'custom' or 'benchmark'. Defaults to "benchmark".
            examples_dir (str): Path to the JSONL file containing data. Defaults to "" (will use benchmark examples).
        """
        self.mode = mode
        self.examples_dir = examples_dir
        self.data: List[Dict[str, Any]] = []

        # Only if in "benchmark" mode.
        self.reference_answers: Dict[str, Any] = {}

        if self.mode == "custom":
            if self.examples_dir == "":
                raise ValueError("examples_dir must be provided if mode is 'custom'.")
            if not os.path.exists(self.examples_dir):
                raise ValueError("examples_dir does not exist.")

        elif self.mode == "benchmark":
            current_file_path = os.path.dirname(__file__)
            examples_dir = os.path.join(
                current_file_path, "evaluation_examples", "WebVoyager_data.jsonl"
            )
            examples_answers_dir = os.path.join(
                current_file_path, "evaluation_examples", "reference_answer.json"
            )
            try:
                with open(examples_answers_dir, "r") as file:
                    self.reference_answers = json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {examples_answers_dir}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file: {e}")
        else:
            raise ValueError("Mode must be either 'custom' or 'benchmark'.")

        try:
            with open(self.examples_dir, "r") as file:
                for line in file:
                    self.data.append(json.loads(line.strip()))
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.examples_dir}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file: {e}")

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific task by its ID."""
        for task in self.data:
            if task.get("task_id") == task_id:
                return task
        return None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Retrieve all tasks."""
        return self.data

    def get_all_answers(self) -> Dict[str, Any]:
        """Retrieve the entire reference answers dictionary."""
        return self.reference_answers

    def get_tasks_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Retrieve all tasks with a specific level."""
        return [task for task in self.data if task.get("Level") == level]

    def get_all_sources(self) -> List[str]:
        """Retrieve all source names (keys) from the reference answers."""
        return list(self.reference_answers.keys())

    def get_notice_by_source(self, source: str) -> Optional[str]:
        """Retrieve the notice for a specific source."""
        if source in self.reference_answers:
            return self.reference_answers[source].get("notice", None)
        return None

    def get_answers_by_source(self, source: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve all answers for a specific source."""
        if source in self.reference_answers:
            return self.reference_answers[source].get("answers", None)
        return None

    def get_answer_by_id(self, source: str, answer_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific answer by its ID for a given source."""
        answers = self.get_answers_by_source(source)
        if answers:
            for answer in answers:
                if answer.get("id") == answer_id:
                    return answer
        return None

    def get_all_possible_answers(self, source: str) -> List[Dict[str, Any]]:
        """Retrieve all possible answers for a given source."""
        answers = self.get_answers_by_source(source)
        if answers:
            return [answer for answer in answers if answer.get("type") == "possible"]
        return []

    def get_all_unique_types(self, source: str) -> List[str]:
        """Retrieve all unique answer types for a given source."""
        answers = self.get_answers_by_source(source)
        if answers:
            return list({answer.get("type") for answer in answers if "type" in answer})
        return []


class GAIADataManager:
    """GAIA data manager to process and read GAIA_web.jsonl files."""

    def __init__(self, mode: str = "benchmark", file_path: str = "") -> None:
        """Initialize the data manager and load data from a JSONL file.

        Args:
            mode (str): The mode to run the manager in. Can be either 'custom' or 'benchmark'. Defaults to "benchmark".
            file_path (str): Path to the GAIA JSONL file. Required if mode is 'custom'.
        """
        self.mode = mode
        self.file_path = file_path
        self.data: List[Dict[str, Any]] = []

        if self.mode == "custom":
            if not self.file_path:
                raise ValueError("file_path must be provided if mode is 'custom'.")
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            try:
                with open(self.file_path, "r") as file:
                    for line in file:
                        self.data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file: {e}")

        elif self.mode == "benchmark":
            # In benchmark mode, load the default benchmark examples.
            current_file_path = os.path.dirname(__file__)
            default_benchmark_path = os.path.join(
                current_file_path, "evaluation_examples", "GAIA_web.jsonl"
            )
            if not os.path.exists(default_benchmark_path):
                raise FileNotFoundError(
                    f"Default benchmark file not found: {default_benchmark_path}"
                )
            try:
                with open(default_benchmark_path, "r") as file:
                    for line in file:
                        self.data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in benchmark file: {e}")
        else:
            raise ValueError("Mode must be either 'custom' or 'benchmark'.")

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific task by its ID."""
        for task in self.data:
            if task.get("task_id") == task_id:
                return task
        return None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Retrieve all tasks."""
        return self.data

    def get_tasks_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Retrieve all tasks with a specific level."""
        return [task for task in self.data if task.get("Level") == level]

    def get_all_task_ids(self) -> List[str]:
        """Retrieve all task IDs."""
        return [task.get("task_id") for task in self.data if "task_id" in task]

    def get_question_by_task_id(self, task_id: str) -> Optional[str]:
        """Retrieve the question (ques) for a specific task by its ID."""
        task = self.get_task_by_id(task_id)
        if task:
            return task.get("ques", None)
        return None

    def get_final_answer_by_task_id(self, task_id: str) -> Optional[str]:
        """Retrieve the final answer for a specific task by its ID."""
        task = self.get_task_by_id(task_id)
        if task:
            return task.get("Final answer", None)
        return None

    def get_tasks_with_web_reference(self) -> List[Dict[str, Any]]:
        """Retrieve all tasks that contain a web reference."""
        return [task for task in self.data if task.get("web")]

    def get_all_web_references(self) -> List[str]:
        """Retrieve all web references from the dataset."""
        return [task.get("web") for task in self.data if "web" in task]
