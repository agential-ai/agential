"""WebVoyager data manager."""

from typing import Dict, List, Any, Optional
import json
import os


class WebVoyagerDataManager:
    """WebVoyager data manager to process and read JSONL files."""

    def __init__(self, mode: str = "benchmark", examples_dir: str = "") -> None:
        """
        Initialize the data manager and load data from a JSONL file.

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

    def get_tasks_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Retrieve all tasks with a specific level."""
        return [task for task in self.data if task.get("Level") == level]


