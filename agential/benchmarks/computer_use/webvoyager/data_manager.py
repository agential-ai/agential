"""WebVoyager data manager."""

from typing import Dict, List, Any, Optional
import json
import os


class WebVoyagerDataManager:
    """WebVoyager data manager to process and read JSONL files."""

    def __init__(self, examples_dir: str = "") -> None:
        """
        Initialize the data manager and load data from a JSONL file.

        Args:
            examples_dir (str): Path to the JSONL file containing data. Defaults to "" (will use benchmark examples).
        """
        self.data = []

        if not examples_dir:
            current_file_path = os.path.dirname(__file__)
            examples_dir = os.path.join(
                current_file_path, "evaluation_examples", "WebVoyager_data.jsonl"
            )

        self.examples_dir = examples_dir

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


