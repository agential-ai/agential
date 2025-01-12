"""OSWorldDataLoader Example Retriever."""

import json
import os
from glob import glob

from typing import Any, Dict, List

TYPE_TO_LOOK = ["googledrive", "login", "googledrive_file"]


class OSWorldDataLoader:
    """OSWorld Processor to load and manage data.
    
    Parameters:
        examples_dir (str): Path to the directory containing the JSON examples. Defaults to "", which implies using the benchmark tasks.
        mode (str): The mode to run the benchmark in. Can be either 'custom' or 'benchmark'. Defaults to "custom".
        test_type (str): The type of test to run. This parameter is used if mode is "benchmark", which implies to use the benchmark tasks. Defaults to "".
        path_to_google_settings (str): The path to the Google settings file. Required for benchmark multi-app tasks ("benchmark" mode). Defaults to "".
        path_to_googledrive_settings (str): The path to the Google Drive settings file. Required for benchmark multi-app tasks ("benchmark" mode). Defaults to "".
        ignore_files (List[str]): List of files to ignore. Defaults to ['__pycache__'].
    """

    def __init__(
        self, 
        examples_dir: str = "", 
        mode: str = "custom",
        test_type: str = "",
        path_to_google_settings: str = "",
        path_to_googledrive_settings: str = "",
        ignore_files: List[str] = ['__pycache__'],
    ) -> None:
        """Initialization."""
        self.examples_dir = examples_dir
        self.mode = mode
        self.ignore_files = ignore_files

        self.tasks: Dict[str, List[str]] = {}
        self.data: Dict[str, Any] = {}

        # Only used for benchmark mode.
        self.test_type = test_type
        self.path_to_google_settings = path_to_google_settings
        self.path_to_googledrive_settings = path_to_googledrive_settings

        if self.mode == "custom":
            if self.examples_dir == "":
                raise ValueError("examples_dir must be provided if mode is 'custom'.")
            if not os.path.exists(self.examples_dir):
                raise ValueError("examples_dir does not exist.")
            
            self._load_data()
        elif self.mode == "benchmark":
            current_file_path = os.path.dirname(__file__)
            evaluation_examples_path = os.path.join(current_file_path, "evaluation_examples")
            
            self.examples_dir = os.path.join(evaluation_examples_path, "examples")

            # Check if the path_to_google_settings and path_to_googledrive_settings are valid.
            if self.path_to_google_settings == "" or not os.path.exists(self.path_to_google_settings):
                raise ValueError("`path_to_google_settings` file not found.")

            if self.path_to_googledrive_settings == "" or not os.path.exists(self.path_to_googledrive_settings):
                raise ValueError("`path_to_googledrive_settings` settings file not found.")
            
            # Get self.tasks. 
            test_file: str = os.path.join(evaluation_examples_path, f"{self.test_type}.json")
            try:
                with open(test_file, "r") as f:
                    self.tasks = json.load(f)
            except FileNotFoundError:
                task_set_options = [
                    os.path.splitext(os.path.basename(file))[0] for file in glob(os.path.join(evaluation_examples_path, "test_*.json"))
                ]
                raise FileNotFoundError(f"Using benchmark tasks and task set {test_file} not found. Available options: {', '.join(task_set_options)}.")

            self._load_data()
            self._update_credentials()
        else:
            raise ValueError("Mode must be either 'custom' or 'benchmark'.")

    def _load_data(self) -> None:
        """Load all JSON files into self.data."""
        for domain in os.listdir(self.examples_dir):
            # Ignore the domain if it's in the ignore list.
            if domain in self.ignore_files:
                continue

            # Skip the domain if it's not in the set of domains for the specified benchmark tasks.
            if self.mode == "benchmark" and domain not in self.tasks.keys():
                continue

            domain_path = os.path.join(self.examples_dir, domain)
            if os.path.isdir(domain_path):  # Ensure it's a directory.
                self.data[domain] = {}
                for task_file in os.listdir(domain_path):
                    if task_file.endswith(".json"):  # Ensure it's a JSON file.
                        task_id = os.path.splitext(task_file)[
                            0
                        ]  # Get the task ID (filename without extension).

                        # Skip the task if it's not in the set of tasks for the specified domain in the benchmark tasks.
                        if self.mode == "benchmark" and task_id not in self.tasks[domain]:
                            continue

                        task_path = os.path.join(domain_path, task_file)
                        with open(task_path, "r") as f:
                            self.data[domain][task_id] = json.load(f)

    def _change_example_credential(self, example: Dict[str, Any]) -> Any:
        """Modifies credential settings in a given example based on file type.

        Args:
            example (Dict[str, Any]): The task configuration to be updated.

        Returns:
            Dict[str, Any]: The updated task configuration.
        """
        for item in example["config"]:
            if item["type"] in TYPE_TO_LOOK:
                file_type = item["parameters"]["settings_file"].split(".")[-1]
                if file_type == "yml":
                    item["parameters"][
                        "settings_file"
                    ] = self.path_to_googledrive_settings
                elif file_type == "json" and item["parameters"]["platform"] == "googledrive":
                    item["parameters"][
                        "settings_file"
                    ] = self.path_to_google_settings

        path = example["evaluator"]["result"]
        if (
            path["type"] in TYPE_TO_LOOK
            and path["settings_file"].split(".")[-1] == "yml"
        ):
            path["settings_file"] = self.path_to_googledrive_settings

        return example

    def _update_credentials(self) -> None:
        """Updates credentials for the specified domain and/or task.

        Args:
            domain (str, optional): The domain whose tasks' credentials should be updated.
            task_id (str, optional): The task ID to update credentials for.

        Returns:
            Dict[str, Any]: The updated credentials for the specified domain and/or task.
        """
        for domain in self.data.keys():
            for task in self.data[domain].keys():
                self.data[domain][task] = self._change_example_credential(
                    self.data[domain][task]
                )

    def get(self, domain: str = "", task_id: str = "") -> Any:
        """Retrieve data for a specific domain, task_id, or both.

        Args:
            domain (str): The domain to filter data by.
            task_id (str): The task ID to filter data by.

        Returns:
            Dict[str, Any]: The data for the specified domain/task ID, or all data if no filters are applied.
        """
        if domain and task_id:
            # Return specific task data
            return self.data.get(domain, {}).get(task_id)
        elif domain:
            # Return all data for a domain
            return self.data.get(domain)
        elif task_id:
            # Search all domains for the task_id
            for domain_data in self.data.values():
                if task_id in domain_data:
                    return domain_data[task_id]
        else:
            # Return all data
            return self.data
