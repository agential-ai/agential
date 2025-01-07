import json
import os
from typing import Any

import json
import os
from typing import Any


class OSWorldEnv:
    """OSWorld Processor to load and manage data."""

    def __init__(self, examples_dir: str) -> None:
        """
        Initialize the OSWorldProcessor.

        Args:
            examples_dir (str): Path to the directory containing the JSON examples.
        """
        self.examples_dir = examples_dir
        self.data = {}  # Dictionary to store all loaded data
        self.load_data()


    def load_data(self) -> None:
        """Load all JSON files into self.data."""
        for domain in os.listdir(self.examples_dir):
            domain_path = os.path.join(self.examples_dir, domain)
            if os.path.isdir(domain_path):  # Ensure it's a directory
                self.data[domain] = {}
                for task_file in os.listdir(domain_path):
                    if task_file.endswith(".json"):
                        task_id = os.path.splitext(task_file)[0]  # Get the task ID (filename without extension)
                        task_path = os.path.join(domain_path, task_file)
                        with open(task_path, "r") as f:
                            self.data[domain][task_id] = json.load(f)


    def get(self, domain: str = None, task_id: str = None) -> Any:
        """Retrieve data for a specific domain, task_id, or both.

        Args:
            domain (str): The domain to filter data by.
            task_id (str): The task ID to filter data by.

        Returns:
            Any: The data for the specified domain/task ID, or all data if no filters are applied.
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