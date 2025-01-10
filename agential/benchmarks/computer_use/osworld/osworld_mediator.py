"""OSWorld bridging OSWorldBenchmark and OSWorldDataLoader Retriever."""

import random
from typing import Any, Dict

from agential.benchmarks.computer_use.osworld.osworld_data_loader import (
    OSWorldDataLoader,
)
from agential.benchmarks.computer_use.osworld.osworld_benchmark import OSWorldBenchmark

TYPE_TO_LOOK = ["googledrive", "login", "googledrive_file"]


class OSWorldMediator:
    """A class to manage and process tasks within the OSWorldMediator environment.

    This class provides functionality to update credentials, reset tasks,
    and interact with the OSWorldMediator environment using the provided processor.

    Attributes:
        examples_dir (str): Directory containing example configurations.
        path_to_google_settings (str): Path to Google-specific settings.
        path_to_googledrive_settings (str): Path to GoogleDrive-specific settings.
        osworld_benchmark (OSWorldBenchmark): Processor for executing tasks in OSWorld.
    """

    def __init__(
        self,
        examples_dir: str,
        path_to_google_settings: str,
        path_to_googledrive_settings: str,
        osworld_benchmark: OSWorldBenchmark,
    ):
        """Initializes the OSWorldMediator instance with the specified settings and processor.

        Args:
            examples_dir (str): Directory containing example configurations.
            path_to_google_settings (str): Path to Google-specific settings.
            path_to_googledrive_settings (str): Path to Google Drive-specific settings.
            osworld_benchmark (OSWorldBenchmark): Processor for OSWorld tasks.
        """
        self.examples_dir = examples_dir
        self.path_to_google_settings = path_to_google_settings
        self.path_to_googledrive_settings = path_to_googledrive_settings

        self.osworld_data_loader = OSWorldDataLoader(self.examples_dir)
        self.data = self.osworld_data_loader.data
        self._update_credential()
        self.osworld_benchmark = osworld_benchmark

    def _change_credential(self, example: Dict[str, Any]) -> Any:
        """Modifies credential settings in a given example based on file type.

        Args:
            example (Dict[str, Any]): The task configuration to be updated.

        Returns:
            Dict[str, Any]: The updated task configuration.
        """
        try:
            for item in example["config"]:
                if item["type"] in TYPE_TO_LOOK:
                    file_type = item["parameters"]["settings_file"].split(".")[-1]
                    if file_type == "yml":
                        item["parameters"][
                            "settings_file"
                        ] = self.path_to_googledrive_settings
                    else:
                        item["parameters"][
                            "settings_file"
                        ] = self.path_to_google_settings

            path = example["evaluator"]["result"]
            if (
                path["type"] in TYPE_TO_LOOK
                and path["settings_file"].split(".")[-1] == "yml"
            ):
                path["settings_file"] = self.path_to_googledrive_settings

        except (KeyError, TypeError, AttributeError) as e:
            return example

        return example

    def _update_credential(self) -> None:
        """Updates credentials for the specified domain and/or task.

        Args:
            domain (str, optional): The domain whose tasks' credentials should be updated.
            task_id (str, optional): The task ID to update credentials for.

        Returns:
            Dict[str, Any]: The updated credentials for the specified domain and/or task.
        """
        for each_domain in self.data.keys():
            for each_task in self.data[each_domain].keys():
                self.data[each_domain][each_task] = self._change_credential(
                    self.data[each_domain][each_task]
                )

    def reset(self, domain: str, example: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Resets the OSWorld environment for the specified domain and/or task.

        Args:
            domain (str, optional): The domain to reset.
            task_id (str, optional): The task ID to reset.

        Returns:
            Dict[str, Any]: The result of the reset operation.
        """
        if not example:
            task_id = random.choice(list(self.data[domain].keys()))
            example = self.data[domain][task_id]
        return self.osworld_benchmark.reset(task_config=example)
