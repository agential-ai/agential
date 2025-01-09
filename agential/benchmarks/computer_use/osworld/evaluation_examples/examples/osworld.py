"""OSWorld bridging OSWorldProcessor and OSWorldEnv Retriever."""

from typing import Dict, Any

from agential.benchmarks.computer_use.osworld.evaluation_examples.examples.osworld_env import (
    OSWorldEnv,
)
from agential.benchmarks.computer_use.osworld.osworld_processor import OSWorldProcessor

TYPE_TO_LOOK = ["googledrive", "login", "googledrive_file"]


class OSWorld:
    """
    A class to manage and process tasks within the OSWorld environment.

    This class provides functionality to update credentials, reset tasks,
    and interact with the OSWorld environment using the provided processor.

    Attributes:
        examples_dir (str): Directory containing example configurations.
        path_to_google_settings (str): Path to Google-specific settings.
        osworld_env (OSWorldEnv): The OSWorld environment instance.
        data (Dict): The data extracted from the OSWorld environment.
        osworld_processor (OSWorldProcessor): Processor for executing tasks in OSWorld.
    """

    def __init__(
        self,
        examples_dir: str,
        path_to_google_settings: str,
        path_to_googledrive_settings: str,
        osworld_processor: OSWorldProcessor,
    ):
        """
        Initializes the OSWorld instance with the specified settings and processor.

        Args:
            examples_dir (str): Directory containing example configurations.
            path_to_google_settings (str): Path to Google-specific settings.
            path_to_googledrive_settings (str): Path to Google Drive-specific settings.
            osworld_processor (OSWorldProcessor): Processor for OSWorld tasks.
        """
        self.examples_dir = examples_dir
        self.path_to_google_settings = path_to_google_settings
        self.path_to_googledrive_settings = path_to_googledrive_settings

        self.osworld_env = OSWorldEnv(self.examples_dir)
        self.data = self.osworld_env.data
        self.osworld_processor = osworld_processor

    def _change_credential(self, example: Dict[str, Any]) -> Any:
        """
        Modifies credential settings in a given example based on file type.

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
                else:
                    item["parameters"]["settings_file"] = self.path_to_google_settings

        path = example["evaluator"]["result"]
        if (
            path["type"] in TYPE_TO_LOOK
            and path["settings_file"].split(".")[-1] == "yml"
        ):
            path["settings_file"] = self.path_to_googledrive_settings

        return example

    def update_credential(self, domain: str = "", task_id: str = "") -> Dict[str, Any]:
        """
        Updates credentials for the specified domain and/or task.

        Args:
            domain (str, optional): The domain whose tasks' credentials should be updated.
            task_id (str, optional): The task ID to update credentials for.

        Returns:
            Dict[str, Any]: The updated credentials for the specified domain and/or task.
        """
        temp_data: Dict[str, Any] = {}

        if domain is not None and task_id is not None:
            return self._change_credential(self.data[domain][task_id])
        elif domain is not None:
            for each_task in self.data[domain].keys():
                temp_data[domain][each_task] = self._change_credential(each_task)
            return temp_data
        elif task_id is not None:
            for each_domain in self.data.keys():
                temp_data[each_domain][task_id] = self._change_credential(
                    each_domain[task_id]
                )
            return temp_data
        else:
            for each_domain in self.data.keys():
                for each_task in self.data[each_domain].keys():
                    temp_data[each_domain][each_task] = self._change_credential(
                        each_task
                    )
            return temp_data

    def reset(self, domain: str = "", task_id: str = "") -> Dict[str, Any]:
        """
        Resets the OSWorld environment for the specified domain and/or task.

        Args:
            domain (str, optional): The domain to reset.
            task_id (str, optional): The task ID to reset.

        Returns:
            Dict[str, Any]: The result of the reset operation.
        """
        example = self.update_credential(domain=domain, task_id=task_id)
        return self.osworld_processor.reset(task_config=example)
