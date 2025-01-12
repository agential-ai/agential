"""OSWorldBenchmark Benchmark."""

import os
from typing import Any, Dict, Tuple

from glob import glob
import json

from desktop_env.desktop_env import DesktopEnv

from agential.benchmarks.computer_use.base import BaseComputerUseBenchmark
from agential.benchmarks.computer_use.osworld.osworld_data_loader import OSWorldDataLoader


class OSWorld(BaseComputerUseBenchmark):
    """The OSWorldBenchmark benchmark class simulates an environment for evaluating computer-use tasks.
    This class interacts with the `DesktopEnv` to simulate user interactions within an operating system,
    enabling the evaluation of tasks such as GUI navigation, application usage, and system interactions.

    This class extends the `BaseComputerUseBenchmark` and implements the abstract methods
    to manage the benchmark lifecycle, including initialization, task execution, resetting, and evaluation.

    Parameters:
        examples_dir (str): The directory containing the benchmark examples. Defaults to "" (or the benchmark examples) if nothing provided.
        test_type (str): The type of test to run. This parameter is used if examples_dir is "", which implies to use the benchmark tasks. Defaults to "test_all".
        path_to_google_settings (str): The path to the Google settings file. Required for multi-app tasks. Defaults to "".
        path_to_googledrive_settings (str): The path to the Google Drive settings file. Required for multi-app tasks. Defaults to "".
        **kwargs (Any): Configuration parameters passed to the `DesktopEnv` initialization
            and the parent `BaseComputerUseBenchmark` class.
    """

    def __init__(
        self, 
        examples_dir: str = "", 
        test_type: str = "test_all", 
        path_to_google_settings: str = "",
        path_to_googledrive_settings: str = "",
        **kwargs: Any
    ) -> None:
        """Initialization."""
        super().__init__(**kwargs)

        # Options:
        # - custom example dir
        #   - test_type doesn't matter
        # - no examples dir (default to benchmark examples)
        #   - test_type matters; default to test_all
        #   - update credentials

        self.examples_dir = examples_dir
        self.test_type = test_type
        self.path_to_google_settings = path_to_google_settings
        self.path_to_googledrive_settings = path_to_googledrive_settings

        # Get data loader.
        if self.examples_dir:
            self.osworld_data_loader = OSWorldDataLoader(self.examples_dir)
        else:  # Use benchmark examples.
            current_file_path = os.path.dirname(__file__)
            evaluation_examples_path = os.path.join(current_file_path, "evaluation_examples")
            examples_path = os.path.join(evaluation_examples_path, "examples")

            self.osworld_data_loader = OSWorldDataLoader(examples_path)
            test_file: str = os.path.join(evaluation_examples_path, f"{self.test_type}.json")
            
            self.tasks = {}
            try:
                with open(test_file, "r") as f:
                    self.tasks = json.load(f)
            except FileNotFoundError:
                task_set_options = [
                    os.path.splitext(os.path.basename(file))[0] for file in glob(os.path.join(evaluation_examples_path, "test_*.json"))
                ]
                raise FileNotFoundError(f"Using benchmark tasks and task set {test_file} not found. Available options: {', '.join(task_set_options)}.")


        # Instantiate environment.
        # try:
        #     self.env = DesktopEnv(**kwargs)
        # except:
        #     base_path = os.path.abspath(".")
        #     vmware_vm_data_path = os.path.join(base_path, "vmware_vm_data")

        #     ubuntu_folders = sorted(
        #         [
        #             folder for folder in glob(os.path.join(vmware_vm_data_path, "Ubuntu*"))
        #             if os.path.isdir(folder)
        #         ],
        #         key=os.path.getmtime,
        #         reverse=True
        #     )

        #     if not ubuntu_folders:
        #         raise FileNotFoundError("No Ubuntu# folders found in the vmware_vm_data directory.")

        #     latest_ubuntu_folder = ubuntu_folders[0]
        #     print(f"Using Ubuntu VM folder: {latest_ubuntu_folder}")

        #     vmx_file = glob(os.path.join(latest_ubuntu_folder, "*.vmx"))
        #     if not vmx_file:
        #         raise FileNotFoundError(f"No .vmx file found in the folder: {latest_ubuntu_folder}")

        #     path_to_vm = vmx_file[0]
        #     drive, rest = os.path.splitdrive(path_to_vm)
        #     path_to_vm = drive.upper() + rest

        #     print(f"Initializing DesktopEnv with VM path: {path_to_vm}")
        #     kwargs['path_to_vm'] = path_to_vm
        #     self.env = DesktopEnv(**kwargs)
        #     print("DesktopEnv initialized successfully.")

    def close(self) -> None:
        """Closes the benchmark environment and any associated resources.

        This method shuts down the `DesktopEnv` instance, ensuring that any resources,
        such as running applications or simulated processes, are properly closed.
        """
        self.env.close()

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """Resets the environment to its initial state for a new evaluation.

        This method prepares the `DesktopEnv` for a new round of benchmarking by resetting
        the environment, clearing any state, and applying the provided task configuration.

        Args:
            task_config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                     for the task (default is None).
            seed (Optional[int]): The seed value for random number generation (default is None).
            options (Optional[Dict]): Additional options for resetting the environment (default is None).

        Returns:
            obs (Dict[str, Any]): The updated state or configuration of the environment after reset.
        """
        return self.env.reset(**kwargs)

    def step(
        self, **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, bool]]:
        """Executes a single step in the benchmark task.

        This method performs an action within the `DesktopEnv`, such as interacting
        with the environment, navigating the GUI, or simulating system-level tasks.
        After each step, the environment may be updated.

        Args:
            action: The action to be performed by the agent within the environment.
            pause (int, default=2): The number of seconds to pause after performing the action.

        Returns:
            obs (Dict[str, Any]): Observation of the scence such as screenshot, accessibility tree.
            reward (float): Reward based on how the agent performs in order to guide toward the goal.
            done (bool): If agent is at the done state.
            info (Dict[str, bool]): Inormation such as is the agent is done, failed, etc.
        """
        return self.env.step(**kwargs)

    def evaluate(self) -> float:
        """Evaluates the current state of the environment or task.

        This method triggers the evaluation of the environment, which could include
        performance metrics, task completion assessment, or other evaluation criteria
        based on the `DesktopEnv`.

        Returns:
            metric (float): An evaluation of how well the agent performs on an instruction.
        """
        return self.env.evaluate()

    def render(self) -> bytes:
        """Renders the environment's current state for visualization purposes.

        This method displays or visualizes the current state of the `DesktopEnv`,
        which can be useful for debugging or understanding the agent's progress.

        Returns:
            bytes: A bytes object containing the rendered image data.
        """
        return self.env.render()
