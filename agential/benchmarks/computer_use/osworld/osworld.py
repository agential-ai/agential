"""OSWorld Benchmark."""

import subprocess

from typing import Any

from desktop_env.desktop_env import DesktopEnv

from agential.benchmarks.computer_use.base import BaseComputerUseBenchmark

import os
import subprocess

class OSWorld(BaseComputerUseBenchmark):
    """The OSWorld benchmark class simulates an environment for evaluating computer-use tasks.
    This class interacts with the `DesktopEnv` to simulate user interactions within an operating system,
    enabling the evaluation of tasks such as GUI navigation, application usage, and system interactions.

    This class extends the `BaseComputerUseBenchmark` and implements the abstract methods
    to manage the benchmark lifecycle, including initialization, task execution, resetting, and evaluation.

    Parameters:
        path_to_google_settings (str): The path to the Google settings.json file.
        path_to_googledrive_settings (str): The path to the Google Drive settings.yml file.
        **kwargs (Any): Configuration parameters passed to the `DesktopEnv` initialization
            and the parent `BaseComputerUseBenchmark` class.
    """

    def __init__(
        self, 
        path_to_google_settings: str, 
        path_to_googledrive_settings: str, 
        **kwargs: Any
    ) -> None:
        """Initialization."""
        super().__init__(**kwargs)

        self.path_to_google_settings = path_to_google_settings
        self.path_to_googledrive_settings = path_to_googledrive_settings

        self.path_to_vm = kwargs.get("path_to_vm")

        try:
            # If the provided vmware_vm_data path does not exist, delete it from the kwargs.
            if self.path_to_vm is not None and not os.path.exists(self.path_to_vm):
                del kwargs["path_to_vm"]
            self.env = DesktopEnv(**kwargs)
        except:
            try:
                vmrun_command = f"vmrun start {self.path_to_vm}"
                subprocess.run(vmrun_command, check=True)
                self.env = DesktopEnv(**kwargs)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e}")

    def close(self) -> None:
        """Closes the benchmark environment and any associated resources.

        This method shuts down the `DesktopEnv` instance, ensuring that any resources,
        such as running applications or simulated processes, are properly closed.
        """
        self.env.close()

    def reset(self, **kwargs: Any) -> Any:
        """Resets the environment to its initial state for a new evaluation.

        This method prepares the `DesktopEnv` for a new round of benchmarking by resetting
        the environment, clearing any state, and applying the provided task configuration.

        Args:
            task_config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                     for the task (default is None).
            seed (Optional[int]): The seed value for random number generation (default is None).
            options (Optional[Dict]): Additional options for resetting the environment (default is None).

        Returns:
            dict: The updated state or configuration of the environment after reset.
        """
        return self.env.reset(**kwargs)

    def step(self, **kwargs: Any) -> Any:
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
            info (Dict[str, Any]): Inormation such as is the agent is done, failed, etc.
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
