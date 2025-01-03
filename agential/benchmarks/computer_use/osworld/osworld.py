"""OSWorld Benchmark."""

from typing import Any, Dict, Optional, Tuple
from desktop_env.desktop_env import DesktopEnv

from agential.benchmarks.computer_use.base import BaseComputerUseBenchmark

example: Dict[str, Any] = {}

env = DesktopEnv(action_space="pyautogui")

obs = env.reset(task_config=example)
obs, reward, done, info = env.step("pyautogui.rightClick()")

# TODO: Write BaseBenchmark
# TODO: Write BaseComputerUseBenchmark
# TODO: Write OSWorld
#           - it's mostly done but we need a standard interface
# TODO: Documentation
#           - this one definitely needs lots of documentation on setup (best to reference the OSWorld code base)
#           - don't forget to credit the original code!
# TODO: Testing
# TODO: Linting, code coverage


class OSWorld(BaseComputerUseBenchmark):
    """
    The OSWorld benchmark class simulates an environment for evaluating computer-use tasks.
    This class interacts with the `DesktopEnv` to simulate user interactions within an operating system,
    enabling the evaluation of tasks such as GUI navigation, application usage, and system interactions.

    This class extends the `BaseComputerUseBenchmark` and implements the abstract methods
    to manage the benchmark lifecycle, including initialization, task execution, resetting, and evaluation.

    Attributes:
        env (DesktopEnv): An instance of the `DesktopEnv` class that represents the simulated environment.

    Methods:
        __init__(**kwargs):
            Initializes the OSWorld benchmark with the given configuration parameters.

        close():
            Closes the benchmark environment and any associated resources.

        reset(task_config=None, seed=None, options=None):
            Resets the environment to its initial state for a new evaluation.

        step(action, pause=2):
            Executes a single step of the benchmark task, performing an action within the environment.

        evaluate():
            Evaluates the current state of the environment or task.

        render():
            Renders the environment's current state for visualization purposes.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the OSWorld benchmark with the provided configuration parameters.

        Args:
            **kwargs: Configuration parameters passed to the `DesktopEnv` initialization
                      and the parent `BaseComputerUseBenchmark` class.
        """
        super().__init__(**kwargs)
        self.env = DesktopEnv(**kwargs)

    def close(self) -> None:
        """
        Closes the benchmark environment and any associated resources.

        This method shuts down the `DesktopEnv` instance, ensuring that any resources,
        such as running applications or simulated processes, are properly closed.

        Raises:
            Any exception raised by the `DesktopEnv.close` method will propagate.
        """
        self.env.close()

    def reset(self, **kargs: Any) -> Any:
        """
        Resets the environment to its initial state for a new evaluation.

        This method prepares the `DesktopEnv` for a new round of benchmarking by resetting
        the environment, clearing any state, and applying the provided task configuration.

        Args:
            task_config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                     for the task (default is None).
            seed (Optional[int]): The seed value for random number generation (default is None).
            options (Optional[Dict]): Additional options for resetting the environment (default is None).

        Returns:
            dict: The updated state or configuration of the environment after reset.

        Raises:
            Any exception raised by the `DesktopEnv.reset` method will propagate.
        """
        return self.env.reset()

    def step(self, **kwargs: Any) -> Any:
        """
        Executes a single step in the benchmark task.

        This method performs an action within the `DesktopEnv`, such as interacting
        with the environment, navigating the GUI, or simulating system-level tasks.
        After each step, the environment may be updated.

        Args:
            action: The action to be performed by the agent within the environment.
            pause (int, default=2): The number of seconds to pause after performing the action.

        Raises:
            Any exception raised by the `DesktopEnv.step` method will propagate.
        """
        action = self.kwargs.get("action", None)
        return self.env.step(action)

    def evaluate(self) -> float:
        """
        Evaluates the current state of the environment or task.

        This method triggers the evaluation of the environment, which could include
        performance metrics, task completion assessment, or other evaluation criteria
        based on the `DesktopEnv`.

        Raises:
            Any exception raised by the `DesktopEnv.evaluate` method will propagate.
        """
        return self.env.evaluate()

    def render(self) -> Dict[str, Any]:
        """
        Renders the environment's current state for visualization purposes.

        This method displays or visualizes the current state of the `DesktopEnv`,
        which can be useful for debugging or understanding the agent's progress.

        Raises:
            Any exception raised by the `DesktopEnv.render` method will propagate.
        """
        return self.env.render()
