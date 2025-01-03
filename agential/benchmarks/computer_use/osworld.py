"""OSWorld Benchmark."""

from typing import Any, Dict, Optional
from desktop_env.desktop_env import DesktopEnv

example = {}

env = DesktopEnv(action_space="pyautogui")

obs = env.reset(task_config=example)
obs, reward, done, info = env.step("pyautogui.rightClick()")


class OSWorld(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env = DesktopEnv(**kwargs)

    def close(self):
        self.env.close()

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        self.env.reset()

    def step(self, action, pause=2):
        self.env.step() 

    def evaluate(self):
        self.env.evaluate()

    def render(self):
        self.env.render()