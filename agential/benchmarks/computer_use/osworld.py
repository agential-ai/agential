"""OSWorld Benchmark."""

from desktop_env.desktop_env import DesktopEnv

example = {}

env = DesktopEnv(action_space="pyautogui")

obs = env.reset(task_config=example)
obs, reward, done, info = env.step("pyautogui.rightClick()")


class OSWorld(BaseBenchmark, DesktopEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
