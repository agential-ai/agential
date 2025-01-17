"""WebVoyager benchmark."""

from typing import Any
from agential.benchmarks.computer_use.base import BaseComputerUseBenchmark
from selenium import webdriver

def driver_config(args):
    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )
    return options

class WebVoyager(BaseComputerUseBenchmark):
    def __init__(self) -> None:
        options = driver_config(args)

    def close(self) -> None:
        pass

    def reset(self, **kargs: Any) -> Any:
        pass

    def step(self, action: Any) -> Any:
        pass

