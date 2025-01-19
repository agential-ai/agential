"""Base (WebVoyager) Agent strategy class."""

from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from argparse import Namespace
from typing import Dict, Any, Tuple, Optional

from agential.agents.base.strategies import BaseAgentStrategy
from agential.core.llm import BaseLLM, Response
from agential.agents.computer_use.webvoyager_baseline.output import WebVoyagerBaseOutput


class WebVoyagerBaseStrategy(BaseAgentStrategy):
    """An abstract base class for defining strategies for the Web Voyager Agent.

    This class provides a foundation for creating strategies tailored to the Web Voyager Agent
    by inheriting from the BaseAgentStrategy. It allows interaction with the language model (LLM)
    for generating answers and critiques, with testing capabilities.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(self, llm: BaseLLM, testing: bool = False) -> None:
        """Initializes the WebVoyagerBaseStrategy with the provided language model and testing flag.

        Args:
            llm (BaseLLM): The language model used for generating answers and critiques.
            testing (bool): Whether the generation is for testing purposes. Defaults to False.
        """
        super().__init__(llm, testing)

    def setup_logger(folder_path: str) -> None:
        """Sets up a logger to record logs in a file named 'agent.log' inside the specified folder.

        Args:
            folder_path (str): The directory path where the log file will be saved.

        Returns:
            None

        This function creates a new log file or overwrites an existing one and sets the logging level to INFO.
        """
        raise NotImplementedError

    def driver_config(args: Namespace) -> Tuple[webdriver.ChromeOptions, bool]:
        """Configures options for the Chrome WebDriver based on the provided arguments.

        Args:
            args (Namespace): Command-line arguments that include configuration settings for the WebDriver.

        Returns:
            Tuple[webdriver.ChromeOptions, bool]: A configured ChromeOptions instance for use with Selenium WebDriver.

        This function configures browser options for headless mode, device scaling, and custom preferences for downloads.
        """
        raise NotImplementedError

    def format_msg(
        it: int,
        init_msg: str,
        pdf_obs: str,
        warn_obs: str,
        web_img_b64: str,
        web_text: str,
    ) -> Dict[str, str]:
        """Formats the message to be sent to the GPT model, including a screenshot and relevant observations.

        Args:
            it (int): The iteration number.
            init_msg (str): The initial message to be sent.
            pdf_obs (str): Observations related to PDF files, if any.
            warn_obs (str): Warnings related to the action, if any.
            web_img_b64 (str): Base64 encoded image of the screenshot.
            web_text (str): Text content from the webpage.

        Returns:
            dict: A dictionary representing the formatted message for the GPT model.

        This function formats the message based on the iteration number and includes either a screenshot or accessibility tree, along with observations.
        """
        raise NotImplementedError

    def format_msg_text_only(
        it: int, init_msg: str, pdf_obs: str, warn_obs: str, ac_tree: str
    ) -> Dict[str, str]:
        """Formats a message with only text content, including the accessibility tree and relevant observations.

        Args:
            it (int): The iteration number.
            init_msg (str): The initial message to be sent.
            pdf_obs (str): Observations related to PDF files, if any.
            warn_obs (str): Warnings related to the action, if any.
            ac_tree (str): The accessibility tree in text format.

        Returns:
            dict: A dictionary representing the formatted message for the GPT model.

        This function formats the message based on the iteration number and includes the accessibility tree in text format, along with observations.
        """

        raise NotImplementedError

    def generate(
        self,
        system_prompt: str,
        system_prompt_text_only: str,
        output_dir: str,
        download_dir: str,
        test_file: str,
        max_iter: int,
        seed: int,
        max_attached_imgs: int,
        temperature: float,
        text_only: bool,
        headless: bool,
        save_accessibility_tree: bool,
        force_device_scale: bool,
        window_width: int,
        window_height: int,
        fix_box_color: bool,
    ) -> WebVoyagerBaseOutput:
        raise NotImplementedError

    def generate_thought(
        self,
        messages: list[Any],
        seed: Optional[int],
        max_tokens: int = 1000,
        timeout: int = 30,
    ) -> Response:
        """Generates a thought response using the specified model and input payload.

        Args:
            payload (Dict): A dictionary containing the input parameters for the model, including:
                - "messages" (list): The input messages for the model.
                - "max_tokens" (int): The maximum number of tokens for the response.
                - "temperature" (float): The sampling temperature for response generation.
                - "top_p" (float): The nucleus sampling parameter.
            model (str): The model used to generate the response.

        Returns:
            Response: The generated output text from the model.
        """
        raise NotImplementedError

    def exec_action_click(
        info: Dict[str, Any], web_ele: WebElement, driver_task: webdriver
    ) -> None:
        """
        Executes a click action on the specified web element using Selenium WebDriver.

        Args:
            info (dict): Information related to the action to be performed.
            web_ele (WebElement): The web element to be clicked.
            driver_task (WebDriver): The Selenium WebDriver instance executing the action.

        Returns:
            None

        This function sets the target attribute of the element to '_self' and performs a click, followed by a short wait.
        """
        raise NotImplementedError

    def exec_action_type(
        info: Dict[str, Any], web_ele: WebElement, driver_task: webdriver
    ) -> None:
        """
        Types content into the specified web element (input or textarea) using Selenium WebDriver.

        Args:
            info (dict): Information related to the action, including the content to be typed.
            web_ele (WebElement): The web element (input or textarea) where content will be typed.
            driver_task (WebDriver): The Selenium WebDriver instance executing the action.

        Returns:
            str: A warning message if the element is not a valid textbox, otherwise an empty string.

        This function clears the existing content in the element, types the provided content, and performs the action.
        """
        raise NotImplementedError

    def exec_action_scroll(
        info: Dict[str, Any],
        web_ele: WebElement,
        driver_task: webdriver,
        args: Namespace,
        obs_info: Dict[str, Any],
    ) -> None:
        """
        Executes a scroll action on the webpage, either scrolling the window or a specific element.

        Args:
            info (dict): Information related to the scroll action.
            web_eles (list): A list of web elements to scroll.
            driver_task (WebDriver): The Selenium WebDriver instance executing the action.
            args (Namespace): Command-line arguments containing configuration settings.
            obs_info (dict): Observations related to the web elements.

        Returns:
            None

        This function performs a scroll action either on the whole window or on a specific element identified by the provided index.
        """
        raise NotImplementedError

    def reset(  ######## Fix documentation #############
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Resets the agent's internal state, including actions, thoughts, and observations.

        Args:
            actions (List[Dict[str, Any]]): The list of past actions to reset.
            thought (List[str]): The list of past thoughts to reset.
            observations (List[Any]): The list of past observations to reset.

        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[Any]]: A tuple containing the reset actions, thoughts, and observations.
        """
        raise NotImplementedError
