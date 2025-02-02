"""Base (WebVoyager) Agent strategy class."""

import re
import time

from typing import Any, Dict, Optional

from agential.agents.computer_use.webvoyager_baseline.functional import (
    clip_message_and_obs,
    clip_message_and_obs_text_only,
)
from agential.agents.computer_use.webvoyager_baseline.output import WebVoyagerBaseOutput
from agential.agents.computer_use.webvoyager_baseline.strategies.base import (
    WebVoyagerBaseStrategy,
)
from agential.core.llm import BaseLLM, Response


class WebVoyagerGeneralStrategy(WebVoyagerBaseStrategy):
    """A strategy class for the Web Voyager Agent.

    This class defines methods for generating actions, thoughts, and observations
    in an agent-based environment, specifically tailored to the Web Voyager Agent.
    """

    def __init__(self, llm: BaseLLM, testing: bool = False) -> None:
        """Initializes the WebVoyagerBaseStrategy with the provided language model and testing flag.

        Args:
            llm (BaseLLM): The language model used for generating answers and critiques.
            testing (bool): Whether the generation is for testing purposes. Defaults to False.
        """
        super().__init__(llm=llm, testing=testing)

    def format_msg(
        self, 
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
        if it == 1:
            init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
            init_msg_format = {
                "role": "user",
                "content": [
                    {"type": "text", "text": init_msg},
                ],
            }
            init_msg_format["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{web_img_b64}"},
                }
            )
            return init_msg_format
        else:
            if not pdf_obs:
                curr_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{web_img_b64}"
                            },
                        },
                    ],
                }
            else:
                curr_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{web_img_b64}"
                            },
                        },
                    ],
                }
            return curr_msg

    def format_msg_text_only(
        self,
        it: int, 
        init_msg: str, 
        pdf_obs: str, 
        warn_obs: str, 
        ac_tree: str
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
        if it == 1:
            init_msg_format = {"role": "user", "content": init_msg + "\n" + ac_tree}
            return init_msg_format
        else:
            if not pdf_obs:
                curr_msg = {
                    "role": "user",
                    "content": f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}",
                }
            else:
                curr_msg = {
                    "role": "user",
                    "content": f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}",
                }
            return curr_msg

    def generate_thought(
        self,
        messages: list[Any],
        seed: Optional[int],
        max_tokens: int = 1000,
        timeout: int = 30,
    ) -> Response:
        """Generates a thought response using the specified model and input payload.

        Args:
            messages (list): The input messages for the model.
            max_tokens (int): The maximum number of tokens for the response.
            seed (Optional[int]): The seed for reproducibility in random operations.
            timeout (Optional[float]): The maximum time in seconds to wait for a response.


        Returns:
            Response: The generated output text from the model.
        """
        response = self.llm(messages, max_tokens, seed, timeout)

        return response

    def generate(
        self,
        system_prompt: str,
        system_prompt_text_only: str,
        seed: int,
        max_attached_imgs: int,
        temperature: float,
        text_only: bool,
        task: Dict[str, Any],
        obs: Dict[str, Any]
    ) -> WebVoyagerBaseOutput:
        start_time = time.time()

        pattern = r"Thought:|Action:|Observation:"

        messages = [{"role": "system", "content": system_prompt}]
        obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
        if text_only:
            messages = [{"role": "system", "content": system_prompt_text_only}]
            obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

        init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
        init_msg = init_msg.replace("https://www.example.com", task["web"])
        init_msg = init_msg + obs_prompt

        it = 0

        if not text_only:
            curr_msg = self.format_msg(
                it, init_msg, obs.pdf_obs, obs.warn_obs, obs.encoded_image_som, obs.web_eles_text
            )
        else:
            curr_msg = self.format_msg_text_only(
                it, init_msg, obs.pdf_obs, obs.warn_obs, obs.ac_tree
            )
        messages.append(curr_msg)

        # Clip messages, too many attached images may cause confusion
        if not text_only:
            messages = clip_message_and_obs(messages, max_attached_imgs)
        else:
            messages = clip_message_and_obs_text_only(
                messages, max_attached_imgs
            )

        response = self.generate_thought(messages=messages, seed=seed)
        prompt_tokens = response.prompt_tokens
        completion_tokens = response.completion_tokens
        gpt_4v_res = response.output_text

        messages.append({"role": "assistant", "content": gpt_4v_res})

        # extract action info
        try:
            assert "Thought:" in gpt_4v_res and "Action:" in gpt_4v_res
        except AssertionError as e:
            print(e)            

        action = re.split(pattern, gpt_4v_res)[2].strip()
        thought = re.split(pattern, gpt_4v_res)[1].strip()
        observation = re.split(pattern, gpt_4v_res)[3].strip()

        end_time = time.time()

        return WebVoyagerBaseOutput(
            answer=response.output_text,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            total_tokens=response.total_tokens,
            total_prompt_cost=response.prompt_cost,
            total_completion_cost=response.completion_cost,
            total_cost=response.total_cost,
            total_prompt_time=response.prompt_time,
            total_time=end_time - start_time,
            additional_info={
                "response": response.output_text,
                "actions": action,
                "thoughts": thought,
                "observations": observation,
                "messages": messages,
            },
        )

    def reset(  ######## Fix documentation #############
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Resets the agent's internal state, including actions, thoughts, and observations.

        Args:
            actions (List[Dict[str, Any]]): The list of past actions to reset.
            thought (List[str]): The list of past thoughts to reset.
            observations (List[Any]): The list of past observations to reset.

        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[Any]]: A tuple containing the reset actions, thoughts, and observations.
        """
        return None
