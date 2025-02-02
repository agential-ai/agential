"""Base (WebVoyager) Agent strategy class."""

import re
import time
import pytest

from typing import Any, Dict, Optional

from agential.agents.computer_use.webvoyager_baseline.functional import (
    clip_message_and_obs,
    clip_message_and_obs_text_only,
)
from agential.agents.computer_use.webvoyager_baseline.output import WebVoyagerBaseOutput
from agential.agents.computer_use.webvoyager_baseline.strategies.general import WebVoyagerGeneralStrategy
from agential.agents.computer_use.webvoyager_baseline.strategies.base import WebVoyagerBaseStrategy

from agential.core.llm import BaseLLM, Response, MockLLM

def test__init__(self, llm: BaseLLM, testing: bool = False) -> None:
    """Test WebVoyagerGeneralStrategy initialization."""
    responses = [
        '```json\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```'
    ]
    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)
    assert strategy.testing == True
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy, WebVoyagerBaseStrategy)
    
@pytest.fixture
def sample_data_format_msg():
    return {
        "it": 1,
        "init_msg": "Initial message.",
        "pdf_obs": "Sample PDF observation.",
        "warn_obs": "Sample warning observation.",
        "web_img_b64": "base64image==",
        "web_text": "Sample web text."
    }

def test_format_msg(sample_data_format_msg: Dict[str, Any]):
    responses = [
        '```json\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```'
    ]
    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)
    
    result = strategy.format_msg(
        sample_data_format_msg['it'],
        sample_data_format_msg["init_msg"],
        sample_data_format_msg["pdf_obs"],
        sample_data_format_msg["warn_obs"],
        sample_data_format_msg["web_img_b64"],
        sample_data_format_msg["web_text"]
    )

    assert result["role"] == "user"
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "text"
    assert sample_data_format_msg["web_text"] in result["content"][0]["text"]
    assert result["content"][1]["type"] == "image_url"
    assert result["content"][1]["image_url"]["url"] == f"data:image/png;base64,{sample_data_format_msg['web_img_b64']}"

    sample_data_format_msg["it"] = 2
    sample_data_format_msg["pdf_obs"] = ""

    result = strategy.format_msg(
        it=sample_data_format_msg["it"],
        init_msg=sample_data_format_msg["init_msg"],
        pdf_obs=sample_data_format_msg["pdf_obs"],
        warn_obs=sample_data_format_msg["warn_obs"],
        web_img_b64=sample_data_format_msg["web_img_b64"],
        web_text=sample_data_format_msg["web_text"]
    )

    assert result["role"] == "user"
    assert "Observation:" in result["content"][0]["text"]
    assert sample_data_format_msg["warn_obs"] in result["content"][0]["text"]
    assert result["content"][1]["image_url"]["url"] == f"data:image/png;base64,{sample_data_format_msg['web_img_b64']}"

    sample_data_format_msg["it"] = 2

    result = strategy.format_msg(
        it=sample_data_format_msg["it"],
        init_msg=sample_data_format_msg["init_msg"],
        pdf_obs=sample_data_format_msg["pdf_obs"],
        warn_obs=sample_data_format_msg["warn_obs"],
        web_img_b64=sample_data_format_msg["web_img_b64"],
        web_text=sample_data_format_msg["web_text"]
    )

    assert result["role"] == "user"
    assert "Observation:" in result["content"][0]["text"]
    assert sample_data_format_msg["pdf_obs"] in result["content"][0]["text"]
    assert result["content"][1]["image_url"]["url"] == f"data:image/png;base64,{sample_data_format_msg['web_img_b64']}"
    
@pytest.fixture
def sample_data_format_msg_text_only():
    return {
        "it": 1,
        "init_msg": "Initial message.",
        "pdf_obs": "Sample PDF observation.",
        "warn_obs": "Sample warning observation.",
        "ac_tree": "Accessibility tree data."
    }

def test_format_msg_text_only(sample_data_format_msg_text_only: Dict[str, Any]) -> None:
    responses = [
        '```json\n{\n  "action_type": "CLICK",\n  "x": 300,\n  "y": 200\n}\n```'
    ]
    llm = MockLLM("gpt-4o", responses=responses)
    strategy = WebVoyagerGeneralStrategy(llm=llm, testing=True)

    result = strategy.format_msg_text_only(
        it=sample_data_format_msg_text_only["it"],
        init_msg=sample_data_format_msg_text_only["init_msg"],
        pdf_obs=sample_data_format_msg_text_only["pdf_obs"],
        warn_obs=sample_data_format_msg_text_only["warn_obs"],
        ac_tree=sample_data_format_msg_text_only["ac_tree"]
    )
    
    assert result["role"] == "user"
    assert result["content"] == sample_data_format_msg_text_only["init_msg"] + "\n" + sample_data_format_msg_text_only["ac_tree"]

    sample_data_format_msg_text_only["it"] = 2
    sample_data_format_msg_text_only["pdf_obs"] = ""  # No PDF observation, just warning observation
    result = strategy.format_msg_text_only(
        it=sample_data_format_msg_text_only["it"],
        init_msg=sample_data_format_msg_text_only["init_msg"],
        pdf_obs=sample_data_format_msg_text_only["pdf_obs"],
        warn_obs=sample_data_format_msg_text_only["warn_obs"],
        ac_tree=sample_data_format_msg_text_only["ac_tree"]
    )
    
    assert result["role"] == "user"
    assert result["content"] == f"Observation:{sample_data_format_msg_text_only['warn_obs']} please analyze the accessibility tree and give the Thought and Action.\n{sample_data_format_msg_text_only['ac_tree']}"

    sample_data_format_msg_text_only["it"] = 2
    result = strategy.format_msg_text_only(
        it=sample_data_format_msg_text_only["it"],
        init_msg=sample_data_format_msg_text_only["init_msg"],
        pdf_obs=sample_data_format_msg_text_only["pdf_obs"],
        warn_obs=sample_data_format_msg_text_only["warn_obs"],
        ac_tree=sample_data_format_msg_text_only["ac_tree"]
    )
    
    assert result["role"] == "user"
    assert result["content"] == f"Observation:{sample_data_format_msg_text_only['warn_obs']} please analyze the accessibility tree and give the Thought and Action.\n{sample_data_format_msg_text_only['ac_tree']}"
    

# def generate_thought(
#     self,
#     messages: list[Any],
#     seed: Optional[int],
#     max_tokens: int = 1000,
#     timeout: int = 30,
# ) -> Response:
#     """Generates a thought response using the specified model and input payload.

#     Args:
#         messages (list): The input messages for the model.
#         max_tokens (int): The maximum number of tokens for the response.
#         seed (Optional[int]): The seed for reproducibility in random operations.
#         timeout (Optional[float]): The maximum time in seconds to wait for a response.


#     Returns:
#         Response: The generated output text from the model.
#     """
#     response = self.llm(messages, max_tokens, seed, timeout)

#     return response

# def generate(
#     self,
#     system_prompt: str,
#     system_prompt_text_only: str,
#     seed: int,
#     max_attached_imgs: int,
#     temperature: float,
#     text_only: bool,
#     task: Dict[str, Any],
#     obs: Dict[str, Any]
# ) -> WebVoyagerBaseOutput:
#     start_time = time.time()

#     pattern = r"Thought:|Action:|Observation:"

#     messages = [{"role": "system", "content": system_prompt}]
#     obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
#     if text_only:
#         messages = [{"role": "system", "content": system_prompt_text_only}]
#         obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

#     init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
#     init_msg = init_msg.replace("https://www.example.com", task["web"])
#     init_msg = init_msg + obs_prompt

#     it = 0

#     if not text_only:
#         curr_msg = self.format_msg(
#             it, init_msg, obs.pdf_obs, obs.warn_obs, obs.encoded_image_som, obs.web_eles_text
#         )
#     else:
#         curr_msg = self.format_msg_text_only(
#             it, init_msg, obs.pdf_obs, obs.warn_obs, obs.ac_tree
#         )
#     messages.append(curr_msg)

#     # Clip messages, too many attached images may cause confusion
#     if not text_only:
#         messages = clip_message_and_obs(messages, max_attached_imgs)
#     else:
#         messages = clip_message_and_obs_text_only(
#             messages, max_attached_imgs
#         )

#     response = self.generate_thought(messages=messages, seed=seed)
#     prompt_tokens = response.prompt_tokens
#     completion_tokens = response.completion_tokens
#     gpt_4v_res = response.output_text

#     messages.append({"role": "assistant", "content": gpt_4v_res})

#     # extract action info
#     try:
#         assert "Thought:" in gpt_4v_res and "Action:" in gpt_4v_res
#     except AssertionError as e:
#         print(e)            

#     action = re.split(pattern, gpt_4v_res)[2].strip()
#     thought = re.split(pattern, gpt_4v_res)[1].strip()
#     observation = re.split(pattern, gpt_4v_res)[3].strip()

#     end_time = time.time()

#     return WebVoyagerBaseOutput(
#         answer=response.output_text,
#         total_prompt_tokens=prompt_tokens,
#         total_completion_tokens=completion_tokens,
#         total_tokens=response.total_tokens,
#         total_prompt_cost=response.prompt_cost,
#         total_completion_cost=response.completion_cost,
#         total_cost=response.total_cost,
#         total_prompt_time=response.prompt_time,
#         total_time=end_time - start_time,
#         additional_info={
#             "response": response.output_text,
#             "actions": action,
#             "thoughts": thought,
#             "observations": observation,
#             "messages": messages,
#         },
#     )

# def reset(  ######## Fix documentation #############
#     self, *args: Any, **kwargs: Any
# ) -> None:
#     """Resets the agent's internal state, including actions, thoughts, and observations.

#     Args:
#         actions (List[Dict[str, Any]]): The list of past actions to reset.
#         thought (List[str]): The list of past thoughts to reset.
#         observations (List[Any]): The list of past observations to reset.

#     Returns:
#         Tuple[List[str], List[Dict[str, Any]], List[Any]]: A tuple containing the reset actions, thoughts, and observations.
#     """
#     return None
