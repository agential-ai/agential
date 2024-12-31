import json
import logging
import os
import time
from dotenv import load_dotenv
from http import HTTPStatus
from typing import Any, Dict, List, Tuple

import dashscope
import google.generativeai as genai
import requests
from groq import Groq

from agential.agents.OSWorldBaseline.strategies.base import OSWorldBaselineAgentBaseStrategy
from agential.core.llm import BaseLLM
from litellm import completion
from dashscope.api_entities.dashscope_response import GenerationResponse

from agential.agents.OSWorldBaseline.functional import (
    encode_image,
    encoded_img_to_pil_img,
    save_to_tmp_img_file,
    linearize_accessibility_tree,
    tag_screenshot,
    parse_actions_from_string,
    parse_code_from_string,
    parse_code_from_som_string,
    trim_accessibility_tree
)

logger = logging.getLogger("desktopenv.agent")
pure_text_settings = ['a11y_tree']

class OSWorldBaselineAgentGeneralStrategy(OSWorldBaselineAgentBaseStrategy):

    def __init__(
        self, 
        testing: bool = False
    ) -> None:
        super().__init__(testing=testing)
        self.messages: List = []

    def generate(
        self, 
        platform: str,
        model: str,
        max_tokens: int,
        top_p: float,
        temperature: float,
        action_space: str,
        observation_type: str, 
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List, 
        actions: List, 
        thoughts: List,
        _system_message: str,
        instruction: str,
        obs: Dict,
    ) -> Tuple[str, List, List, List, List, List]:

        masks, thoughts_list, actions_list, observations_list = self.generate_observation(
            platform,
            observation_type,
            max_trajectory_length,
            a11y_tree_max_tokens,
            observations,
            actions,
            thoughts,
            _system_message,
            instruction,
            obs
        )
        
        if not self.testing:
            try:
                response = self.generate_thought({
                    "model": model,
                    "messages": self.messages,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature},
                    model,
                    observation_type
                )
            except Exception as e:
                logger.error("Failed to call" + model + ", Error: " + str(e))
                response = ""
        else:
            response = """
            ```json
            {
            "action_type": "CLICK",
            "x": 300,
            "y": 200
            }
            ```
            """

        logger.info("RESPONSE: %s", response)

        try:
            actions, actions_list = self.generate_action(action_space, observation_type, actions_list, response, masks)
            thoughts_list.append(response)
        except ValueError as e:
            print("Failed to parse action from response", e)
            actions, actions_list = [], []
            thoughts_list.append("")

        return response, actions, actions_list, thoughts_list, observations_list, self.messages

    
    def generate_observation(
        self, 
        _platform: str,
        observation_type: str, 
        max_trajectory_length: int,
        a11y_tree_max_tokens: int,
        observations: List, 
        actions: List, 
        thoughts: List,
        _system_message: str,
        instruction: str,
        obs: Dict,
    ) -> Tuple[List, List, List, List]:

        system_message = _system_message + "\nYou are asked to complete the following task: {}".format(instruction)

        # Prepare the payload for the API call
        masks: List= []

        self.messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        })
        
        assert len(observations) == len(actions) and len(actions) == len(thoughts) \
            , "The number of observations and actions should be the same."

        if len(observations) > max_trajectory_length:
            if max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = observations[-max_trajectory_length:]
                _actions = actions[-max_trajectory_length:]
                _thoughts = thoughts[-max_trajectory_length:]
        else:
            _observations = observations
            _actions = actions
            _thoughts = thoughts

        for previous_obs, previous_action, previous_thought in zip(_observations, _actions, _thoughts):

            if observation_type == "screenshot_a11y_tree":
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                self.messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif observation_type in ["som"]:
                _screenshot = previous_obs["screenshot"]

                self.messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the tagged screenshot as below. What's the next step that you will do to help with the task?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif observation_type == "screenshot":
                _screenshot = previous_obs["screenshot"]

                self.messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif observation_type == "a11y_tree":
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                self.messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                        }
                    ]
                })
            else:
                raise ValueError("Invalid observation_type type: " + observation_type)

            self.messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": previous_thought.strip() if len(previous_thought) > 0 else "No valid action"
                    },
                ]
            })

        if observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = encode_image(obs["screenshot"])
            linearized_accessibility_tree = linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                         platform=_platform) if observation_type == "screenshot_a11y_tree" else None
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        a11y_tree_max_tokens)

            if observation_type == "screenshot_a11y_tree":
                observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": linearized_accessibility_tree
                })
            else:
                observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": None
                })

            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                        if observation_type == "screenshot"
                        else "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        elif observation_type == "a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                         platform=_platform)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        a11y_tree_max_tokens)

            observations.append({
                "screenshot": None,
                "accessibility_tree": linearized_accessibility_tree
            })

            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    }
                ]
            })
        elif observation_type == "som":
            # Add som to the screenshot
            masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = tag_screenshot(obs["screenshot"], obs[
                "accessibility_tree"], _platform)
            base64_image = encode_image(tagged_screenshot)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        a11y_tree_max_tokens)

            observations.append({
                "screenshot": base64_image,
                "accessibility_tree": linearized_accessibility_tree
            })

            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the tagged screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        else:
            raise ValueError("Invalid observation_type type: " + observation_type)

        return masks, thoughts, actions, observations


    def generate_thought(
        self,
        payload: Dict,
        model: str,
        observation_type: str, 
    ) -> str:

        load_dotenv()
        if model.startswith("gpt"):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            }
            logger.info("Generating content with GPT model: %s", model)
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                if response.json()['error']['code'] == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()['choices'][0]['message']['content']
        elif model.startswith("claude"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            claude_messages = []

            for i, message in enumerate(messages):
                claude_message = {
                    "role": message["role"],
                    "content": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                for part in message["content"]:

                    if part['type'] == "image_url":
                        image_source = {}
                        image_source["type"] = "base64"
                        image_source["media_type"] = "image/png"
                        image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                        claude_message['content'].append({"type": "image", "source": image_source})

                    if part['type'] == "text":
                        claude_message['content'].append({"type": "text", "text": part['text']})

                claude_messages.append(claude_message)

            # the claude not support system message in our endpoint, so we concatenate it at the first user message
            if claude_messages[0]['role'] == "system":
                claude_system_message_item = claude_messages[0]['content'][0]
                claude_messages[1]['content'].insert(0, claude_system_message_item)
                claude_messages.pop(0)

            logger.debug("CLAUDE MESSAGE: %s", repr(claude_messages))

            headers = {
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": claude_messages,
                "temperature": temperature,
                "top_p": top_p
            }

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()['content'][0]['text']
        elif model.startswith("mistral"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            assert observation_type in pure_text_settings, f"The model {model} can only support text-based input, please consider change based model or settings"

            mistral_messages = []

            for i, message in enumerate(messages):
                mistral_message = {
                    "role": message["role"],
                    "content": ""
                }

                for part in message["content"]:
                    mistral_message['content'] = part['text'] if part['type'] == "text" else ""

                mistral_messages.append(mistral_message)

            from openai import OpenAI

            client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"],
                            base_url='https://api.together.xyz',
                            )

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", model)
                    response = client.chat.completions.create( # type: ignore
                        messages=mistral_messages, # type: ignore
                        model=model,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )
                    break
                except:
                    if flag == 0:
                        mistral_messages = [mistral_messages[0]] + mistral_messages[-1:]
                    else:
                        mistral_messages[-1]["content"] = ' '.join(mistral_messages[-1]["content"].split()[:-500])
                    flag = flag + 1

            try:
                return response.choices[0].message.content # type: ignore
            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""
        elif model.startswith("THUDM"):
            # THUDM/cogagent-chat-hf
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            cog_messages = []

            for i, message in enumerate(messages):
                cog_message = {
                    "role": message["role"],
                    "content": []
                }

                for part in message["content"]:
                    if part['type'] == "image_url":
                        cog_message['content'].append(
                            {"type": "image_url", "image_url": {"url": part['image_url']['url']}})

                    if part['type'] == "text":
                        cog_message['content'].append({"type": "text", "text": part['text']})

                cog_messages.append(cog_message)

            # the cogagent not support system message in our endpoint, so we concatenate it at the first user message
            if cog_messages[0]['role'] == "system":
                cog_system_message_item = cog_messages[0]['content'][0]
                cog_messages[1]['content'].insert(0, cog_system_message_item)
                cog_messages.pop(0)

            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": cog_messages,
                "temperature": temperature,
                "top_p": top_p
            }

            base_url = "http://127.0.0.1:8000"

            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=False)
            if response.status_code == 200:
                decoded_line = response.json()
                content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
                return content
            else:
                print("Failed to call LLM: ", response.status_code)
                return ""
        elif model in ["gemini-pro", "gemini-pro-vision"]:
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            if model == "gemini-pro":
                assert observation_type in pure_text_settings, f"The model {model} can only support text-based input, please consider change based model or settings"

            gemini_messages = []
            for i, message in enumerate(messages):
                role_mapping = {
                    "assistant": "model",
                    "user": "user",
                    "system": "system"
                }
                gemini_message = {
                    "role": role_mapping[message["role"]],
                    "parts": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"

                # The gemini only support the last image as single image input
                if i == len(messages) - 1:
                    for part in message["content"]:
                        gemini_message['parts'].append(part['text']) if part['type'] == "text" else gemini_message['parts'].append(encoded_img_to_pil_img(part['image_url']['url'])) # type: ignore
                else:
                    for part in message["content"]:
                        gemini_message['parts'].append(part['text']) if part['type'] == "text" else None # type: ignore

                gemini_messages.append(gemini_message) # type: ignore

            # the gemini not support system message in our endpoint, so we concatenate it at the first user message
            if gemini_messages[0]['role'] == "system":
                gemini_messages[1]['parts'][0] = gemini_messages[0]['parts'][0] + "\n" + gemini_messages[1]['parts'][0] # type: ignore
                gemini_messages.pop(0)

            # since the gemini-pro-vision donnot support multi-turn message
            if model == "gemini-pro-vision":
                message_history_str = ""
                for message in gemini_messages:
                    message_history_str += "<|" + message['role'] + "|>\n" + message['parts'][0] + "\n"
                gemini_messages = [{"role": "user", "parts": [message_history_str, gemini_messages[-1]['parts'][1]]}]
                # gemini_messages[-1]['parts'][1].save("output.png", "PNG")

            # print(gemini_messages)
            api_key = os.environ.get("GENAI_API_KEY")
            assert api_key is not None, "Please set the GENAI_API_KEY environment variable"
            genai.configure(api_key=api_key)
            logger.info("Generating content with Gemini model: %s", model)
            request_options = {"timeout": 120}
            gemini_model = genai.GenerativeModel(model)

            response = gemini_model.generate_content(
                gemini_messages,
                generation_config={
                    "candidate_count": 1,
                    # "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature
                },
                safety_settings={
                    "harassment": "block_none",
                    "hate": "block_none",
                    "sex": "block_none",
                    "danger": "block_none"
                },
                request_options=request_options
            )
            return response.text
        elif model == "gemini-1.5-pro-latest":
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            gemini_messages = []
            for i, message in enumerate(messages):
                role_mapping = {
                    "assistant": "model",
                    "user": "user",
                    "system": "system"
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                gemini_message = {
                    "role": role_mapping[message["role"]],
                    "parts": []
                }

                # The gemini only support the last image as single image input
                for part in message["content"]:

                    if part['type'] == "image_url":
                        # Put the image at the beginning of the message
                        gemini_message['parts'].insert(0, encoded_img_to_pil_img(part['image_url']['url'])) # type: ignore
                    elif part['type'] == "text":
                        gemini_message['parts'].append(part['text']) # type: ignore
                    else:
                        raise ValueError("Invalid content type: " + part['type'])

                gemini_messages.append(gemini_message)

            # the system message of gemini-1.5-pro-latest need to be inputted through model initialization parameter
            system_instruction = None
            if gemini_messages[0]['role'] == "system":
                system_instruction = gemini_messages[0]['parts'][0]
                gemini_messages.pop(0)

            api_key = os.environ.get("GENAI_API_KEY")
            assert api_key is not None, "Please set the GENAI_API_KEY environment variable"
            genai.configure(api_key=api_key)
            logger.info("Generating content with Gemini model: %s", model)
            request_options = {"timeout": 120}
            gemini_model = genai.GenerativeModel(
                model,
                system_instruction=system_instruction
            )

            with open("response.json", "w") as f:
                messages_to_save = []
                for message in gemini_messages:
                    messages_to_save.append({
                        "role": message["role"],
                        "content": [part if isinstance(part, str) else "image" for part in message["parts"]]
                    })
                json.dump(messages_to_save, f, indent=4)

            response = gemini_model.generate_content(
                gemini_messages,
                generation_config={
                    "candidate_count": 1,
                    # "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature
                },
                safety_settings={
                    "harassment": "block_none",
                    "hate": "block_none",
                    "sex": "block_none",
                    "danger": "block_none"
                },
                request_options=request_options
            )

            return response.text
        elif model == "llama3-70b":
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            assert observation_type in pure_text_settings, f"The model {model} can only support text-based input, please consider change based model or settings"

            groq_messages = []

            for i, message in enumerate(messages):
                groq_message = {
                    "role": message["role"],
                    "content": ""
                }

                for part in message["content"]:
                    groq_message['content'] = part['text'] if part['type'] == "text" else ""

                groq_messages.append(groq_message)

            # The implementation based on Groq API
            client = Groq( # type: ignore
                api_key=os.environ.get("GROQ_API_KEY")
            )

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", model)
                    response = client.chat.completions.create( # type: ignore
                        messages=groq_messages, # type: ignore
                        model="llama3-70b-8192",
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )
                    break
                except:
                    if flag == 0:
                        groq_messages = [groq_messages[0]] + groq_messages[-1:]
                    else:
                        groq_messages[-1]["content"] = ' '.join(groq_messages[-1]["content"].split()[:-500])
                    flag = flag + 1

            try:
                return response.choices[0].message.content # type: ignore
            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""
        elif model.startswith("qwen"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            qwen_messages = []

            for i, message in enumerate(messages):
                qwen_message = {
                    "role": message["role"],
                    "content": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                for part in message["content"]:
                    qwen_message['content'].append(
                        {"image": "file://" + save_to_tmp_img_file(part['image_url']['url'])}) if part[
                                                                                                      'type'] == "image_url" else None
                    qwen_message['content'].append({"text": part['text']}) if part['type'] == "text" else None

                qwen_messages.append(qwen_message)

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", model)

                    if model in ["qwen-vl-plus", "qwen-vl-max"]:
                        response = dashscope.MultiModalConversation.call(
                            model=model,
                            messages=qwen_messages,
                            result_format="message",
                            max_length=max_tokens,
                            top_p=top_p,
                            temperature=temperature
                        )

                    elif model in ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-0428", "qwen-max-0403",
                                        "qwen-max-0107", "qwen-max-longcontext"]:
                        response = dashscope.Generation.call(
                            model=model,
                            messages=qwen_messages,
                            result_format="message",
                            max_length=max_tokens,
                            top_p=top_p,
                            temperature=temperature
                        )

                    else:
                        raise ValueError("Invalid model: " + model)

                    if response.status_code == HTTPStatus.OK:
                        break
                    else:
                        logger.error('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                            response.request_id, response.status_code, # type: ignore
                            response.code, response.message # type: ignore
                        )) # type: ignore
                        raise Exception("Failed to call LLM: " + response.message) # type: ignore
                except:
                    if flag == 0:
                        qwen_messages = [qwen_messages[0]] + qwen_messages[-1:]
                    else:
                        for i in range(len(qwen_messages[-1]["content"])):
                            if "text" in qwen_messages[-1]["content"][i]:
                                qwen_messages[-1]["content"][i]["text"] = ' '.join(
                                    qwen_messages[-1]["content"][i]["text"].split()[:-500])
                    flag = flag + 1

            try:
                if model in ["qwen-vl-plus", "qwen-vl-max"]:
                    return response['output']['choices'][0]['message']['content'][0]['text'] # type: ignore
                else:
                    return response['output']['choices'][0]['message']['content'] # type: ignore

            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""
        else:
            raise ValueError("Invalid model: " + model)

        return ""

    
    def generate_action(
        self,
        action_space: str,
        observation_type: str,
        actions_list: List,
        response: str,
        masks: List,
        ) -> Tuple[List, List]:
        
        if observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + action_space)

            actions_list.append(actions)

            return actions, actions_list
        
        elif observation_type in ["som"]:
            # parse from the response
            if action_space == "computer_13":
                raise ValueError("Invalid action space: " + action_space)
            elif action_space == "pyautogui":
                actions = parse_code_from_som_string(response, masks)
            else:
                raise ValueError("Invalid action space: " + action_space)

            actions_list.append(actions)

            return actions, actions_list

        return actions, actions_list

    
    def reset(
        self,
        actions: List,
        thought: List,
        observations: List,
    ) -> Tuple[List, List, List]:
        thought.clear()
        actions.clear()
        observations.clear()
        self.messages.clear()

        return thought, actions, observations