"""Base (WebVoyager) Agent strategy class."""

import os
import re
import logging
import time
import json
import shutil
import dotenv

from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from typing import Dict, Any, Tuple, Optional

from agential.agents.computer_use.webvoyager_baseline.output import WebVoyagerBaseOutput
from agential.agents.computer_use.webvoyager_baseline.strategies.base import WebVoyagerBaseStrategy
from agential.core.llm import BaseLLM, Response

from agential.agents.computer_use.webvoyager_baseline.functional import (
    clip_message_and_obs,
    clip_message_and_obs_text_only,
    encode_image,
    extract_information,
    get_pdf_retrieval_ans_from_assistant,
    get_webarena_accessibility_tree,
    get_web_element_rect,
    print_message
)

dotenv.load_dotenv()

class WebVoyagerGeneralStrategy(WebVoyagerBaseStrategy):
    """A strategy class for the Web Voyager Agent.

    This class defines methods for generating actions, thoughts, and observations
    in an agent-based environment, specifically tailored to the Web Voyager Agent.
    """
    def __init__(
        self, 
        llm: BaseLLM, 
        testing: bool = False
    ) -> None:
        """Initializes the WebVoyagerBaseStrategy with the provided language model and testing flag.

        Args:
            llm (BaseLLM): The language model used for generating answers and critiques.
            testing (bool): Whether the generation is for testing purposes. Defaults to False.
        """
        super().__init__(llm=llm, testing=testing)

    def setup_logger(
        folder_path: str
    ) -> None:
        """Sets up a logger to record logs in a file named 'agent.log' inside the specified folder.

        Args:
            folder_path (str): The directory path where the log file will be saved.

        Returns:
            None

        This function creates a new log file or overwrites an existing one and sets the logging level to INFO.
        """
        log_file_path = os.path.join(folder_path, 'agent.log')

        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def driver_config(
        args: Dict[str, Any]
    ) -> Tuple[webdriver.ChromeOptions, bool]:
        """Configures options for the Chrome WebDriver based on the provided arguments.

        Args:
            args (Namespace): Command-line arguments that include configuration settings for the WebDriver.

        Returns:
            webdriver.ChromeOptions: A configured ChromeOptions instance for use with Selenium WebDriver.

        This function configures browser options for headless mode, device scaling, and custom preferences for downloads.
        """
        options = webdriver.ChromeOptions()

        if args["save_accessibility_tree"]:
            args["force_device_scale"] = True

        if args["force_device_scale"]:
            options.add_argument("--force-device-scale-factor=1")
        if args["headless"]:
            options.add_argument("--headless")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
        options.add_experimental_option(
            "prefs", {
                "download.default_directory": args["download_dir"],
                "plugins.always_open_pdf_externally": True
            }
        )
        return options, args["force_device_scale"]

    def format_msg(
        it: int, 
        init_msg: str,
        pdf_obs: str,
        warn_obs: str,
        web_img_b64: str,
        web_text: str
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
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': init_msg},
                ]
            }
            init_msg_format['content'].append({"type": "image_url",
                                            "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
            return init_msg_format
        else:
            if not pdf_obs:
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            'type': 'image_url',
                            'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                        }
                    ]
                }
            else:
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            'type': 'image_url',
                            'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                        }
                    ]
                }
            return curr_msg

    def format_msg_text_only(it: int, 
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
            init_msg_format = {
                'role': 'user',
                'content': init_msg + '\n' + ac_tree
            }
            return init_msg_format
        else:
            if not pdf_obs:
                curr_msg = {
                    'role': 'user',
                    'content': f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
                }
            else:
                curr_msg = {
                    'role': 'user',
                    'content': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
                }
            return curr_msg

    def generate_thought(
        self,
        messages: list[Any],
        seed: Optional[int],
        max_tokens: int = 1000,
        timeout: int = 30
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
        response = self.llm(
            messages,
            max_tokens,
            seed,
            timeout
        )

        return response

    def generate( ############# Fix Documentation and return items #################
        self,
        system_prompt: str,
        system_prompt_text_only: str,
        output_dir: str,
        download_dir: str,
        test_file: str = 'data/test.json',
        max_iter: int = 5,
        seed: int = None,
        max_attached_imgs: int = 1,
        temperature: float = 1.0,
        text_only: bool = False,
        headless: bool = False,
        save_accessibility_tree: bool = False,
        force_device_scale: bool = False,
        window_width: int = 1024,
        window_height: int = 768,
        fix_box_color: bool = False,
    ) -> WebVoyagerBaseOutput:

        api_key = os.getenv("OPENAI_ORGANIZATION")
        client = OpenAI(api_key=api_key)

        options, force_device_scale = self.driver_config(
                args = {
                    "save_accessibility_tree": save_accessibility_tree,
                    "force_device_scale": force_device_scale,
                    "headless": headless,
                    "download_dir": download_dir
                }
            )

        # Save Result file
        current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        result_dir = os.path.join(output_dir, current_time)
        os.makedirs(result_dir, exist_ok=True)

        # Load tasks
        tasks = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                tasks.append(json.loads(line))


        for task_id in range(len(tasks)):
            task = tasks[task_id]
            task_dir = os.path.join(result_dir, 'task{}'.format(task["id"]))
            os.makedirs(task_dir, exist_ok=True)
            self.setup_logger(task_dir)
            logging.info(f'########## TASK{task["id"]} ##########')

            driver_task = webdriver.Chrome(options=options)

            # About window size, 765 tokens
            # You can resize to height = 512 by yourself (255 tokens, Maybe bad performance)
            driver_task.set_window_size(window_width, window_height)  # larger height may contain more web information
            driver_task.get(task['web'])
            try:
                driver_task.find_element(By.TAG_NAME, 'body').click()
            except:
                pass
            # sometimes enter SPACE, the page will sroll down
            driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
            time.sleep(5)

            # We only deal with PDF file
            for filename in os.listdir(download_dir):
                file_path = os.path.join(download_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            download_files = []  # sorted(os.listdir(download_dir))

            fail_obs = ""  # When error execute the action
            pdf_obs = ""  # When download PDF file
            warn_obs = ""  # Type warning
            pattern = r'Thought:|Action:|Observation:'

            messages = [{'role': 'system', 'content': system_prompt}]
            obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
            if text_only:
                messages = [{'role': 'system', 'content': system_prompt_text_only}]
                obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

            init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
            init_msg = init_msg.replace('https://www.example.com', task['web'])
            init_msg = init_msg + obs_prompt

            it = 0
            accumulate_prompt_token = 0
            accumulate_completion_token = 0

            while it < max_iter:
                logging.info(f'Iter: {it}')
                it += 1
                if not fail_obs:
                    try:
                        if not text_only:
                            rects, web_eles, web_eles_text = get_web_element_rect(driver_task, fix_color=fix_box_color)
                        else:
                            accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                            ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                    except Exception as e:
                        if not text_only:
                            logging.error('Driver error when adding set-of-mark.')
                        else:
                            logging.error('Driver error when obtaining accessibility tree.')
                        logging.error(e)
                        break

                    img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
                    driver_task.save_screenshot(img_path)

                    # accessibility tree
                    if (not text_only) and save_accessibility_tree:
                        accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                        get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                    # encode image
                    b64_img = encode_image(img_path)

                    # format msg
                    if not text_only:
                        curr_msg = self.format_msg(it, init_msg, pdf_obs, warn_obs, b64_img, web_eles_text)
                    else:
                        curr_msg = self.format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree)
                    messages.append(curr_msg)
                else:
                    curr_msg = {
                        'role': 'user',
                        'content': fail_obs
                    }
                    messages.append(curr_msg)

                # Clip messages, too many attached images may cause confusion
                if not text_only:
                    messages = clip_message_and_obs(messages, max_attached_imgs)
                else:
                    messages = clip_message_and_obs_text_only(messages, max_attached_imgs)

                response = self.generate_thought(messages=messages, seed=seed)
                prompt_tokens = response.prompt_tokens
                completion_tokens = response.completion_tokens
                gpt_4v_res = response.output_text

                accumulate_prompt_token += prompt_tokens
                accumulate_completion_token += completion_tokens
                logging.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
                logging.info('API call complete...')
                messages.append({'role': 'assistant', 'content': gpt_4v_res})


                # remove the rects on the website
                if (not text_only) and rects:
                    logging.info(f"Num of interactive elements: {len(rects)}")
                    for rect_ele in rects:
                        driver_task.execute_script("arguments[0].remove()", rect_ele)
                    rects = []
                    # driver_task.save_screenshot(os.path.join(task_dir, 'screenshot{}_no_box.png'.format(it)))

                # extract action info
                try:
                    assert 'Thought:' in gpt_4v_res and 'Action:' in gpt_4v_res
                except AssertionError as e:
                    logging.error(e)
                    fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                    continue

                # bot_thought = re.split(pattern, gpt_4v_res)[1].strip()
                chosen_action = re.split(pattern, gpt_4v_res)[2].strip()
                # print(chosen_action)
                action_key, info = extract_information(chosen_action)

                fail_obs = ""
                pdf_obs = ""
                warn_obs = ""
                # execute action
                try:
                    window_handle_task = driver_task.current_window_handle
                    driver_task.switch_to.window(window_handle_task)

                    if action_key == 'click':
                        if not text_only:
                            click_ele_number = int(info[0])
                            web_ele = web_eles[click_ele_number]
                        else:
                            click_ele_number = info[0]
                            element_box = obs_info[click_ele_number]['union_bound']
                            element_box_center = (element_box[0] + element_box[2] // 2,
                                                element_box[1] + element_box[3] // 2)
                            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                        ele_tag_name = web_ele.tag_name.lower()
                        ele_type = web_ele.get_attribute("type")

                        self.exec_action_click(info, web_ele, driver_task)

                        # deal with PDF file
                        current_files = sorted(os.listdir(download_dir))
                        if current_files != download_files:
                            # wait for download finish
                            time.sleep(10)
                            current_files = sorted(os.listdir(download_dir))

                            current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in download_files and pdf_file.endswith('.pdf')]
                            if current_download_file:
                                pdf_file = current_download_file[0]
                                pdf_obs = get_pdf_retrieval_ans_from_assistant(client, os.path.join(download_dir, pdf_file), task['ques'])
                                shutil.copy(os.path.join(download_dir, pdf_file), task_dir)
                                pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                            download_files = current_files

                        if ele_tag_name == 'button' and ele_type == 'submit':
                            time.sleep(10)

                    elif action_key == 'wait':
                        time.sleep(5)

                    elif action_key == 'type':
                        if not text_only:
                            type_ele_number = int(info['number'])
                            web_ele = web_eles[type_ele_number]
                        else:
                            type_ele_number = info['number']
                            element_box = obs_info[type_ele_number]['union_bound']
                            element_box_center = (element_box[0] + element_box[2] // 2,
                                                element_box[1] + element_box[3] // 2)
                            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                        warn_obs = self.exec_action_type(info, web_ele, driver_task)
                        if 'wolfram' in task['web']:
                            time.sleep(5)

                    elif action_key == 'scroll':
                        if not text_only:
                            self.exec_action_scroll(
                                info, 
                                web_eles, 
                                driver_task, 
                                window_height, 
                                text_only,
                                None
                            )
                        else:
                            self.exec_action_scroll(
                                info, 
                                None, 
                                driver_task, 
                                window_height, 
                                text_only, 
                                obs_info
                            )

                    elif action_key == 'goback':
                        driver_task.back()
                        time.sleep(2)

                    elif action_key == 'google':
                        driver_task.get('https://www.google.com/')
                        time.sleep(2)

                    elif action_key == 'answer':
                        logging.info(info['content'])
                        logging.info('finish!!')
                        break

                    else:
                        raise NotImplementedError
                    fail_obs = ""
                except Exception as e:
                    logging.error('driver error info:')
                    logging.error(e)
                    if 'element click intercepted' not in str(e):
                        fail_obs = "The action you have chosen cannot be exected. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                    else:
                        fail_obs = ""
                    time.sleep(2)

            print_message(messages, task_dir)
            driver_task.quit()
            logging.info(f'Total cost: {accumulate_prompt_token / 1000 * 0.01 + accumulate_completion_token / 1000 * 0.03}')

    def exec_action_click(info: Dict[str, Any], web_ele: WebElement, driver_task: webdriver) -> None:
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
        driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
        web_ele.click()
        time.sleep(3)

    def exec_action_type(info: Dict[str, Any], web_ele: WebElement, driver_task: webdriver) -> None:
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
        warn_obs = ""
        type_content = info['content']

        ele_tag_name = web_ele.tag_name.lower()
        ele_type = web_ele.get_attribute("type")
        # outer_html = web_ele.get_attribute("outerHTML")
        if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
            warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
        try:
            # Not always work to delete
            web_ele.clear()
            # Another way to delete
            if platform.system() == 'Darwin':
                web_ele.send_keys(Keys.COMMAND + "a")
            else:
                web_ele.send_keys(Keys.CONTROL + "a")
            web_ele.send_keys(" ")
            web_ele.send_keys(Keys.BACKSPACE)
        except:
            pass

        actions = ActionChains(driver_task)
        actions.click(web_ele).perform()
        actions.pause(1)

        try:
            driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
        except:
            pass

        actions.send_keys(type_content)
        actions.pause(2)

        actions.send_keys(Keys.ENTER)
        actions.perform()
        time.sleep(10)
        return warn_obs

    def exec_action_scroll(info: Dict[str, Any], web_eles: WebElement, driver_task: webdriver, window_height: int, text_only: bool, obs_info: Dict[str, Any]) -> None:
        """
        Executes a scroll action on the webpage, either scrolling the window or a specific element.

        Args:
            info (dict): Information related to the scroll action.
            web_eles (list): A list of web elements to scroll.
            driver_task (WebDriver): The Selenium WebDriver instance executing the action.
            args (Dict[str,Any]): Command-line arguments containing configuration settings.
            obs_info (dict): Observations related to the web elements.

        Returns:
            None

        This function performs a scroll action either on the whole window or on a specific element identified by the provided index.
        """
        scroll_ele_number = info['number']
        scroll_content = info['content']
        if scroll_ele_number == "WINDOW":
            if scroll_content == 'down':
                driver_task.execute_script(f"window.scrollBy(0, {window_height*2//3});")
            else:
                driver_task.execute_script(f"window.scrollBy(0, {-window_height*2//3});")
        else:
            if not text_only:
                scroll_ele_number = int(scroll_ele_number)
                web_ele = web_eles[scroll_ele_number]
            else:
                element_box = obs_info[scroll_ele_number]['union_bound']
                element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
                web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
            actions = ActionChains(driver_task)
            driver_task.execute_script("arguments[0].focus();", web_ele)
            if scroll_content == 'down':
                actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
            else:
                actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
        time.sleep(3)

    def reset(  ######## Fix documentation #############
        self, 
        *args: Any, 
        **kwargs: Any
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
        




