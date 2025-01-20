"""WebVoyager benchmark."""

import os
import time
from typing import Any, Dict, Tuple, Union
import platform
from agential.benchmarks.computer_use.base import BaseComputerUseBenchmark
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import base64
from openai import OpenAI

from agential.benchmarks.computer_use.webvoyager.utils import (
    get_pdf_retrieval_ans_from_assistant,
    get_web_element_rect,
    get_webarena_accessibility_tree,
)


def driver_config(
    download_dir: str,
    headless: bool,
    force_device_scale: bool,
):
    options = webdriver.ChromeOptions()

    if force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": download_dir,
            "plugins.always_open_pdf_externally": True,
        },
    )
    return options


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        # Not always work to delete.
        web_ele.clear()
        # Another way to delete.
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


def exec_action_scroll(info, web_eles, driver_task, window_height, text_only, obs_info):
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


def exec_action_click(web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)


class WebVoyager(BaseComputerUseBenchmark):
    def __init__(
        self,
        openai_client: OpenAI,
        download_dir: str,
        headless: bool,
        force_device_scale: bool,
        max_iter: int = 5,
        text_only: bool = False,
        fix_box_color: bool = False,
        window_width: int = 1024,
        window_height: int = 768,
    ) -> None:
        super().__init__()
        self.openai_client = openai_client
        self.download_dir = download_dir
        self.options = driver_config(
            download_dir=self.download_dir,
            headless=headless,
            force_device_scale=force_device_scale,
        )
        self.max_iter = max_iter
        self.text_only = text_only
        self.fix_box_color = fix_box_color
        self.window_width = window_width
        self.window_height = window_height
        self.download_dir = download_dir

        # For evaluation.
        self.pattern = r"Thought:|Action:|Observation:"

        self.finished = False
        self.task = None
        self.driver_task = None
        self.download_files = []
        self.it = 0
        self._prev_result = None

    def close(self) -> None:
        pass

    def reset(self, task: Dict[str, Any]) -> Any:
        self.task = task

        if self.driver_task:
            self.driver_task.quit()
        
        self.driver_task = webdriver.Chrome(options=self.options)
        self.driver_task.set_window_size(
            self.window_width, self.window_height
        )  # larger height may contain more web information
        self.driver_task.get(task["web"])

        try:
            self.driver_task.find_element(By.TAG_NAME, "body").click()
        except:
            pass

        self.driver_task.execute_script(
            """window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};"""
        )
        time.sleep(5)

        # TODO: questionable
        for filename in os.listdir(self.download_dir):
            file_path = os.path.join(self.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        self.finished = False
        self.download_files = []
        self.it = 0
        self._prev_result = None

    def step(self, action_key: str, params: Union[Tuple, Dict[str, str]]) -> Any:
        if not self.task and not self.driver_task:
            raise ValueError("Please reset the environment first.")
            
        # TODO: robust error handling of action_key and info

        if self.it >= self.max_iter or self.finished:
            return self._prev_result

        reward = 0
        rects = None
        web_eles = None
        web_eles_text = None
        ac_tree = None
        obs_info = None

        try:
            if not self.text_only:
                rects, web_eles, web_eles_text = get_web_element_rect(
                    self.driver_task, fix_color=self.fix_box_color
                )
            else:
                ac_tree, obs_info = get_webarena_accessibility_tree(
                    self.driver_task
                )
        except Exception:
            if not self.text_only:
                raise RuntimeError("Driver error when adding set-of-mark.")
            else:
                raise RuntimeError(
                    "Driver error when obtaining accessibility tree."
                )

        screenshot_data = self.driver_task.get_screenshot_as_png()
        encoded_image = base64.b64encode(screenshot_data).decode("utf-8")

        if (not self.text_only) and rects:
            for rect_ele in rects:
                self.driver_task.execute_script("arguments[0].remove()", rect_ele)

        # Execute action.
        fail_obs = ""
        pdf_obs = ""
        warn_obs = ""
        try:
            window_handle_task = self.driver_task.current_window_handle
            self.driver_task.switch_to.window(window_handle_task)

            if action_key == "click":
                if not self.text_only:
                    click_ele_number = int(params[0])
                    web_ele = web_eles[click_ele_number]
                else:
                    click_ele_number = params[0]
                    element_box = obs_info[click_ele_number]["union_bound"]
                    element_box_center = (
                        element_box[0] + element_box[2] // 2,
                        element_box[1] + element_box[3] // 2,
                    )
                    web_ele = self.driver_task.execute_script(
                        "return document.elementFromPoint(arguments[0], arguments[1]);",
                        element_box_center[0],
                        element_box_center[1],
                    )

                ele_tag_name = web_ele.tag_name.lower()
                ele_type = web_ele.get_attribute("type")

                exec_action_click(web_ele, self.driver_task)

                current_files = sorted(os.listdir(self.download_dir))
                if current_files != self.download_files:
                    # Wait for download finish.
                    time.sleep(10)
                    current_files = sorted(os.listdir(self.download_dir))

                    current_download_file = [
                        pdf_file
                        for pdf_file in current_files
                        if pdf_file not in self.download_files and pdf_file.endswith(".pdf")
                    ]

                    # New download files.
                    if current_download_file:
                        pdf_file = current_download_file[0]
                        pdf_obs = (
                            "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: "
                            + get_pdf_retrieval_ans_from_assistant(
                                self.openai_client,
                                os.path.join(self.download_dir, pdf_file),
                                self.task["ques"],
                            )
                        )
                    self.download_files = current_files

                if ele_tag_name == "button" and ele_type == "submit":
                    time.sleep(10)

            elif action_key == "wait":
                time.sleep(5)

            elif action_key == "type":
                if not self.text_only:
                    type_ele_number = int(params["number"])
                    web_ele = web_eles[type_ele_number]
                else:
                    type_ele_number = params["number"]
                    element_box = obs_info[type_ele_number]["union_bound"]
                    element_box_center = (
                        element_box[0] + element_box[2] // 2,
                        element_box[1] + element_box[3] // 2,
                    )
                    web_ele = self.driver_task.execute_script(
                        "return document.elementFromPoint(arguments[0], arguments[1]);",
                        element_box_center[0],
                        element_box_center[1],
                    )

                warn_obs = exec_action_type(params, web_ele, self.driver_task)
                if "wolfram" in self.task["web"]:
                    time.sleep(5)

            elif action_key == "scroll":
                if not self.text_only:
                    exec_action_scroll(params, web_eles, self.driver_task, self.window_height, self.text_only, None)
                else:
                    exec_action_scroll(params, None, self.driver_task, self.window_height, self.text_only, obs_info)
            elif action_key == "goback":
                self.driver_task.back()
                time.sleep(2)
            elif action_key == "google":
                self.driver_task.get("https://www.google.com/")
                time.sleep(2)
            elif action_key == "answer":
                self.finished = True
            else:
                raise NotImplementedError
            fail_obs = ""
        except Exception as e:
            if "element click intercepted" not in str(e):
                fail_obs = "The action you have chosen cannot be executed. Please double-check if you have selected the wrong Numerical Label, Action, or Action format. Then, provide the revised Thought and Action."
            else:
                fail_obs = ""
            time.sleep(2)

        self.it += 1

        obs = {
            "screenshot": encoded_image, 
            "fail": fail_obs, 
            "pdf": pdf_obs, 
            "warn": warn_obs
        }
        done = self.it >= self.max_iter or self.finished
        
        info = {}
        if not self.text_only:
            info["rects"] = rects
            info["web_elements"] = web_eles
            info["web_elements_text"] = web_eles_text
        else:
            info["accessibility_tree"] = ac_tree
            info["obs_info"] = obs_info

        result = (obs, reward, done, info)  # TODO: fix reward
        self._prev_result = result

        return result