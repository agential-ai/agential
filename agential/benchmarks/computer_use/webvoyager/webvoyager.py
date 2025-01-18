"""WebVoyager benchmark."""

import os
import time
from typing import Any, Dict
from agential.benchmarks.computer_use.base import BaseComputerUseBenchmark
from selenium import webdriver
from selenium.webdriver.common.by import By
import base64
import shutil

from agential.benchmarks.computer_use.webvoyager.utils import get_web_element_rect, get_webarena_accessibility_tree, extract_information

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
        "prefs", {
            "download.default_directory": download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )
    return options

class WebVoyager(BaseComputerUseBenchmark):
    def __init__(
        self,
        download_dir: str,
        headless: bool,
        force_device_scale: bool,
        max_iter: int = 5,
        text_only: bool = False,
        fix_box_color: bool = False,
        window_width: int = 1024,
        window_height: int = 768,
    ) -> None:
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
        self.fail_obs = ""  # When error execute the action
        self.pdf_obs = ""  # When download PDF file
        self.warn_obs = ""  # Type warning
        self.pattern = r'Thought:|Action:|Observation:'

        self.task = None
        self.driver_task = None
        self.download_files = []
        self.it = 0

    def close(self) -> None:
        pass

    def reset(self, task: Dict[str, Any]) -> Any:
        self.task = task
        self.driver_task = webdriver.Chrome(options=self.options)
        self.driver_task.set_window_size(self.window_width, self.window_height)  # larger height may contain more web information
        self.driver_task.get(task['web'])

        try:
            self.driver_task.find_element(By.TAG_NAME, 'body').click()
        except:
            pass

        self.driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        # TODO: questionable
        for filename in os.listdir(self.download_dir):
            file_path = os.path.join(self.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        self.download_files = []
        self.it = 0


    def step(self, action_key: str) -> Any:
        # TODO: save/return acc tree, screenshot, fail_obs, pdf_obs, warn_obs, web_eles_text

        if self.it < self.max_iter:
            return  # TODO: handle this


        if not self.fail_obs:
            try:
                if not self.text_only:
                    rects, web_eles, web_eles_text = get_web_element_rect(self.driver_task, fix_color=self.fix_box_color)
                else:
                    ac_tree, obs_info = get_webarena_accessibility_tree(self.driver_task)

            except Exception:
                if not self.text_only:
                    raise RuntimeError('Driver error when adding set-of-mark.')
                else:
                    raise RuntimeError('Driver error when obtaining accessibility tree.')

            screenshot_data = self.driver_task.get_screenshot_as_png()
            encoded_image = base64.b64encode(screenshot_data).decode("utf-8")
        
        if (not self.text_only) and rects:
            for rect_ele in rects:
                self.driver_task.execute_script("arguments[0].remove()", rect_ele)
            rects = []

        # Execute action.
        self.fail_obs = ""
        self.pdf_obs = ""
        self.warn_obs = ""
        try:
            window_handle_task = self.driver_task.current_window_handle
            self.driver_task.switch_to.window(window_handle_task)

            if action_key == 'click':
                if not self.text_only:
                    click_ele_number = int(info[0])
                    web_ele = web_eles[click_ele_number]
                else:
                    click_ele_number = info[0]
                    element_box = obs_info[click_ele_number]['union_bound']
                    element_box_center = (element_box[0] + element_box[2] // 2,
                                            element_box[1] + element_box[3] // 2)
                    web_ele = self.driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                ele_tag_name = web_ele.tag_name.lower()
                ele_type = web_ele.get_attribute("type")

                exec_action_click(info, web_ele, self.driver_task)

                # deal with PDF file
                current_files = sorted(os.listdir(args.download_dir))
                if current_files != download_files:
                    # wait for download finish
                    time.sleep(10)
                    current_files = sorted(os.listdir(args.download_dir))

                    current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in download_files and pdf_file.endswith('.pdf')]
                    if current_download_file:
                        pdf_file = current_download_file[0]
                        pdf_obs = get_pdf_retrieval_ans_from_assistant(client, os.path.join(args.download_dir, pdf_file), task['ques'])
                        shutil.copy(os.path.join(args.download_dir, pdf_file), task_dir)
                        pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                    download_files = current_files

                if ele_tag_name == 'button' and ele_type == 'submit':
                    time.sleep(10)

            elif action_key == 'wait':
                time.sleep(5)

            elif action_key == 'type':
                if not args.text_only:
                    type_ele_number = int(info['number'])
                    web_ele = web_eles[type_ele_number]
                else:
                    type_ele_number = info['number']
                    element_box = obs_info[type_ele_number]['union_bound']
                    element_box_center = (element_box[0] + element_box[2] // 2,
                                            element_box[1] + element_box[3] // 2)
                    web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                warn_obs = exec_action_type(info, web_ele, driver_task)
                if 'wolfram' in task['web']:
                    time.sleep(5)

            elif action_key == 'scroll':
                if not args.text_only:
                    exec_action_scroll(info, web_eles, driver_task, args, None)
                else:
                    exec_action_scroll(info, None, driver_task, args, obs_info)

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

        self.it += 1

