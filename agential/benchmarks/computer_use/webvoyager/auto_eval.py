"""WebVoyager evaluation.

Reference: https://github.com/MinorJerry/WebVoyager/blob/main/evaluation/auto_eval.py"""

import os
import json
import time
import re

from agential.benchmarks.computer_use.webvoyager.utils import encode_image
from openai import OpenAI

SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."""
USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """


def auto_eval_with_llm(process_dir, openai_client: OpenAI, api_model, img_num):
    res_files = sorted(os.listdir(process_dir))
    with open(os.path.join(process_dir, "interact_messages.json")) as fr:
        it_messages = json.load(fr)

    if len(it_messages) == 1:
        return 0

    task_info = it_messages[1]["content"]
    if type(task_info) == list:
        task_info = task_info[0]["text"]
    assert "Now given a task" in task_info
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info)
    task_content = matches.group(1).strip()

    ans_info = it_messages[-1]["content"]
    if "Action: ANSWER" not in ans_info:
        return 0
    
    pattern_ans = r"ANSWER[; ]+\[?(.[^\]]*)\]?"
    matches_ans = re.search(pattern_ans, ans_info)
    answer_content = matches_ans.group(1).strip()

    whole_content_img = []
    pattern_png = r"screenshot(\d+)\.png"
    matches = [
        (filename, int(re.search(pattern_png, filename).group(1)))
        for filename in res_files
        if re.search(pattern_png, filename)
    ]
    matches.sort(key=lambda x: x[1])
    end_files = matches[-img_num:]
    for png_file in end_files:
        b64_img = encode_image(os.path.join(process_dir, png_file[0]))
        whole_content_img.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
            }
        )

    user_prompt_tmp = USER_PROMPT.replace("<task>", task_content)
    user_prompt_tmp = user_prompt_tmp.replace("<answer>", answer_content)
    user_prompt_tmp = user_prompt_tmp.replace("<num>", str(img_num))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt_tmp}]
            + whole_content_img
            + [{"type": "text", "text": "Your verdict:\n"}],
        },
    ]
    while True:
        try:
            openai_response = openai_client.chat.completions.create(
                model=api_model,
                messages=messages,
                max_tokens=1000,
                seed=42,
                temperature=0,
            )

            break
        except Exception as e:
            print(e)
            if type(e).__name__ == "RateLimitError":
                time.sleep(10)
            elif type(e).__name__ == "APIError":
                time.sleep(15)
            elif type(e).__name__ == "InvalidRequestError":
                exit(0)
            else:
                time.sleep(10)
    gpt_4v_res = openai_response.choices[0].message.content

    auto_eval_res = 0 if "NOT SUCCESS" in gpt_4v_res else 1
    if "SUCCESS" not in gpt_4v_res:
        auto_eval_res = None
    return auto_eval_res
