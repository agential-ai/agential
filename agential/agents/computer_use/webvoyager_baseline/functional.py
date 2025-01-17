"""Functional module for WebVoyager."""

import base64
import re
import os
import json
import time
import logging
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Any
from selenium import webdriver ########## Add selenium==4.15.2 to Poetry ###############
from agential.agents.computer_use.webvoyager_baseline.functional_webarena import (
    fetch_browser_info, 
    fetch_page_accessibility_tree,
    parse_accessibility_tree, 
    clean_accesibility_tree
)


def resize_image(image_path: str) -> Image.Image:
    """
    Resizes an image to ensure its smallest dimension is no less than 512 pixels,
    maintaining the aspect ratio. If resizing is unnecessary, the original image is returned.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image.Image: The resized image. (Currently saved to the same path, return is commented out)
    """

    image = Image.open(image_path)
    width, height = image.size

    if min(width, height) < 512:
        return image
    elif width < height:
        new_width = 512
        new_height = int(height * (new_width / width))
    else:
        new_height = 512
        new_width = int(width * (new_height / height))

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    resized_image.save(image_path)


def encode_image(image_path: str) -> str:
    """
    Encodes an image file into a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_web_element_rect(browser: webdriver, fix_color: bool =True) -> Tuple[List, List, str]:
    """
    Identifies and marks interactive web elements on a webpage by drawing rectangles around them.

    Args:
        browser: Web browser instance used to execute JavaScript for element identification.
        fix_color (bool): If True, uses a fixed color for marking; otherwise, uses random colors.

    Returns:
        tuple: Contains:
            - rects (List): Rectangle coordinates for marked elements.
            - elements (List): List of web elements marked.
            - formatted_text (str): Text representation of elements and their labels.
    """
    if fix_color:
        selected_function = "getFixedColor"
        # color_you_like = '#5210da'
    else:
        selected_function = "getRandomColor"

    js_script = """
        let labels = [];

        function markPage() {
            var bodyRect = document.body.getBoundingClientRect();

            var items = Array.prototype.slice.call(
                document.querySelectorAll('*')
            ).map(function(element) {
                var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
                var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
                
                var rects = [...element.getClientRects()].filter(bb => {
                var center_x = bb.left + bb.width / 2;
                var center_y = bb.top + bb.height / 2;
                var elAtCenter = document.elementFromPoint(center_x, center_y);

                return elAtCenter === element || element.contains(elAtCenter) 
                }).map(bb => {
                const rect = {
                    left: Math.max(0, bb.left),
                    top: Math.max(0, bb.top),
                    right: Math.min(vw, bb.right),
                    bottom: Math.min(vh, bb.bottom)
                };
                return {
                    ...rect,
                    width: rect.right - rect.left,
                    height: rect.bottom - rect.top
                }
                });

                var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);

                return {
                element: element,
                include: 
                    (element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT") ||
                    (element.tagName === "BUTTON" || element.tagName === "A" || (element.onclick != null) || window.getComputedStyle(element).cursor == "pointer") ||
                    (element.tagName === "IFRAME" || element.tagName === "VIDEO" || element.tagName === "LI" || element.tagName === "TD" || element.tagName === "OPTION")
                ,
                area,
                rects,
                text: element.textContent.trim().replace(/\s{2,}/g, ' ')
                };
            }).filter(item =>
                item.include && (item.area >= 20)
            );

            // Only keep inner clickable items
            // first delete button inner clickable items
            const buttons = Array.from(document.querySelectorAll('button, a, input[type="button"], div[role="button"]'));

            //items = items.filter(x => !buttons.some(y => y.contains(x.element) && !(x.element === y) ));
            items = items.filter(x => !buttons.some(y => items.some(z => z.element === y) && y.contains(x.element) && !(x.element === y) ));
            items = items.filter(x => 
                !(x.element.parentNode && 
                x.element.parentNode.tagName === 'SPAN' && 
                x.element.parentNode.children.length === 1 && 
                x.element.parentNode.getAttribute('role') &&
                items.some(y => y.element === x.element.parentNode)));

            items = items.filter(x => !items.some(y => x.element.contains(y.element) && !(x == y)))

            // Function to generate random colors
            function getRandomColor(index) {
                var letters = '0123456789ABCDEF';
                var color = '#';
                for (var i = 0; i < 6; i++) {
                color += letters[Math.floor(Math.random() * 16)];
                }
                return color;
            }

            function getFixedColor(index) {
                var color = '#000000'
                return color
            }
            //function getFixedColor(index){
            //    var colors = ['#FF0000', '#00FF00', '#0000FF', '#000000']; // Red, Green, Blue, Black
            //    return colors[index % 4];
            //}
            

            // Lets create a floating border on top of these elements that will always be visible
            items.forEach(function(item, index) {
                item.rects.forEach((bbox) => {
                newElement = document.createElement("div");
                var borderColor = COLOR_FUNCTION(index);
                newElement.style.outline = `2px dashed ${borderColor}`;
                newElement.style.position = "fixed";
                newElement.style.left = bbox.left + "px";
                newElement.style.top = bbox.top + "px";
                newElement.style.width = bbox.width + "px";
                newElement.style.height = bbox.height + "px";
                newElement.style.pointerEvents = "none";
                newElement.style.boxSizing = "border-box";
                newElement.style.zIndex = 2147483647;
                // newElement.style.background = `${borderColor}80`;
                
                // Add floating label at the corner
                var label = document.createElement("span");
                label.textContent = index;
                label.style.position = "absolute";
                //label.style.top = "-19px";
                label.style.top = Math.max(-19, -bbox.top) + "px";
                //label.style.left = "0px";
                label.style.left = Math.min(Math.floor(bbox.width / 5), 2) + "px";
                label.style.background = borderColor;
                label.style.color = "white";
                label.style.padding = "2px 4px";
                label.style.fontSize = "12px";
                label.style.borderRadius = "2px";
                newElement.appendChild(label);
                
                document.body.appendChild(newElement);
                labels.push(newElement);
                // item.element.setAttribute("-ai-label", label.textContent);
                });
            })

            // For the first way
            // return [labels, items.map(item => ({
            //     rect: item.rects[0] // assuming there's at least one rect
            // }))];

            // For the second way
            return [labels, items]
        }
        return markPage();""".replace("COLOR_FUNCTION", selected_function)
    rects, items_raw = browser.execute_script(js_script)

    # format_ele_text = [f"[{web_ele_id}]: \"{items_raw[web_ele_id]['text']}\";" for web_ele_id in range(len(items_raw)) if items_raw[web_ele_id]['text'] ]
    format_ele_text = []
    for web_ele_id in range(len(items_raw)):
        label_text = items_raw[web_ele_id]['text']
        ele_tag_name = items_raw[web_ele_id]['element'].tag_name
        ele_type = items_raw[web_ele_id]['element'].get_attribute("type")
        ele_aria_label = items_raw[web_ele_id]['element'].get_attribute("aria-label")
        input_attr_types = ['text', 'search', 'password', 'email', 'tel']

        if not label_text:
            if (ele_tag_name.lower() == 'input' and ele_type in input_attr_types) or ele_tag_name.lower() == 'textarea' or (ele_tag_name.lower() == 'button' and ele_type in ['submit', 'button']):
                if ele_aria_label:
                    format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{ele_aria_label}\";")
                else:
                    format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\";" )

        elif label_text and len(label_text) < 200:
            if not ("<img" in label_text and "src=" in label_text):
                if ele_tag_name in ["button", "input", "textarea"]:
                    if ele_aria_label and (ele_aria_label != label_text):
                        format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\", \"{ele_aria_label}\";")
                    else:
                        format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\";")
                else:
                    if ele_aria_label and (ele_aria_label != label_text):
                        format_ele_text.append(f"[{web_ele_id}]: \"{label_text}\", \"{ele_aria_label}\";")
                    else:
                        format_ele_text.append(f"[{web_ele_id}]: \"{label_text}\";")


    format_ele_text = '\t'.join(format_ele_text)
    return rects, [web_ele['element'] for web_ele in items_raw], format_ele_text


def extract_information(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extracts specific patterns of information from a given text based on predefined patterns.

    Args:
        text (str): Input text to parse.

    Returns:
        tuple: 
            - key (str): Type of action or pattern identified.
            - details (dict): Additional details (if any) about the pattern.
    """
    patterns = {
        "click": r"Click \[?(\d+)\]?",
        "type": r"Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?",
        "scroll": r"Scroll \[?(\d+|WINDOW)\]?[; ]+\[?(up|down)\]?",
        "wait": r"^Wait",
        "goback": r"^GoBack",
        "google": r"^Google",
        "answer": r"ANSWER[; ]+\[?(.[^\]]*)\]?"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            if key in ["click", "wait", "goback", "google"]:
                return key, match.groups()
            else:
                return key, {"number": match.group(1), "content": match.group(2)} if key in ["type", "scroll"] else {"content": match.group(1)}
    return None, None


def clip_message(msg: List[Dict[str, Any]], max_img_num: int) -> List[Dict[str, Any]]:
    """
    Clips messages to include only a specified number of user-uploaded images.

    Args:
        msg (List): List of message dictionaries containing 'role' and 'content'.
        max_img_num (int): Maximum number of images to retain.

    Returns:
        List: A list of clipped messages.
    """
    clipped_msg = []
    img_num = 0
    for idx in range(len(msg)):
        curr_msg = msg[len(msg) - 1 - idx]
        if curr_msg['role'] != 'user':
            clipped_msg = [curr_msg] + clipped_msg
        else:
            if type(curr_msg['content']) == str:
                clipped_msg = [curr_msg] + clipped_msg
            elif img_num < max_img_num:
                img_num += 1
                clipped_msg = [curr_msg] + clipped_msg
            else:
                curr_msg_clip = {
                    'role': curr_msg['role'],
                    'content': curr_msg['content'][0]["text"]
                }
                clipped_msg = [curr_msg_clip] + clipped_msg
    return clipped_msg


def clip_message_and_obs(msg: List[Dict[str, Any]], max_img_num: int) -> List[Dict[str, Any]]:
    """
    Clips messages and observations, replacing large content with placeholders when limits are exceeded.

    Args:
        msg (list): List of message dictionaries.
        max_img_num (int): Maximum number of images to retain.

    Returns:
        list: A list of clipped messages with placeholders for omitted content.
    """
    clipped_msg = []
    img_num = 0
    for idx in range(len(msg)):
        curr_msg = msg[len(msg) - 1 - idx]
        if curr_msg['role'] != 'user':
            clipped_msg = [curr_msg] + clipped_msg
        else:
            if type(curr_msg['content']) == str:
                clipped_msg = [curr_msg] + clipped_msg
            elif img_num < max_img_num:
                img_num += 1
                clipped_msg = [curr_msg] + clipped_msg
            else:
                msg_no_pdf = curr_msg['content'][0]["text"].split("Observation:")[0].strip() + "Observation: A screenshot and some texts. (Omitted in context.)"
                msg_pdf = curr_msg['content'][0]["text"].split("Observation:")[0].strip() + "Observation: A screenshot, a PDF file and some texts. (Omitted in context.)"
                curr_msg_clip = {
                    'role': curr_msg['role'],
                    'content': msg_no_pdf if "You downloaded a PDF file" not in curr_msg['content'][0]["text"] else msg_pdf
                }
                clipped_msg = [curr_msg_clip] + clipped_msg
    return clipped_msg


def clip_message_and_obs_text_only(msg: List[Dict[str, Any]], max_tree_num: int) -> List[Dict[str, Any]]:
    """
    Clips text-based messages and replaces large accessibility tree data with placeholders.

    Args:
        msg (list): List of message dictionaries.
        max_tree_num (int): Maximum number of trees to retain.

    Returns:
        list: A list of clipped messages with placeholders for omitted trees.
    """
    clipped_msg = []
    tree_num = 0
    for idx in range(len(msg)):
        curr_msg = msg[len(msg) - 1 - idx]
        if curr_msg['role'] != 'user':
            clipped_msg = [curr_msg] + clipped_msg
        else:
            if tree_num < max_tree_num:
                tree_num += 1
                clipped_msg = [curr_msg] + clipped_msg
            else:
                msg_no_pdf = curr_msg['content'].split("Observation:")[0].strip() + "Observation: An accessibility tree. (Omitted in context.)"
                msg_pdf = curr_msg['content'].split("Observation:")[0].strip() + "Observation: An accessibility tree and a PDF file. (Omitted in context.)"
                curr_msg_clip = {
                    'role': curr_msg['role'],
                    'content': msg_no_pdf if "You downloaded a PDF file" not in curr_msg['content'] else msg_pdf
                }
                clipped_msg = [curr_msg_clip] + clipped_msg
    return clipped_msg


def print_message(json_object: List[Dict[str, Any]], save_dir: str = None) -> None:
    """
    Logs and optionally saves user and system messages to a file.

    Args:
        json_object (list): List of message dictionaries.
        save_dir (str, optional): Directory to save the messages in JSON format.

    Returns:
        None
    """
    remove_b64code_obj = []
    for obj in json_object:
        if obj['role'] != 'user':
            # print(obj)
            logging.info(obj)
            remove_b64code_obj.append(obj)
        else:
            if type(obj['content']) == str:
                # print(obj)
                logging.info(obj)
                remove_b64code_obj.append(obj)
            else:
                print_obj = {
                    'role': obj['role'],
                    'content': obj['content']
                }
                for item in print_obj['content']:
                    if item['type'] == 'image_url':
                        item['image_url'] =  {"url": "data:image/png;base64,{b64_img}"}
                # print(print_obj)
                logging.info(print_obj)
                remove_b64code_obj.append(print_obj)
    if save_dir:
        with open(os.path.join(save_dir, 'interact_messages.json'), 'w', encoding='utf-8') as fw:
            json.dump(remove_b64code_obj, fw, indent=2)


def get_webarena_accessibility_tree(browser: webdriver, save_file: str = None) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts the accessibility tree of a webpage and cleans it for processing.

    Args:
        browser: Web browser instance used to fetch the accessibility tree.
        save_file (str, optional): If provided, saves the tree to the specified file.

    Returns:
        tuple:
            - content (str): Cleaned accessibility tree content.
            - obs_nodes_info (dict): Information about observed nodes.
    """
    browser_info = fetch_browser_info(browser)
    accessibility_tree = fetch_page_accessibility_tree(browser_info, browser, current_viewport_only=True)
    content, obs_nodes_info = parse_accessibility_tree(accessibility_tree)
    content = clean_accesibility_tree(content)
    if save_file:
        with open(save_file + '.json', 'w', encoding='utf-8') as fw:
            json.dump(obs_nodes_info, fw, indent=2)
        with open(save_file + '.txt', 'w', encoding='utf-8') as fw:
            fw.write(content)


    return content, obs_nodes_info


def compare_images(img1_path: str, img2_path: str) -> int:
    """
    Compares two images by calculating the total pixel-wise difference.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.

    Returns:
        int: Total difference between the images.
    """
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    img1_array = np.asarray(img1)
    img2_array = np.asarray(img2)

    difference = np.abs(img1_array - img2_array)

    total_difference = np.sum(difference)

    return total_difference


def get_pdf_retrieval_ans_from_assistant(client: Any, pdf_path: str, task: str) -> str:
    """
    Sends a PDF file to an assistant API and retrieves a task-related response.

    Args:
        client: API client for communication with the assistant.
        pdf_path (str): Path to the PDF file.
        task (str): Task description or query.

    Returns:
        str: Messages from the client api call.
    """
    # print("You download a PDF file that will be retrieved using the Assistant API.")
    logging.info("You download a PDF file that will be retrieved using the Assistant API.")
    file = client.files.create(
        file=open(pdf_path, "rb"),
        purpose='assistants'
    )
    # print("Create assistant...")
    logging.info("Create assistant...")
    assistant = client.beta.assistants.create(
        instructions="You are a helpful assistant that can analyze the content of a PDF file and give an answer that matches the given task, or retrieve relevant content that matches the task.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id]
    )
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=task,
        file_ids=[file.id]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    while True:
        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == 'completed':
            break
        time.sleep(2)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    messages_text = messages.data[0].content[0].text.value
    file_deletion_status = client.beta.assistants.files.delete(
        assistant_id=assistant.id,
        file_id=file.id
    )
    # print(file_deletion_status)
    logging.info(file_deletion_status)
    assistant_deletion_status = client.beta.assistants.delete(assistant.id)
    # print(assistant_deletion_status)
    logging.info(assistant_deletion_status)
    return messages_text