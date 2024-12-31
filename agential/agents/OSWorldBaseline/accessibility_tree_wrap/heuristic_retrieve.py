"""OSWorldBaseline Agent Accessbility Tree Wrap."""

import io

from typing import List, Tuple
from xml.etree.ElementTree import Element

from PIL import Image, ImageDraw, ImageFont

state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"


def judge_node(
    node: Element, platform: str = "ubuntu", check_image: bool = False
) -> bool:
    """Determines whether a given XML node qualifies for processing based on platform and various node properties.

    Args:
        node (Element): The XML node to be evaluated.
        platform (str): The platform to consider when evaluating the node ('ubuntu' or 'windows'). Default is 'ubuntu'.
        check_image (bool): Flag to indicate whether to check for image-related properties of the node. Default is False.

    Returns:
        bool: True if the node passes all checks, False otherwise.

    Raises:
        ValueError: If the platform is not 'ubuntu' or 'windows'.
    """
    if platform == "ubuntu":
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
    elif platform == "windows":
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    keeps: bool = (
        node.tag.startswith("document")
        or node.tag.endswith("item")
        or node.tag.endswith("button")
        or node.tag.endswith("heading")
        or node.tag.endswith("label")
        or node.tag.endswith("scrollbar")
        or node.tag.endswith("searchbox")
        or node.tag.endswith("textbox")
        or node.tag.endswith("link")
        or node.tag.endswith("tabelement")
        or node.tag.endswith("textfield")
        or node.tag.endswith("textarea")
        or node.tag.endswith("menu")
        or node.tag
        in {
            "alert",
            "canvas",
            "check-box",
            "combo-box",
            "entry",
            "icon",
            "image",
            "paragraph",
            "scroll-bar",
            "section",
            "slider",
            "static",
            "table-cell",
            "terminal",
            "text",
            "netuiribbontab",
            "start",
            "trayclockwclass",
            "traydummysearchcontrol",
            "uiimage",
            "uiproperty",
            "uiribboncommandbar",
        }
    )
    keeps = (
        keeps
        and (
            platform == "ubuntu"
            and node.get("{{{:}}}showing".format(_state_ns), "false") == "true"
            and node.get("{{{:}}}visible".format(_state_ns), "false") == "true"
            or platform == "windows"
            and node.get("{{{:}}}visible".format(_state_ns), "false") == "true"
        )
        and (
            node.get("{{{:}}}enabled".format(_state_ns), "false") == "true"
            or node.get("{{{:}}}editable".format(_state_ns), "false") == "true"
            or node.get("{{{:}}}expandable".format(_state_ns), "false") == "true"
            or node.get("{{{:}}}checkable".format(_state_ns), "false") == "true"
        )
        and (
            node.get("name", "") != ""
            or node.text is not None
            and len(node.text) > 0
            or check_image
            and node.get("image", "false") == "true"
        )
    )

    coordinates: Tuple[int, int] = eval(
        node.get("{{{:}}}screencoord".format(_component_ns), "(-1, -1)")
    )
    sizes: Tuple[int, int] = eval(
        node.get("{{{:}}}size".format(_component_ns), "(-1, -1)")
    )
    keeps = (
        keeps
        and coordinates[0] >= 0
        and coordinates[1] >= 0
        and sizes[0] > 0
        and sizes[1] > 0
    )
    return keeps


def filter_nodes(
    root: Element, platform: str = "ubuntu", check_image: bool = False
) -> List:
    """Filters the XML tree by evaluating each node with the `judge_node` function.

    Args:
        root (Element): The root of the XML tree to be filtered.
        platform (str): The platform to consider when filtering the nodes ('ubuntu' or 'windows'). Default is 'ubuntu'.
        check_image (bool): Flag to indicate whether to check for image-related properties of nodes. Default is False.

    Returns:
        List: A list of XML nodes that pass the filtering criteria.
    """
    filtered_nodes = []

    for node in root.iter():
        if judge_node(node, platform, check_image):
            filtered_nodes.append(node)
            # print(ET.tostring(node, encoding="unicode"))

    return filtered_nodes


def draw_bounding_boxes(
    nodes: List[Element],
    image_file_content: bytes,
    down_sampling_ratio: float = 1.0,
    platform: str = "ubuntu",
) -> Tuple[List, List, str, bytes]:
    """Draws bounding boxes around specified nodes on a provided image, and returns the updated image with bounding boxes.

    Args:
        nodes (List[Element]): A list of XML nodes representing the elements to be bounded.
        image_file_content (bytes): The content of the image in bytes (used to load the image).
        down_sampling_ratio (float): A ratio to scale down the image and coordinates (default is 1.0, meaning no scaling).
        platform (str): The platform to consider when evaluating the nodes ('ubuntu' or 'windows'). Default is 'ubuntu'.

    Returns:
        Tuple[List, List, str, bytes]:
            - A list of bounding box coordinates (marks).
            - A list of nodes that had bounding boxes drawn.
            - A string containing information about each node (index, tag, name, text).
            - The updated image content in bytes.

    Raises:
        ValueError: If the coordinates or sizes of the nodes are invalid.
        IOError: If the specified font cannot be loaded.
    """
    if platform == "ubuntu":
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    # Load the screenshot image
    image_stream = io.BytesIO(image_file_content)
    image = Image.open(image_stream)
    if float(down_sampling_ratio) != 1.0:
        image = image.resize(
            (
                int(image.size[0] * down_sampling_ratio),
                int(image.size[1] * down_sampling_ratio),
            )
        )
    draw = ImageDraw.Draw(image)
    marks = []
    drew_nodes = []
    text_informations: List[str] = ["index\ttag\tname\ttext"]

    try:
        # Adjust the path to the font file you have or use a default one
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # Fallback to a basic font if the specified font can't be loaded
        font = ImageFont.load_default()

    index = 1

    # Loop over all the visible nodes and draw their bounding boxes
    for _node in nodes:
        coords_str = _node.attrib.get("{{{:}}}screencoord".format(_component_ns))
        size_str = _node.attrib.get("{{{:}}}size".format(_component_ns))

        if coords_str and size_str:
            try:
                # Parse the coordinates and size from the strings
                coords = tuple(map(int, coords_str.strip("()").split(", ")))
                size = tuple(map(int, size_str.strip("()").split(", ")))

                import copy

                original_coords = copy.deepcopy(coords)
                original_size = copy.deepcopy(size)

                if float(down_sampling_ratio) != 1.0:
                    # Downsample the coordinates and size
                    coords = tuple(int(coord * down_sampling_ratio) for coord in coords)
                    size = tuple(int(s * down_sampling_ratio) for s in size)

                # Check for negative sizes
                if size[0] <= 0 or size[1] <= 0:
                    raise ValueError(f"Size must be positive, got: {size}")

                # Calculate the bottom-right corner of the bounding box
                bottom_right = (coords[0] + size[0], coords[1] + size[1])

                # Check that bottom_right > coords (x1 >= x0, y1 >= y0)
                if bottom_right[0] < coords[0] or bottom_right[1] < coords[1]:
                    raise ValueError(
                        f"Invalid coordinates or size, coords: {coords}, size: {size}"
                    )

                # Check if the area only contains one color
                cropped_image = image.crop((*coords, *bottom_right))  # type: ignore
                if len(set(list(cropped_image.getdata()))) == 1:
                    continue

                # Draw rectangle on image
                draw.rectangle([coords, bottom_right], outline="red", width=1)

                # Draw index number at the bottom left of the bounding box with black background
                text_position = (
                    coords[0],
                    bottom_right[1],
                )  # Adjust Y to be above the bottom right
                text_bbox: Tuple[int, int, int, int] = draw.textbbox(
                    text_position, str(index), font=font, anchor="lb"
                )
                # offset: int = bottom_right[1]-text_bbox[3]
                # text_bbox = (text_bbox[0], text_bbox[1]+offset, text_bbox[2], text_bbox[3]+offset)

                # draw.rectangle([text_position, (text_position[0] + 25, text_position[1] + 18)], fill='black')
                draw.rectangle(text_bbox, fill="black")
                draw.text(
                    text_position, str(index), font=font, anchor="lb", fill="white"
                )

                # each mark is an x, y, w, h tuple
                marks.append(
                    [
                        original_coords[0],
                        original_coords[1],
                        original_size[0],
                        original_size[1],
                    ]
                )
                drew_nodes.append(_node)

                if _node.text:
                    node_text = (
                        _node.text
                        if '"' not in _node.text
                        else '"{:}"'.format(_node.text.replace('"', '""'))
                    )
                elif _node.get("{{{:}}}class".format(class_ns_windows), "").endswith(
                    "EditWrapper"
                ) and _node.get("{{{:}}}value".format(_value_ns)):
                    node_text = _node.get("{{{:}}}value".format(_value_ns), "")
                    node_text = (
                        node_text
                        if '"' not in node_text
                        else '"{:}"'.format(node_text.replace('"', '""'))
                    )
                else:
                    node_text = '""'
                text_information: str = "{:d}\t{:}\t{:}\t{:}".format(
                    index, _node.tag, _node.get("name", ""), node_text
                )
                text_informations.append(text_information)

                index += 1

            except ValueError:
                pass

    output_image_stream = io.BytesIO()
    image.save(output_image_stream, format="PNG")
    image_content = output_image_stream.getvalue()

    return marks, drew_nodes, "\n".join(text_informations), image_content
