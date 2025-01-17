"""Functional module from WebArena for WebVoyager."""

import re

from typing import Any, Dict, TypedDict

from selenium import webdriver


class AccessibilityTreeNode(TypedDict):
    """A dictionary type representing a node in the accessibility tree.

    This class defines the structure of an individual node in the accessibility
    tree, which represents accessible elements in the DOM. It includes various
    properties related to the node's role, bounds, and relationships with other nodes.

    Attributes:
        nodeId (str): The unique identifier of the node.
        ignored (bool): Indicates whether the node is ignored in the accessibility tree.
        role (dict[str, Any]): The role of the element (e.g., "button", "link").
        chromeRole (dict[str, Any]): The role of the element as recognized by the browser.
        name (dict[str, Any]): The accessible name of the element.
        properties (list[dict[str, Any]]): Additional properties associated with the node.
        childIds (list[str]): The list of child node IDs.
        parentId (str): The ID of the parent node.
        backendDOMNodeId (str): The ID of the corresponding DOM node.
        frameId (str): The frame ID where the node is located.
        bound (list[float] | None): The bounds of the node (if available).
        union_bound (list[float] | None): The union bounds of the node (if available).
        offsetrect_bound (list[float] | None): The offset bounding rectangle of the node (if available).
    """

    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class BrowserConfig(TypedDict):
    """A dictionary type representing the configuration of the browser window.

    This class contains information about the browser window's dimensions and
    device pixel ratio, which are essential for calculating viewport-related data.

    Attributes:
        win_top_bound (float): The top boundary of the browser window.
        win_left_bound (float): The left boundary of the browser window.
        win_width (float): The width of the browser window.
        win_height (float): The height of the browser window.
        win_right_bound (float): The right boundary of the browser window.
        win_lower_bound (float): The bottom boundary of the browser window.
        device_pixel_ratio (float): The device pixel ratio (should be 1.0).
    """

    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    """A dictionary type representing information about the browser state.

    This class includes the browser's DOM tree and the browser configuration,
    such as window boundaries and device pixel ratio.

    Attributes:
        DOMTree (dict[str, Any]): The DOM tree structure of the page.
        config (BrowserConfig): The configuration of the browser window.
    """

    DOMTree: dict[str, Any]
    config: BrowserConfig


IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
)

AccessibilityTree = list[AccessibilityTreeNode]

IN_VIEWPORT_RATIO_THRESHOLD = 0.6


def fetch_browser_info(
    # page: Page,
    browser: webdriver,
) -> BrowserInfo:
    """Fetches detailed information about the browser state, including the DOM tree
    and window configuration.

    This function extracts the DOM tree via a Chrome DevTools Protocol (CDP) command
    and calibrates the bounds of the page. It also retrieves information about
    the browser's window dimensions and device pixel ratio.

    Args:
        browser: The browser instance used to execute CDP commands.

    Returns:
        BrowserInfo: The extracted browser information, including the DOM tree and window configuration.
    """
    # extract domtree
    tree = browser.execute_cdp_cmd(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": [],
            "includeDOMRects": True,
            "includePaintOrder": True,
        },
    )

    # calibrate the bounds, in some cases, the bounds are scaled somehow
    bounds = tree["documents"][0]["layout"]["bounds"]
    b = bounds[0]
    n = b[2] / browser.get_window_size()["width"]
    bounds = [[x / n for x in bound] for bound in bounds]
    tree["documents"][0]["layout"]["bounds"] = bounds

    # extract browser info
    # win_top_bound = page.evaluate("window.pageYOffset")
    # win_left_bound = page.evaluate("window.pageXOffset")
    # win_width = page.evaluate("window.screen.width")
    # win_height = page.evaluate("window.screen.height")

    win_top_bound = browser.execute_script("return window.pageYOffset;")
    win_left_bound = browser.execute_script("return window.pageXOffset;")
    win_width = browser.execute_script("return window.screen.width;")
    win_height = browser.execute_script("return window.screen.height;")
    win_right_bound = win_left_bound + win_width
    win_lower_bound = win_top_bound + win_height
    device_pixel_ratio = browser.execute_script("return window.devicePixelRatio;")
    assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

    config: BrowserConfig = {
        "win_top_bound": win_top_bound,
        "win_left_bound": win_left_bound,
        "win_width": win_width,
        "win_height": win_height,
        "win_right_bound": win_right_bound,
        "win_lower_bound": win_lower_bound,
        "device_pixel_ratio": device_pixel_ratio,
    }

    # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
    info: BrowserInfo = {"DOMTree": tree, "config": config}

    return info


def get_element_in_viewport_ratio(
    elem_left_bound: float,
    elem_top_bound: float,
    width: float,
    height: float,
    config: BrowserConfig,
) -> float:
    """Calculates the ratio of an element's bounding box that is visible within the viewport.

    This function compares the element's bounding box with the current viewport
    boundaries to determine the ratio of the element's area that is visible.

    Args:
        elem_left_bound (float): The left boundary of the element.
        elem_top_bound (float): The top boundary of the element.
        width (float): The width of the element.
        height (float): The height of the element.
        config (BrowserConfig): The browser window configuration.

    Returns:
        float: The ratio of the element's area visible within the viewport.
    """
    elem_right_bound = elem_left_bound + width
    elem_lower_bound = elem_top_bound + height

    win_left_bound = 0
    win_right_bound = config["win_width"]
    win_top_bound = 0
    win_lower_bound = config["win_height"]

    # Compute the overlap in x and y axes
    overlap_width = max(
        0,
        min(elem_right_bound, win_right_bound) - max(elem_left_bound, win_left_bound),
    )
    overlap_height = max(
        0,
        min(elem_lower_bound, win_lower_bound) - max(elem_top_bound, win_top_bound),
    )

    # Compute the overlap area
    ratio = overlap_width * overlap_height / width * height
    return ratio


def get_bounding_client_rect(browser, backend_node_id: str) -> Dict[str, Any]:
    """Retrieves the bounding client rectangle for an element in the browser.

    This function executes a CDP command to resolve a DOM node by its backend
    node ID and retrieves the bounding client rectangle (position and size)
    of the element.

    Args:
        browser: The browser instance used to execute CDP commands.
        backend_node_id (str): The backend node ID of the element.

    Returns:
        dict[str, Any]: The bounding client rectangle data for the element,
                         or an error message if the operation fails.
    """
    try:
        remote_object = browser.execute_cdp_cmd(
            "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
        )
        remote_object_id = remote_object["object"]["objectId"]
        response = browser.execute_cdp_cmd(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                    function() {
                        if (this.nodeType == 3) {
                            var range = document.createRange();
                            range.selectNode(this);
                            var rect = range.getBoundingClientRect().toJSON();
                            range.detach();
                            return rect;
                        } else {
                            return this.getBoundingClientRect().toJSON();
                        }
                    }
                """,
                "returnByValue": True,
            },
        )
        return response
    except:
        return {"result": {"subtype": "error"}}


def fetch_page_accessibility_tree(
    info: BrowserInfo,
    browser: webdriver,
    # client: CDPSession,
    current_viewport_only: bool,
) -> AccessibilityTree:
    """Fetches the accessibility tree of the page and filters nodes that are
    not in the current viewport.

    This function retrieves the accessibility tree via the Chrome DevTools Protocol
    and processes the nodes based on their visibility within the current viewport.

    Args:
        info (BrowserInfo): The browser information, including the DOM tree and configuration.
        browser: The browser instance used to execute CDP commands.
        current_viewport_only (bool): Whether to filter the tree to only include nodes
                                      within the current viewport.

    Returns:
        AccessibilityTree: The filtered list of accessibility tree nodes.
    """
    accessibility_tree: AccessibilityTree = browser.execute_cdp_cmd(
        "Accessibility.getFullAXTree", {}
    )["nodes"]

    # a few nodes are repeated in the accessibility tree
    seen_ids = set()
    _accessibility_tree = []
    for node in accessibility_tree:
        if node["nodeId"] not in seen_ids:
            _accessibility_tree.append(node)
            seen_ids.add(node["nodeId"])
    accessibility_tree = _accessibility_tree

    nodeid_to_cursor = {}
    for cursor, node in enumerate(accessibility_tree):
        nodeid_to_cursor[node["nodeId"]] = cursor
        # usually because the node is not visible etc
        if "backendDOMNodeId" not in node:
            node["union_bound"] = None
            continue
        backend_node_id = str(node["backendDOMNodeId"])
        if node["role"]["value"] == "RootWebArea":
            # always inside the viewport
            node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
        else:
            response = get_bounding_client_rect(browser, backend_node_id)
            if response.get("result", {}).get("subtype", "") == "error":
                node["union_bound"] = None
            else:
                x = response["result"]["value"]["x"]
                y = response["result"]["value"]["y"]
                width = response["result"]["value"]["width"]
                height = response["result"]["value"]["height"]
                node["union_bound"] = [x, y, width, height]

    # filter nodes that are not in the current viewport
    if current_viewport_only:

        def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
            # update the node information in the accessibility tree
            nodeid = node["nodeId"]
            node_cursor = nodeid_to_cursor[nodeid]
            parent_nodeid = node["parentId"]
            children_nodeids = node["childIds"]
            parent_cursor = nodeid_to_cursor[parent_nodeid]
            # update the children of the parent node
            assert accessibility_tree[parent_cursor].get("parentId", "Root") is not None
            # remove the nodeid from parent's childIds
            index = accessibility_tree[parent_cursor]["childIds"].index(nodeid)
            accessibility_tree[parent_cursor]["childIds"].pop(index)
            # Insert children_nodeids in the same location
            for child_nodeid in children_nodeids:
                accessibility_tree[parent_cursor]["childIds"].insert(
                    index, child_nodeid
                )
                index += 1
            # update children node's parent
            for child_nodeid in children_nodeids:
                child_cursor = nodeid_to_cursor[child_nodeid]
                accessibility_tree[child_cursor]["parentId"] = parent_nodeid
            # mark as removed
            accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

        config = info["config"]
        for node in accessibility_tree:
            if not node["union_bound"]:
                remove_node_in_graph(node)
                continue

            [x, y, width, height] = node["union_bound"]

            # invisible node
            if width == 0 or height == 0:
                remove_node_in_graph(node)
                continue

            in_viewport_ratio = get_element_in_viewport_ratio(
                elem_left_bound=float(x),
                elem_top_bound=float(y),
                width=float(width),
                height=float(height),
                config=config,
            )

            if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                remove_node_in_graph(node)

        accessibility_tree = [
            node
            for node in accessibility_tree
            if node.get("parentId", "Root") != "[REMOVED]"
        ]

    return accessibility_tree


def parse_accessibility_tree(
    accessibility_tree: AccessibilityTree,
) -> tuple[str, dict[str, Any]]:
    """Parses the accessibility tree into a human-readable string format.

    This function recursively traverses the accessibility tree and formats each
    node's details, including its role, name, and properties, into a string. The
    result can be used for debugging or understanding the structure of the accessibility tree.

    Args:
        accessibility_tree (AccessibilityTree): The list of nodes in the accessibility tree.

    Returns:
        tuple[str, dict[str, Any]]: A tuple containing the formatted accessibility tree string
                                     and a dictionary with additional node information.
    """
    node_id_to_idx = {}
    for idx, node in enumerate(accessibility_tree):
        node_id_to_idx[node["nodeId"]] = idx

    obs_nodes_info = {}

    def dfs(idx: int, obs_node_id: str, depth: int) -> str:
        tree_str = ""
        node = accessibility_tree[idx]
        indent = "\t" * depth
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]
            node_str = f"[{obs_node_id}] {role} {repr(name)}"
            properties = []
            for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
                    )
                except KeyError:
                    pass

            if properties:
                node_str += " " + " ".join(properties)

            # check valid
            if not node_str.strip():
                valid_node = False

            # empty generic node
            if not name.strip():
                if not properties:
                    if role in [
                        "generic",
                        "img",
                        "list",
                        "strong",
                        "paragraph",
                        "banner",
                        "navigation",
                        "Section",
                        "LabelText",
                        "Legend",
                        "listitem",
                    ]:
                        valid_node = False
                elif role in ["listitem"]:
                    valid_node = False

            if valid_node:
                tree_str += f"{indent}{node_str}"
                obs_nodes_info[obs_node_id] = {
                    "backend_id": node["backendDOMNodeId"],
                    "union_bound": node["union_bound"],
                    "text": node_str,
                }

        except:
            valid_node = False

        for _, child_node_id in enumerate(node["childIds"]):
            if child_node_id not in node_id_to_idx:
                continue
            # mark this to save some tokens
            child_depth = depth + 1 if valid_node else depth
            child_str = dfs(node_id_to_idx[child_node_id], child_node_id, child_depth)
            if child_str.strip():
                if tree_str.strip():
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
    return tree_str, obs_nodes_info


def clean_accesibility_tree(tree_str: str) -> str:
    """Further cleans the accessibility tree string by removing certain redundant nodes.

    This function filters out lines in the accessibility tree string that contain
    static text that has already appeared recently, ensuring the output is more concise.

    Args:
        tree_str (str): The string representation of the accessibility tree.

    Returns:
        str: The cleaned-up accessibility tree string.
    """
    clean_lines: list[str] = []
    for line in tree_str.split("\n"):
        if "statictext" in line.lower():
            prev_lines = clean_lines[-3:]
            pattern = r"\[\d+\] StaticText '([^']+)'"

            match = re.search(pattern, line)
            if match:
                static_text = match.group(1)
                if all(static_text not in prev_line for prev_line in prev_lines):
                    clean_lines.append(line)
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines)
