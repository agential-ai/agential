"""Unit tests for OSWorld data manager."""

import json
import os
import tempfile

from typing import Dict
from unittest.mock import MagicMock

from agential.benchmarks.computer_use.osworld.data_manager import (
    OSWorldDataManager,
)

EXAMPLES_DIR: str = (
    "agential/benchmarks/computer_use/osworld/evaluation_examples/examples"
)
PATH_TO_GOOGLE_SETTINGS: str = (
    "agential/benchmarks/computer_use/osworld/evaluation_examples/settings/google/settings.json"
)
PATH_TO_GOOGLEDRIVE_SETTINGS: str = (
    "agential/benchmarks/computer_use/osworld/evaluation_examples/settings/googledrive/settings.yml"
)


def test_init() -> None:
    """Test OSWorld_Env constructor."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the temporary directory as the examples directory
        env = OSWorldDataManager(examples_dir=temp_dir)

        # Assertions
        assert env.examples_dir == temp_dir
        assert isinstance(env.data, Dict)


def test_load_data() -> None:
    """Test load_data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the temporary directory as the examples directory
        env = OSWorldDataManager(examples_dir=temp_dir)

        # Assertions
        assert env.examples_dir == temp_dir
        assert isinstance(env.data, Dict)


def test_get() -> None:
    """Test get."""
    example_1 = {
        "id": "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc",
        "snapshot": "chrome",
        "instruction": "Could you help me extract data in the table from a new invoice uploaded to my Google Drive, then export it to a Libreoffice calc .xlsx file in the desktop?",
        "source": "https://marketplace.uipath.com/listings/extract-data-from-a-new-invoice-file-in-google-drive-and-store-it-in-google-sheets4473",
        "config": [
            {
                "type": "googledrive",
                "parameters": {
                    "settings_file": "evaluation_examples/settings/googledrive/settings.yml",
                    "operation": ["delete", "upload"],
                    "args": [
                        {"query": "title = 'invoice.pdf'", "trash": False},
                        {
                            "url": "https://drive.usercontent.google.com/download?id=1KAhoPFM0AU2dgn_NRt3y7CjOr9Er4vwD&export=download&authuser=0&confirm=t&uuid=e8528cd1-5106-45f3-a644-e1bbf5e08278&at=APZUnTUnTuXfV2Ted_9Wv2QomMvA:1706181110208",
                            "path": ["invoice.pdf"],
                        },
                    ],
                },
            },
            {
                "type": "launch",
                "parameters": {
                    "command": ["google-chrome", "--remote-debugging-port=1337"]
                },
            },
            {
                "type": "launch",
                "parameters": {
                    "command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]
                },
            },
            {
                "type": "chrome_open_tabs",
                "parameters": {
                    "urls_to_open": ["https://news.google.com", "https://x.com"]
                },
            },
            {
                "type": "login",
                "parameters": {
                    "settings_file": "evaluation_examples/settings/google/settings.json",
                    "platform": "googledrive",
                },
            },
        ],
        "trajectory": "trajectories/",
        "related_apps": ["libreoffice_calc", "chrome"],
        "evaluator": {
            "func": "compare_table",
            "result": {
                "type": "vm_file",
                "path": "/home/user/Desktop/invoice.xlsx",
                "dest": "invoice.xlsx",
            },
            "expected": {
                "type": "cloud_file",
                "path": "https://drive.usercontent.google.com/download?id=1gkATnr8bk4JKQbzXZvzifoAQUA2sx5da&export=download&authuser=0&confirm=t&uuid=64ed0549-1627-49e8-8228-1e1925d6f6f7&at=APZUnTXkCm24SrOPuO5C6v4M3BiB:1706181091638",
                "dest": "invoice_gold.xlsx",
            },
            "options": {
                "rules": [
                    {"type": "sheet_data", "sheet_idx0": "RI0", "sheet_idx1": "EI0"}
                ]
            },
        },
    }
    example_2 = {
        "id": "0c825995-5b70-4526-b663-113f4c999dd2",
        "snapshot": "libreoffice_calc",
        "instruction": "I'm working on a comprehensive report for our environmental policy review meeting next week. I need to integrate key insights from an important document, which is a guidebook on the Green Economy, where I'm particularly interested in the 'Introduction' section. Could you extract this section and compile them into a new Google Doc named 'environment_policy_report (draft)' under /environment_policy folder? This will significantly aid in our discussion on aligning our environmental policies with sustainable and green economic practices. Thanks!",
        "source": "authors",
        "config": [
            {
                "type": "googledrive",
                "parameters": {
                    "settings_file": "evaluation_examples/settings/googledrive/settings.yml",
                    "operation": ["delete"],
                    "args": [
                        {
                            "query": "title = 'environment_policy_report (draft).doc' or title = 'environment_policy_report (draft).docx' or title = 'environment_policy_report (draft)'",
                            "trash": False,
                        }
                    ],
                },
            },
            {
                "type": "launch",
                "parameters": {
                    "command": ["google-chrome", "--remote-debugging-port=1337"]
                },
            },
            {
                "type": "launch",
                "parameters": {
                    "command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]
                },
            },
            {
                "type": "login",
                "parameters": {
                    "settings_file": "evaluation_examples/settings/google/settings.json",
                    "platform": "googledrive",
                },
            },
            {
                "type": "command",
                "parameters": {"command": ["mkdir", "-p", "/home/user/Desktop/wwf"]},
            },
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "path": "/home/user/Desktop/wwf/lpr_living_planet_report_2016.pdf",
                            "url": "https://drive.google.com/uc?id=19NCdw_MVP6nH5nC6okYYe8U1mJABfTRK&export=download",
                        },
                        {
                            "path": "/home/user/Desktop/wwf/279c656a32_ENGLISH_FULL.pdf",
                            "url": "https://drive.google.com/uc?id=1ckH1NetfImQ9EyONTO-ZFWA8m8VIUFvD&export=download",
                        },
                        {
                            "path": "/home/user/Desktop/wwf/7g37j96psg_WWF_AR2021_spreads.pdf",
                            "url": "https://drive.google.com/uc?id=1cxLTzmqDKMomOyvho29lvFvhRnb0Y8__&export=download",
                        },
                        {
                            "path": "/home/user/Desktop/GE Guidebook.pdf",
                            "url": "https://drive.google.com/uc?id=1KzC_R3eI3Rmgwz5bkcI8Ohv7ebOrU-Is&export=download",
                        },
                        {
                            "path": "/home/user/Desktop/assessing_and_reporting_water_quality(q&a).pdf",
                            "url": "https://drive.google.com/uc?id=1LFojf3Weflv3fVdrZrgTY1iUaRdbT9kG&export=download",
                        },
                    ]
                },
            },
        ],
        "trajectory": "trajectories/0c825995-5b70-4526-b663-113f4c999dd2",
        "related_apps": ["libreoffice_calc", "chrome", "os"],
        "evaluator": {
            "func": "compare_docx_files",
            "result": {
                "type": "googledrive_file",
                "settings_file": "evaluation_examples/settings/googledrive/settings.yml",
                "path": ["environment_policy", "environment_policy_report (draft)"],
                "dest": "environment_policy_report (draft).docx",
            },
            "expected": {
                "type": "cloud_file",
                "path": "https://drive.google.com/uc?id=1A2ti9JncAfIa6ks7FTJWHtYlZo-68FtM&export=download",
                "dest": "environment_policy_report (draft)_gold.docx",
            },
            "options": {"content_only": True},
        },
    }
    temp_data = {
        "multi_apps": {
            "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc": example_1,
            "0c825995-5b70-4526-b663-113f4c999dd2": example_2,
        }
    }

    domain = "multi_apps"
    task_id = "0c825995-5b70-4526-b663-113f4c999dd2"
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the temporary directory as the examples directory
        env = OSWorldDataManager(examples_dir=temp_dir)

    env.data = temp_data
    # If domain and task_id are not none
    result_1 = env.get(domain=domain, task_id=task_id)
    assert result_1 == example_2

    # If domain is not none and task_is none
    result_2 = env.get(domain=domain)
    assert result_2 == temp_data["multi_apps"]

    # If task_id is not none and domain none
    result_3 = env.get(task_id=task_id)
    assert result_3 == example_2

    # If domain and task_id are none
    result_4 = env.get()
    assert result_4 == temp_data


def test_change_example_credential(
    osworld_google_settings_json_path: str, osworld_googledrive_settings_yml_path: str
) -> None:
    """Test changing credential."""
    manager = OSWorldDataManager(
        mode="benchmark",
        test_type="test_small",
        path_to_google_settings=osworld_google_settings_json_path,
        path_to_googledrive_settings=osworld_googledrive_settings_yml_path,
    )

    example = {
        "config": [
            {
                "type": "googledrive",
                "parameters": {
                    "settings_file": "evaluation_examples/settings/googledrive/file.yml",
                },
            },
            {
                "type": "login",
                "parameters": {
                    "settings_file": "evaluation_examples/settings/google/file.json",
                    "platform": "googledrive",
                },
            },
        ],
        "evaluator": {
            "func": "compare_docx_files",
            "result": {
                "type": "googledrive_file",
                "settings_file": "evaluation_examples/settings/googledrive/file.yml",
                "path": ["environment_policy", "environment_policy_report (draft)"],
                "dest": "environment_policy_report (draft).docx",
            },
        },
    }

    updated_example = manager._change_example_credential(example)
    assert updated_example["config"][0]["parameters"]["settings_file"].endswith(
        "settings.yml"
    )
    assert updated_example["config"][1]["parameters"]["settings_file"].endswith(
        "settings.json"
    )
    assert updated_example["evaluator"]["result"]["settings_file"].endswith(
        "settings.yml"
    )


def test_update_credentials(
    osworld_google_settings_json_path: str, osworld_googledrive_settings_yml_path: str
) -> None:
    """Test update_credentials."""
    manager = OSWorldDataManager(
        mode="benchmark",
        test_type="test_small",
        path_to_google_settings=osworld_google_settings_json_path,
        path_to_googledrive_settings=osworld_googledrive_settings_yml_path,
    )

    example = {
        "config": [
            {
                "type": "googledrive",
                "parameters": {
                    "settings_file": "evaluation_examples/settings/googledrive/file.yml",
                },
            },
            {
                "type": "login",
                "parameters": {
                    "settings_file": "evaluation_examples/settings/google/file.json",
                    "platform": "googledrive",
                },
            },
        ],
        "evaluator": {
            "func": "compare_docx_files",
            "result": {
                "type": "googledrive_file",
                "settings_file": "evaluation_examples/settings/googledrive/file.yml",
                "path": ["environment_policy", "environment_policy_report (draft)"],
                "dest": "environment_policy_report (draft).docx",
            },
        },
    }

    manager.data = {
        "multi_apps": {
            "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc": example,
        }
    }

    manager._update_credentials()
    assert manager.data["multi_apps"]["4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc"]["config"][
        0
    ]["parameters"]["settings_file"].endswith("settings.yml")
    assert manager.data["multi_apps"]["4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc"]["config"][
        1
    ]["parameters"]["settings_file"].endswith("settings.json")
    assert manager.data["multi_apps"]["4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc"][
        "evaluator"
    ]["result"]["settings_file"].endswith("settings.yml")


def test_get_all_domains() -> None:
    """Test the get_all_domains method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "domain1"))
        os.makedirs(os.path.join(temp_dir, "domain2"))

        manager = OSWorldDataManager(mode="custom", examples_dir=temp_dir)
        manager.data = {
            "domain1": {
                "task1": {"metadata": "info1"},
                "task2": {"metadata": "info2"},
            },
            "domain2": {
                "task3": {"metadata": "info3"},
            },
        }

        expected_domains = ["domain1", "domain2"]
        assert set(manager.get_all_domains()) == set(expected_domains)


def test_get_task_ids_by_domain() -> None:
    """Test the get_task_ids_by_domain method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock directory and files
        domain1_dir = os.path.join(temp_dir, "domain1")
        os.makedirs(domain1_dir)

        with open(os.path.join(domain1_dir, "task1.json"), "w") as f:
            json.dump({"metadata": "info1"}, f)
        with open(os.path.join(domain1_dir, "task2.json"), "w") as f:
            json.dump({"metadata": "info2"}, f)

        manager = OSWorldDataManager(mode="custom", examples_dir=temp_dir)

        assert manager.get_task_ids_by_domain("domain1") == ["task1", "task2"]
        assert manager.get_task_ids_by_domain("nonexistent") == []


def test_get() -> None:
    """Test the get method of OSWorldDataManager for retrieving tasks and domains."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock directory and files
        domain1_dir = os.path.join(temp_dir, "domain1")
        domain2_dir = os.path.join(temp_dir, "domain2")
        os.makedirs(domain1_dir)
        os.makedirs(domain2_dir)

        with open(os.path.join(domain1_dir, "task1.json"), "w") as f:
            json.dump({"metadata": "info1"}, f)
        with open(os.path.join(domain1_dir, "task2.json"), "w") as f:
            json.dump({"metadata": "info2"}, f)
        with open(os.path.join(domain2_dir, "task3.json"), "w") as f:
            json.dump({"metadata": "info3"}, f)

        manager = OSWorldDataManager(mode="custom", examples_dir=temp_dir)

        # Retrieve specific task
        assert manager.get("domain1", "task1") == {"metadata": "info1"}

        # Retrieve all tasks in a domain
        assert manager.get("domain1") == {
            "task1": {"metadata": "info1"},
            "task2": {"metadata": "info2"},
        }

        # Retrieve task by task_id across domains
        assert manager.get(task_id="task3") == {"metadata": "info3"}

        # Retrieve all data
        assert manager.get() == manager.data

        # Nonexistent domain or task
        assert manager.get("nonexistent", "task1") is None
        assert manager.get("domain1", "nonexistent") is None


def test_get_data() -> None:
    """Test the get_data method of OSWorldDataManager for retrieving flattened and hierarchical data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock directory and files
        domain1_dir = os.path.join(temp_dir, "domain1")
        domain2_dir = os.path.join(temp_dir, "domain2")
        os.makedirs(domain1_dir)
        os.makedirs(domain2_dir)

        with open(os.path.join(domain1_dir, "task1.json"), "w") as f:
            json.dump({"metadata": "info1"}, f)
        with open(os.path.join(domain1_dir, "task2.json"), "w") as f:
            json.dump({"metadata": "info2"}, f)
        with open(os.path.join(domain2_dir, "task3.json"), "w") as f:
            json.dump({"metadata": "info3"}, f)

        manager = OSWorldDataManager(mode="custom", examples_dir=temp_dir)

        # Flattened data
        flattened_data = manager.get_data(flatten=True)
        expected_flattened_data = {
            "domain1__task1": {"metadata": "info1"},
            "domain1__task2": {"metadata": "info2"},
            "domain2__task3": {"metadata": "info3"},
        }
        assert flattened_data == expected_flattened_data

        # Hierarchical data
        assert manager.get_data(flatten=False) == manager.data


def test_get_domains_summary() -> None:
    """Test the get_domains_summary method of OSWorldDataManager for counting tasks per domain."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock directory and files
        domain1_dir = os.path.join(temp_dir, "domain1")
        domain2_dir = os.path.join(temp_dir, "domain2")
        os.makedirs(domain1_dir)
        os.makedirs(domain2_dir)

        with open(os.path.join(domain1_dir, "task1.json"), "w") as f:
            json.dump({"metadata": "info1"}, f)
        with open(os.path.join(domain1_dir, "task2.json"), "w") as f:
            json.dump({"metadata": "info2"}, f)
        with open(os.path.join(domain2_dir, "task3.json"), "w") as f:
            json.dump({"metadata": "info3"}, f)

        manager = OSWorldDataManager(mode="custom", examples_dir=temp_dir)

        summary = manager.get_domains_summary()
        expected_summary = {
            "domain1": 2,  # 2 tasks in domain1
            "domain2": 1,  # 1 task in domain2
        }
        assert summary == expected_summary
