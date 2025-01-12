"""Unit tests for the OSWorld Bridge OSWorld Benchmark and Example Retriever."""

import tempfile

from typing import Dict

import pytest

from agential.benchmarks.computer_use.osworld.osworld_data_loader import (
    OSWorldDataManager,
)

EXAMPLES_DIR: str = (
    "agential/benchmarks/computer_use/osworld/evaluation_examples/examples"
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
