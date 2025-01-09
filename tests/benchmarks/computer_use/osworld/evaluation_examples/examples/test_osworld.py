"""Unit tests for the OSWorld Example Retriever."""

import os
import subprocess
import tempfile
import unittest

from unittest.mock import MagicMock, patch

import pytest

from agential.benchmarks.computer_use.osworld.evaluation_examples.examples.osworld import (
    OSWorld,
    OSWorldEnv,
)
from agential.benchmarks.computer_use.osworld.osworld_processor import OSWorldProcessor

EXAMPLES_DIR: str = (
    "agential/benchmarks/computer_use/osworld/evaluation_examples/examples"
)
PATH_TO_GOOGLE_SETTINGS: str = (
    "agential/benchmarks/computer_use/osworld/evaluation_examples/settings/google/settings.json"
)
PATH_TO_GOOGLEDRIVE_SETTINGS: str = (
    "agential/benchmarks/computer_use/osworld/evaluation_examples/settings/googledrive/settings.yml"
)
DOMAIN = "multi_apps"
TASK_ID = "0c825995-5b70-4526-b663-113f4c999dd2"


def test_init() -> None:
    """Test OSWorld constructor."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Define the VMX file name
    vmx_file_name = "temp-vm.vmx"

    # Create the full path to the VMX file
    vmx_file_path = os.path.join(temp_dir, vmx_file_name)

    # Add mock VMX content
    vmx_content = """
    .encoding = "UTF-8"
    config.version = "8"
    virtualHW.version = "19"
    memsize = "4096"
    numvcpus = "2"
    displayName = "Temporary Mock VM"
    guestOS = "ubuntu-64"
    ethernet0.present = "TRUE"
    ethernet0.connectionType = "nat"
    ethernet0.addressType = "generated"
    scsi0:0.present = "TRUE"
    scsi0:0.fileName = "temp-vm.vmdk"
    """

    # Write the VMX content to the temporary file
    with open(vmx_file_path, "w") as file:
        file.write(vmx_content.strip())

    vmx_file_path

    env_osworld_processor = MagicMock(spec=OSWorldProcessor)
    env_osworldenv = MagicMock(spec=OSWorldEnv)
    env = MagicMock(spec=OSWorld)
    env.examples_dir = EXAMPLES_DIR
    env.path_to_google_settings = PATH_TO_GOOGLE_SETTINGS
    env.path_to_googledrive_settings = PATH_TO_GOOGLEDRIVE_SETTINGS
    env.osworld_processor = env_osworld_processor
    env.osworld_env = env_osworldenv
    env.data = {}

    assert env.examples_dir == EXAMPLES_DIR
    assert env.path_to_google_settings == PATH_TO_GOOGLE_SETTINGS
    assert env.path_to_googledrive_settings == PATH_TO_GOOGLEDRIVE_SETTINGS
    assert isinstance(env.osworld_processor, OSWorldProcessor)
    assert isinstance(env.osworld_env, OSWorldEnv)
    assert env.data == {}


def test_change_credential() -> None:
    """Test change_credential."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Define the VMX file name
    vmx_file_name = "temp-vm.vmx"

    # Create the full path to the VMX file
    vmx_file_path = os.path.join(temp_dir, vmx_file_name)

    # Add mock VMX content
    vmx_content = """
    .encoding = "UTF-8"
    config.version = "8"
    virtualHW.version = "19"
    memsize = "4096"
    numvcpus = "2"
    displayName = "Temporary Mock VM"
    guestOS = "ubuntu-64"
    ethernet0.present = "TRUE"
    ethernet0.connectionType = "nat"
    ethernet0.addressType = "generated"
    scsi0:0.present = "TRUE"
    scsi0:0.fileName = "temp-vm.vmdk"
    """

    # Write the VMX content to the temporary file
    with open(vmx_file_path, "w") as file:
        file.write(vmx_content.strip())

    vmx_file_path

    example_1_input = {
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
    example_2_input = {
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
    temp_data_input = {
        "multi_apps": {
            "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc": example_1_input,
            "0c825995-5b70-4526-b663-113f4c999dd2": example_2_input,
        }
    }

    example_1_output = {
        "id": "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc",
        "snapshot": "chrome",
        "instruction": "Could you help me extract data in the table from a new invoice uploaded to my Google Drive, then export it to a Libreoffice calc .xlsx file in the desktop?",
        "source": "https://marketplace.uipath.com/listings/extract-data-from-a-new-invoice-file-in-google-drive-and-store-it-in-google-sheets4473",
        "config": [
            {
                "type": "googledrive",
                "parameters": {
                    "settings_file": PATH_TO_GOOGLEDRIVE_SETTINGS,
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
                    "settings_file": PATH_TO_GOOGLE_SETTINGS,
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
    example_2_output = {
        "id": "0c825995-5b70-4526-b663-113f4c999dd2",
        "snapshot": "libreoffice_calc",
        "instruction": "I'm working on a comprehensive report for our environmental policy review meeting next week. I need to integrate key insights from an important document, which is a guidebook on the Green Economy, where I'm particularly interested in the 'Introduction' section. Could you extract this section and compile them into a new Google Doc named 'environment_policy_report (draft)' under /environment_policy folder? This will significantly aid in our discussion on aligning our environmental policies with sustainable and green economic practices. Thanks!",
        "source": "authors",
        "config": [
            {
                "type": "googledrive",
                "parameters": {
                    "settings_file": PATH_TO_GOOGLEDRIVE_SETTINGS,
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
                    "settings_file": PATH_TO_GOOGLE_SETTINGS,
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
                "settings_file": PATH_TO_GOOGLEDRIVE_SETTINGS,
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
    temp_data_output = {
        "multi_apps": {
            "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc": example_1_output,
            "0c825995-5b70-4526-b663-113f4c999dd2": example_2_output,
        }
    }

    env_osworld_processor = MagicMock(spec=OSWorldProcessor)
    env_osworldenv = MagicMock(spec=OSWorldEnv)
    env = MagicMock(spec=OSWorld)
    env.examples_dir = EXAMPLES_DIR
    env.path_to_google_settings = PATH_TO_GOOGLE_SETTINGS
    env.path_to_googledrive_settings = PATH_TO_GOOGLEDRIVE_SETTINGS
    env.osworld_processor = env_osworld_processor
    env.osworld_env = env_osworldenv
    env.data = {}

    env._change_credential.return_value = temp_data_output

    result = env._change_credential(examples=temp_data_input)

    assert result == temp_data_output


def test_update_credential() -> None:
    """Test update_credential."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Define the VMX file name
    vmx_file_name = "temp-vm.vmx"

    # Create the full path to the VMX file
    vmx_file_path = os.path.join(temp_dir, vmx_file_name)

    # Add mock VMX content
    vmx_content = """
    .encoding = "UTF-8"
    config.version = "8"
    virtualHW.version = "19"
    memsize = "4096"
    numvcpus = "2"
    displayName = "Temporary Mock VM"
    guestOS = "ubuntu-64"
    ethernet0.present = "TRUE"
    ethernet0.connectionType = "nat"
    ethernet0.addressType = "generated"
    scsi0:0.present = "TRUE"
    scsi0:0.fileName = "temp-vm.vmdk"
    """

    # Write the VMX content to the temporary file
    with open(vmx_file_path, "w") as file:
        file.write(vmx_content.strip())

    vmx_file_path

    example_1_input = {
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
    example_2_input = {
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
    temp_data_input = {
        "multi_apps": {
            "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc": example_1_input,
            "0c825995-5b70-4526-b663-113f4c999dd2": example_2_input,
        }
    }

    example_1_output = {
        "id": "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc",
        "snapshot": "chrome",
        "instruction": "Could you help me extract data in the table from a new invoice uploaded to my Google Drive, then export it to a Libreoffice calc .xlsx file in the desktop?",
        "source": "https://marketplace.uipath.com/listings/extract-data-from-a-new-invoice-file-in-google-drive-and-store-it-in-google-sheets4473",
        "config": [
            {
                "type": "googledrive",
                "parameters": {
                    "settings_file": PATH_TO_GOOGLEDRIVE_SETTINGS,
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
                    "settings_file": PATH_TO_GOOGLE_SETTINGS,
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
    example_2_output = {
        "id": "0c825995-5b70-4526-b663-113f4c999dd2",
        "snapshot": "libreoffice_calc",
        "instruction": "I'm working on a comprehensive report for our environmental policy review meeting next week. I need to integrate key insights from an important document, which is a guidebook on the Green Economy, where I'm particularly interested in the 'Introduction' section. Could you extract this section and compile them into a new Google Doc named 'environment_policy_report (draft)' under /environment_policy folder? This will significantly aid in our discussion on aligning our environmental policies with sustainable and green economic practices. Thanks!",
        "source": "authors",
        "config": [
            {
                "type": "googledrive",
                "parameters": {
                    "settings_file": PATH_TO_GOOGLEDRIVE_SETTINGS,
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
                    "settings_file": PATH_TO_GOOGLE_SETTINGS,
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
                "settings_file": PATH_TO_GOOGLEDRIVE_SETTINGS,
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
    temp_data_output = {
        "multi_apps": {
            "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc": example_1_output,
            "0c825995-5b70-4526-b663-113f4c999dd2": example_2_output,
        }
    }

    env_osworld_processor = MagicMock(spec=OSWorldProcessor)
    env_osworldenv = MagicMock(spec=OSWorldEnv)
    env = MagicMock(spec=OSWorld)
    env.examples_dir = EXAMPLES_DIR
    env.path_to_google_settings = PATH_TO_GOOGLE_SETTINGS
    env.path_to_googledrive_settings = PATH_TO_GOOGLEDRIVE_SETTINGS
    env.osworld_processor = env_osworld_processor
    env.osworld_env = env_osworldenv
    env.data = {}

    # If domain and task_id not none
    env.update_credential.return_value = example_2_output
    result_1 = env.update_credential(domain=DOMAIN, task_id=TASK_ID)
    assert result_1 == example_2_output
    # If domain is not none and task_id is none
    env.update_credential.return_value = temp_data_output["multi_apps"]
    result_2 = env.update_credential(domain=DOMAIN, task_id=TASK_ID)
    assert result_2 == temp_data_output["multi_apps"]
    # If task_id is not none and domain is none
    env.update_credential.return_value = example_2_output
    result_3 = env.update_credential(domain=DOMAIN, task_id=TASK_ID)
    assert result_3 == example_2_output
    # If task_id and domain are none
    env.update_credential.return_value = temp_data_output
    result_4 = env.update_credential(domain=DOMAIN, task_id=TASK_ID)
    assert result_4 == temp_data_output


def test_reset() -> None:
    """Test reset."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Define the VMX file name
    vmx_file_name = "temp-vm.vmx"

    # Create the full path to the VMX file
    vmx_file_path = os.path.join(temp_dir, vmx_file_name)

    # Add mock VMX content
    vmx_content = """
    .encoding = "UTF-8"
    config.version = "8"
    virtualHW.version = "19"
    memsize = "4096"
    numvcpus = "2"
    displayName = "Temporary Mock VM"
    guestOS = "ubuntu-64"
    ethernet0.present = "TRUE"
    ethernet0.connectionType = "nat"
    ethernet0.addressType = "generated"
    scsi0:0.present = "TRUE"
    scsi0:0.fileName = "temp-vm.vmdk"
    """

    # Write the VMX content to the temporary file
    with open(vmx_file_path, "w") as file:
        file.write(vmx_content.strip())

    vmx_file_path

    obs = {"screenshot": "screenshot in bytes", "accessibility_tree": "tree"}

    env_osworld_processor = MagicMock(spec=OSWorldProcessor)
    env_osworldenv = MagicMock(spec=OSWorldEnv)
    env = MagicMock(spec=OSWorld)
    env.examples_dir = EXAMPLES_DIR
    env.path_to_google_settings = PATH_TO_GOOGLE_SETTINGS
    env.path_to_googledrive_settings = PATH_TO_GOOGLEDRIVE_SETTINGS
    env.osworld_processor = env_osworld_processor
    env.osworld_env = env_osworldenv
    env.data = {}

    env.reset.return_value = obs
    result = env.reset(domain=DOMAIN, task_id=TASK_ID)

    assert result == obs
    pass
