"""OSWorld Benchmark"""
from desktop_env.providers.base import VMManager, Provider
from typing import Tuple
import os

from typing import Tuple

from desktop_env.providers.base import Provider, VMManager


def initializer(
    self,
    provider_name: str = "vmware",
    region: str = None,
    path_to_vm: str = None,
    snapshot_name: str = "init_state",
    action_space: str = "computer_13",
    cache_dir: str = "cache",
    screen_size: Tuple[int] = (1920, 1080),
    headless: bool = False,
    require_a11y_tree: bool = True,
    require_terminal: bool = False,
    os_type: str = "Ubuntu",
):
    """Args:
    provider_name (str): virtualization provider name, default to "vmware"
    region (str): the region for allocate machines, work for cloud services, default to  "us-east-1"
    path_to_vm (str): path to .vmx file
    snapshot_name (str): snapshot name to revert to, default to "init_state"
    action_space (str): "computer_13" | "pyautogui"
    cache_dir (str): cache directory to cache task-related stuffs like
    reference file for evaluation
    screen_size (Tuple[int]): screen size of the VM
    headless (bool): whether to run the VM in headless mode
    require_a11y_tree (bool): whether to require accessibility tree
    require_terminal (bool): whether to require terminal output.
    """
    # Initialize VM manager and vitualization provider
    self.region = region

    # Default
    self.server_port = 5000
    self.chromium_port = 9222
    self.vnc_port = 8006
    self.vlc_port = 8080
    self.manager, self.provider = _create_vm_manager_and_provider(provider_name, region)

    self.os_type = os_type

    # Initialize environment variables
    if path_to_vm:
        self.path_to_vm = (
            os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_vm)))
            if provider_name in {"vmware", "virtualbox"}
            else path_to_vm
        )
    else:
        self.path_to_vm = self.manager.get_vm_path(self.os_type, region)

    self.snapshot_name = snapshot_name
    self.cache_dir_base: str = cache_dir
    # todo: add the logic to get the screen size from the VM
    self.headless = headless
    self.require_a11y_tree = require_a11y_tree
    self.require_terminal = require_terminal

    # Initialize emulator and controller
    if provider_name != "docker":  # Check if this is applicable to other VM providers
        self._start_emulator()

    # mode: human or machine
    self.instruction = None
    assert action_space in ["computer_13", "pyautogui"]
    self.action_space = action_space  # todo: refactor it to the ActType

    # episodic stuffs, like counters, will be updated or reset
    # when calling self.reset()
    self._traj_no: int = -1
    self._step_no: int = 0
    self.action_history: List[Dict[str, any]] = []


def _create_vm_manager_and_provider(provider_name: str, region: str):
    """Factory function to get the Virtual Machine Manager and Provider instances based on the provided provider name."""
    provider_name = provider_name.lower().strip()
    if provider_name == "vmware":
        from desktop_env.providers.vmware.manager import VMwareVMManager
        from desktop_env.providers.vmware.provider import VMwareProvider

        return VMwareVMManager(), VMwareProvider(region)
    elif provider_name == "virtualbox":
        from desktop_env.providers.virtualbox.manager import VirtualBoxVMManager
        from desktop_env.providers.virtualbox.provider import VirtualBoxProvider

        return VirtualBoxVMManager(), VirtualBoxProvider(region)
    elif provider_name in ["aws", "amazon web services"]:
        from desktop_env.providers.aws.manager import AWSVMManager
        from desktop_env.providers.aws.provider import AWSProvider

        return AWSVMManager(), AWSProvider(region)
    elif provider_name == "azure":
        from desktop_env.providers.azure.manager import AzureVMManager
        from desktop_env.providers.azure.provider import AzureProvider

        return AzureVMManager(), AzureProvider(region)
    else:
        raise NotImplementedError(f"{provider_name} not implemented!")
