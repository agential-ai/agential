
from agential.benchmarks.computer_use.osworld.evaluation_examples.examples.osworld_env import OSWorldEnv
from agential.benchmarks.computer_use.osworld.osworld import OSWorldProcessor

TYPE_TO_LOOK = ["googledrive", "login", "googledrive_file"]

class OSWorld():
    # How does this class work:
        # need function to 
    def __init__(
        examples_dir: str,
        path_to_google_settings: str,
        path_to_googledrive_settings: str,
        osworld_processor: OSWorldProcessor
    ):
        self.examples_dir = examples_dir
        self.path_to_google_settings = path_to_google_settings

        self.osworld_env = OSWorldEnv(self.examples_dir)
        self.data = self.osworld_env.data
        self.osworld_processor = osworld_processor

    def _change_credential(self, example: Dict[str, Any]) -> Any:
        for item in example["config"]:
            if item["type"] in TYPE_TO_LOOK:
                file_type = item["parameters"]["settings_file"].split(".")[-1]
                if file_type == "yml":
                    item["parameters"]["settings_file"] = "change yml file"
                else:
                    item["parameters"]["settings_file"] = "change json file"

        path = example["evaluator"]["result"]
        if path["type"] in TYPE_TO_LOOK and path["settings_file"].split(".")[-1] == "yml":
            path["settings_file"] = "change yml file"

        return example

    def update_credential(self, domain: str = None, task_id: str = None) -> Dict[str, Any]:
        if domain not None and task_id not None:
            return self._change_credential(self.data[domain][task_id])
        elif domain not None:
            temp_data: Dict[str, Any] = {}
            for each_task in self.data[domain].keys():
                temp_data[domain][each_task] = self._change_credential(each_task)
            return temp_data
        elif task_id not None:
            temp_data: Dict[str, Any] = {}
            for each_domain in self.data.keys():
                temp_data[each_domain][task_id] = self._change_credential(each_domain[task_id])
            return temp_data
        else:
            temp_data: Dict[str, Any] = {}
            for each_domain in self.data.keys():
                for each_task in self.data[each_domain].keys():
                    temp_data[each_domain][each_task] = self._change_credential(each_task)
            return temp_data

    def reset(self, domain: str = None, task_id: str = None) -> Dict[str, Any]:
        example = self.update_credential(
            domain=domain,
            task_id=task_id
        )
        return self.osworld_processor.reset(task_config=example)

        




    