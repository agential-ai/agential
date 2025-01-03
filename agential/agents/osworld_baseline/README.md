# Agent
## Prompt-based Agents

Code and implementations are originally from:
- [Code: OSWorld Repository](https://github.com/xlang-ai/OSWorld/tree/main)
- [Paper: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/pdf/2404.07972)

### How to Use

```python
from agential.agents.osworld_baseline.output import OSWorldBaseOutput
from agential.agents.osworld_baseline.agent import OSWorldBaseline
from agential.core.llm import LLM

with open("../assets/osworld/accessibility_tree.txt", "r", encoding="utf-8") as file:
    accessibility_tree = file.read()

obs = {
    "screenshot": open("../assets/osworld/output_image.jpeg", 'rb').read(), "accessibility_tree": accessibility_tree
}

instruction = "Please help me to find the nearest restaurant."

agent = OSWorldBaseline(
    model=LLM(model="gpt-4o"),
    observation_type="screenshot",
)

osworld_base_output: OSWorldBaseOutput = agent.generate(instruction, obs)

responses = osworld_base_output.additional_info["response"]
actions = osworld_base_output.additional_info["actions"]
messages = osworld_base_output.additional_info["messages"]
```

### Observation Space & Action Space

**Observation Spaces**

Our agents can process the following observation spaces:

- a11y_tree: Accessibility tree of the current screen.
- screenshot: Screenshot of the current screen.
- screenshot_a11y_tree: Screenshot of the screen overlaid with the accessibility tree.
- som: Set-of-Marks (SOM) with table metadata included.

**Action Spaces**

The supported action spaces are:

- pyautogui: Execute valid Python code using the pyautogui library.
- computer_13: A predefined set of enumerated actions based on the OSWorld project.

