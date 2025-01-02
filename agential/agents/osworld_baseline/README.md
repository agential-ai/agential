# Agent
## Prompt-based Agents

- [Code Reference](https://github.com/xlang-ai/OSWorld/tree/main)
- [Paper Reference](https://arxiv.org/pdf/2404.07972)

### Supported Models
We currently support the following models as the foundational models for the agents:
- `GPT-3.5` (gpt-3.5-turbo-16k, ...)
- `GPT-4` (gpt-4-0125-preview, gpt-4-1106-preview, ...)
- `Gemini-Pro`
- `Gemini-Pro-Vision`
- `Claude-3, 2` (claude-3-haiku-2024030, claude-3-sonnet-2024022, ...)
- ...

And those from the open-source community:
- `Mixtral 8x7B`
- `QWEN`, `QWEN-VL`
- `CogAgent`
- `Llama3`
- ...

### How to use

```python
from agential.agents.OSWorldBaseline.agent import OSWorldBaselineAgent
with open("../tests/assets/osworldbaseline/accessibility_tree.txt", "r", encoding="utf-8") as file:
    accessibility_tree = file.read()

instruction = "Please help me to find the nearest restaurant."
obs = {
    "screenshot": open("path/to/observation.jpg", 'rb').read(), "accessibility_tree": accessibility_tree
}

agent = OSWorldBaselineAgent(
    model=LLM(model="gpt-4o"),
    observation_type="screenshot",
)

response, actions, messages = agent.generate(instruction, obs)
```

### Observation Space and Action Space
We currently support the following observation spaces:
- `a11y_tree`: the accessibility tree of the current screen
- `screenshot`: a screenshot of the current screen
- `screenshot_a11y_tree`: a screenshot of the current screen with the accessibility tree overlay
- `som`: the set-of-mark trick on the current screen, with table metadata included.

And the following action spaces:
- `pyautogui`: valid Python code with `pyautogui` code valid
- `computer_13`: a set of enumerated actions designed by us
```
