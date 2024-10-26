
<h3 align="center">
  <img
    src="https://raw.githubusercontent.com/agential-ai/.github/main/profile/banner_dark.svg#gh-dark-mode-only"
  />
  <img
    src="https://raw.githubusercontent.com/agential-ai/.github/main/profile/banner_light.svg#gh-light-mode-only"
  />
</h3>


<h3 align="center">Language agent experimentation made easy.</h3>

<h3 align="center">

[![codecov](https://codecov.io/gh/agential-ai/agential/branch/main/graph/badge.svg)](https://codecov.io/gh/agential-ai/agential)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</h3>

Agential provides clear implementations of popular LLM-based agents across a variety of reasoning/decision-making and language agent benchmarks, making it easy for researchers to evaluate and compare different agents.

## 🤔 Getting Started 

First, install the library with `pip`:

```
pip install agential
```

Next, let's query the `ReActAgent`!

```python
from agential.llm.llm import LLM
from agential.cog.react.agent import ReActAgent

question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?'

llm = LLM("gpt-3.5-turbo")
agent = ReActAgent(llm=llm, benchmark="hotpotqa")
out = agent.generate(question=question)
```

## 🙏 Acknowledgement

## 😀 Contributing

If you want to contribute, please check the [contributing.md](https://github.com/alckasoc/agential/blob/main/CONTRIBUTING.md) for guidelines!
Please check out the [project document timeline](https://equatorial-jobaria-9ad.notion.site/Project-Lifecycle-Management-70d65e9a76eb4c86b6aed007f717aa41?pvs=4) on Notion and reach out to us if you have any questions!

## 😶‍🌫️ Contact Us!

If you have any questions or suggestions, please feel free to reach out to tuvincent0106@gmail.com!