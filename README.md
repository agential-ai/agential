
<h3 align="center">
  <img
    src="https://raw.githubusercontent.com/agential-ai/.github/main/profile/banner_dark.svg#gh-dark-mode-only"
  />
  <img
    src="https://raw.githubusercontent.com/agential-ai/.github/main/profile/banner_light.svg#gh-light-mode-only"
  />
</h3>

<h3 align="center">
  <p style="font-size:3vw" align="center">Language agent research made easy.</p>
  <p align="center"><a href="https://www.youtube.com/watch?v=5syJjBQ_k6o">You're definitely not you when you're hungry for research.</a></p>
</h3>


<h3 align="center">

[![codecov](https://codecov.io/gh/agential-ai/agential/branch/main/graph/badge.svg)](https://codecov.io/gh/agential-ai/agential)
</h3>


## Features

- 7 different agent methods across 9 different reasoning/decision-making benchmarks!
- Easy-to-Use Interface: Provides intuitive and user-friendly functions for rapid prototyping and development.
- Modularized Implementations: Includes modularized implementations of popular LLM-based agents methods.


## Getting Started 

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


## Project Organization

------------

    ├── agential                       <- Source code for this project.
    │   ├── cog   
    │   │   ├── agent                  <- Model/agent-related modules.
    │   │   │     ├── strategies
    │   │   │     │       ├── base.py
    │   │   │     │       ├── qa.py
    │   │   │     │       ├── math.py
    │   │   │     │       └── code.py
    │   │   │     │
    │   │   │     ├── agent.py
    │   │   │     ├── functional.py
    │   │   │     ├── output.py
    │   │   │     ├── prompts.py
    │   │   │     └── <modules>.py
    │   │
    │   ├── eval                       <- Evaluation-related modules.
    │   │
    │   ├── llm                        <- LLM class.
    │   │
    │   └── utils                      <- Utility methods.
    │       
    ├── docs                           <- An mkdocs project.
    │
    │       
    ├── notebooks                      <- Jupyter notebooks. Naming convention is a number 
    │                                    (for ordering), the creator's initials, and a short `-` delimited │ description, e.g. `1.0-jqp-initial-data-exploration`.
    │  
    │
    ├── references                     <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                        <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                    <- Generated graphics and figures to be used in reporting.
    │
    └── tests                          <- Tests.

---------

## 🙏 Acknowledgement

## 😀 Contributing

If you want to contribute, please check the [contributing.md](https://github.com/alckasoc/agential/blob/main/CONTRIBUTING.md) for guidelines!
Please check out the [project document timeline](https://equatorial-jobaria-9ad.notion.site/Project-Lifecycle-Management-70d65e9a76eb4c86b6aed007f717aa41?pvs=4) on Notion and reach out to us if you have any questions!

## 😶‍🌫️ Contact Us!

If you have any questions or suggestions, please feel free to reach out to tuvincent0106@gmail.com!