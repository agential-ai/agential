
# Agential

[![codecov](https://codecov.io/gh/agential-ai/agential/branch/main/graph/badge.svg)](https://codecov.io/gh/agential-ai/agential)


## Features


 Our primary goal is to provide easy-to-use and clean implementations of popular LLM-based agent methods: an encyclopedia! This library is one of our contributions for our research project empirically surveying and investigating the performance of language agent methods across a diverse set of reasoning/decision-making tasks. 
 
 Learn more about this [here](https://equatorial-jobaria-9ad.notion.site/Project-Lifecycle-Management-70d65e9a76eb4c86b6aed007f717aa41?pvs=4)! 

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
    │   │   │   
    │   │   ├── eval                   <- Agent core modules.
    │   │   │   
    │   │   ├── functional                  
    │   │   │
    │   │   ├── modules           
    │   │   │   ├── memory             <- Memory-related modules.
    │   │   │   ├── plan               <- Planning-related modules.
    │   │   │   ├── reflect            <- Reflecting-related modules.
    │   │   │   └── score              <- Scoring-related modules.
    │   │   │
    │   │   ├── persona             
    │   │   │
    │   │   └── prompts             
    │   │
    │   └── utils                      <- Utility methods.
    │       
    ├── docs                           <- An mkdocs project.
    │
    ├── models                         <- Trained and serialized models, model predictions,
    │                                          or model summaries.
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

## Contributing

If you want to contribute, please check the [contributing.md](https://github.com/alckasoc/agential/blob/main/CONTRIBUTING.md) for guidelines!
