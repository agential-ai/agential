# discussion-agents
Discussing agents provides a framework for building intelligent conversational agents, facilitating natural language understanding, generation, and interaction
 
# Project Desiption

- Clean implementations of popular LLM-based agents in various domains
- Flexible, modular components include modules for memory management, agent modeling, core functionalities, planning, reflection, scoring, and utility methods

* Natural Language Understanding: Discussion Agents utilize advanced natural language processing (NLP) techniques to comprehend user inputs, including text-based messages and spoken language.
* Contextual Reasoning: The agents employ contextual reasoning mechanisms to interpret and respond to user queries in a manner that reflects understanding of the ongoing conversation.
* Topic Exploration: Users can engage with the agents to explore a wide range of topics, from current events and scientific concepts to personal interests and philosophical ideas.
* Multi-Agent Interaction: The project supports interactions between multiple agents, allowing users to participate in group discussions, debates, and collaborative problem-solving activities.
* Customization and Extension: Discussion Agents are designed to be modular and extensible, enabling developers to customize agent behavior, add new features, and integrate with external systems and services.

### Sample input
```python
user_input = "What is the weather forecast for tomorrow?"
# Running the agent
response = react_agent.respond(user_input)
# Displaying the agent's response
print("Agent's response:", response)
```

This README serves as a guide to help you get started with the library, including setup instructions and usage guidelines.

# Developer Setup: Windows

First install [anaconda](https://docs.anaconda.com/free/anaconda/install/windows/) and add to path by going to advanced system settings. Then launch cmd.

Use the following command to create a conda environment `discussion-agents` with Python version 3.10.13. Any Python version above 3.9 is viable.
```
conda create -n discussion-agents python=3.10.13
```
Now activate the environent.
```
conda activate discussion-agents
```
Next, we will install [Poetry](https://python-poetry.org/docs/) using [pipx](https://pipx.pypa.io/stable/docs/).
```
pip install pipx
pipx install poetry
```
Make sure to add poetry to path by adding `C:\Users\<username>\.local\bin` to path in advanced system settings. For other operating systems, the path will be different. Ensure poetry is in the environment variable paths.

Then clone the repository and enter the discussion-agents directory.
``` 
git clone https://github.com/alckasoc/discussion-agents/

```
Finally install all of the packages.
```
poetry install
```

Project Organization
------------

    ├── data
    │   ├── external         <- Data from third party sources.
    │   ├── interim          <- Intermediate data that has been transformed.
    │   ├── processed        <- The final, canonical data sets for modeling.
    │   └── raw              <- The original, immutable data dump.
    │
    ├── discussion_agents    <- Source code for this project.
    │   │
    │   ├── memory           <- Memory-related modules.
    │   │   └── ...
    │   │
    │   ├── agent            <- Model/agent-related modules.
    │   │   └── ...
    │   │
    │   ├── core             <- Agent core modules.
    │   │   └── ...
    │   │
    │   ├── planning         <- Planning-related modules.
    │   │   └── ...
    │   │
    │   ├── reflecting       <- Reflecting-related modules.
    │   │   └── ...
    │   │
    │   ├── scoring          <- Scoring-related modules.
    │   │   └── ...
    │   │
    │   ├── utils            <- Utility methods.
    │   │   └── ...
    │
    ├── docs                 <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models               <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks            <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                           the creator's initials, and a short `-` delimited description, e.g.
    │                           `1.0-jqp-initial-data-exploration`.
    │
    ├── references           <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures          <- Generated graphics and figures to be used in reporting.
    │
    ├── tests                <- Tests.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
