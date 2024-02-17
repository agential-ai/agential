# Discussion-agents

# introduction 

- Clean implementations of popular LLM-based agents in various domains

- Flexible, modular components include modules for memory management, agent modeling, core functionalities, planning, reflection, scoring, and utility methods




# Features

Natural Language Understanding:

Contextual Reasoning:

Topic Exploration

Multi-Agent Interaction

Customization and Extension: 



# Functions


## ReAct 


Concept
----

When a question is posed, the agent engages in a thought-action-observation cycle, mirroring the human cognitive process. It begins with formulating a thought, followed by taking action based on that thought. Finally, it observes the outcome of that action. Through these stages, the agent can generate answers by interacting with the Wikipedia API.



SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS (https://arxiv.org/pdf/2210.03629.pdf)


## Reflexion 


Concept
----


The Reflexion module utilizes verbal reinforcement to enhance output. It employs the Actor, Evaluator, and Self-Reflection models to generate constructive feedback, merging it with observational data. This combined input is then processed by the Self-Reflection Model to produce insightful reflections, stored in the agent's experience database. Ultimately, the goal is to provide valuable insights and guidance to improve the Actor's outputs.

Reflexion: Language Agents with Verbal Reinforcement Learning (https://arxiv.org/pdf/2303.11366.pdf)

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

# Getting started example

```python
 """Test generate."""
    q = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'


    llm = OpenAI(openai_api_key="YOUR_API_KEY")
    agent = ReActAgent(llm=llm)
    out = agent.generate(question=q)

```

```python
out = 
    ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nThought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about the controversies and crimes.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
    ' Search[best kick boxer]\nObservation: The best kick boxer in the world is often a highly debated topic, but some notable names include Semmy Schilt, Peter Aerts, Ernesto Hoost, and Ramon Dekkers.\nThought: Since the question mentions controversies and crimes, I should focus on more recent kick boxers. I will look up the controversies and crimes of Semmy Schilt.\nAction: Lookup[controversies and crimes]\nObservation: (Result 1/1) Semmy Schilt has been involved in several controversies, including accusations of using performance-enhancing drugs and unsportsmanlike conducts such as eye-gouging and low blows.\nThought: The question mentions "unsportsmanlike conducts" specifically, so I will look up more information on those incidents.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result 1/1) Semmy Schilt has been known for his aggressive and sometimes controversial fighting style, with incidents such as eye-gouging and low blows being reported by his opponents.\nThought: The question also mentions crimes outside of the ring, so I will search for any criminal record or charges against Semmy Schilt.\nAction: Search[Semmy Schilt criminal record]\nObservation',



```






Project Organization
------------

 

    ├── data
    │   ├── external                        <- Data from third party sources.
    │   ├── interim                         <- Intermediate data that has been transformed.
    │   ├── processed                       <- The final, canonical data sets for modeling.
    │   └── raw                             <- The original, immutable data dump.
    │
    ├── discussion_agents                   <- Source code for this project.
    │   ├── cog   
    │   │   ├── agent                       <- Model/agent-related modules.
    │   │   │   ├── base.py                 <-
    │   │   │   ├── generative_agents.py
    │   │   │   ├── react.py                <- ReActagent and Zeroshotagent modules
    │   │   │   └── reflexion.py            <->
    │   │   │   
    │   │   ├── eval                        <- Agent core modules.
    │   │   │   └── reflexion.py
    │   │   │   
    │   │   ├── functional                  
    │   │   │   ├── generative_agents.py
    │   │   │   ├── react.py
    │   │   │   └── reflexion.py
    │   │   │
    │   │   ├── modules           
    │   │   │   ├── memory
    │   │   │   ├── plan
    │   │   │   ├── reflect
    │   │   │   └── score
    │   │   ├── persona             
    │   │   │    ├── base.py
    │   │   │    └── generative_agent.py
    │   │   │
    │   │   └── prompts             
    │   │       ├── react.py
    │   │       └── reflexion.py
    │   │
    │   │
    │   └── utils   
    │       ├── fetch.py
    │       ├── format.py
    │       └── parse.py 
    ├── docs                 <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models               <- Trained and serialized models, model predictions, or model summaries.
    │       
    ├── notebooks            <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                           the creator's initials, and a short `-` delimited description, e.g.
    │                           `1.0-jqp-initial-data-exploration`.
    │   ├── 1.0-vt-yt-extract.ipynb                     <- Extract audio from video
    │   ├── 2.0-vt-langchain-playground.ipynb
    │   ├── 3.0-vt-test-langchain-agents.ipynb
    │   ├── 4.0-vt-langchain-gen-agent-classes.ipynb
    │   ├── 5.0-vt-test-planning.ipynb
    │   ├── 6.0-vt-planning-gen-agents.ipynb
    │   ├── 7.0-vt-test-imports.ipynb
    │
    ├── references           <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures          <- Generated graphics and figures to be used in reporting.
    │
    ├── tests                <- Tests.

---------



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
