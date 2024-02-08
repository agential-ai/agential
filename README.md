# discussion-agents

Discussion agents.

# Windows installation 
```
1.First install anaconda https://docs.anaconda.com/free/anaconda/install/windows/ and add to path, or launch anaconda console
2.conda create -n my_env python=3.10.13
3.conda activate my_env
4.pip install pipx
5.pipx install poetry (make sure to add poetry to path by adding C:\Users\<username\.local\bin to path in advanced system settings)
6.In the discussion agents directory - poetry install
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
