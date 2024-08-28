# 🚀 Contributing 

👍🎉 First off, thanks for taking the time to contribute! 🎉👍

Whether it's a bug report, new feature, correction, or additional documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary information to effectively respond to your bug report or contribution.

## 1. 💪 Getting Started

### 🍴 Fork the Repository

To start your journey, you'll need your very own copy of agential. Think of it as your innovation lab. 🧪

- Navigate to the agential repository on GitHub.
- In the upper-right corner, click the `Fork` button.


### Create and Switch to a New Branch

Create a new branch for your contribution and make sure to add your name and the purpose of the branch.

```
git checkout -b <your-name>/<branch-purpose>
```

Check the "Developer Setup" below for the developer workflow.

### 👍🎉 Submit a Pull Request (PR)

- Create a Pull Request from your branch to the main repository. Make sure to include a detailed description of your changes and reference any related issues.

### 🤝 Collaborate

- Be responsive to comments and feedback on your PR.
- Make necessary updates as suggested.
- Once your PR is approved, it will be merged into the main repository. 


## 2. 🔨 Developer Setup

First install [anaconda](https://docs.anaconda.com/free/anaconda/install/windows/) and follow the recommended settings. 

```
conda --version
```

Ensure you have `make`. if it's not installed on your computer, Please follow this [article](https://earthly.dev/blog/makefiles-on-windows/).


Use the following command to create a conda environment `agential` with Python version 3.11.5. Any Python version above 3.11 is viable.

```
conda create -n agential python=3.11.5
```

Now activate the environment.

```
conda activate agential
```

Please ensure you are in your virtual environment prior to beginning the next step. The Anaconda Command Prompt should have changed from `(base) C:\Users\<username>>` ---> `(agential) C:\Users\<username>>`. The change from (base) to (agential) indicates that you are now in your virtual environment.

Next, we will install [Poetry](https://python-poetry.org/docs/) using [pipx](https://pipx.pypa.io/stable/docs/).

```
pip install pipx
pipx install poetry
```

Make sure to add poetry to path by adding `C:\Users\<username>\.local\bin` to path in advanced system settings. For other operating systems, the path will be different. Ensure poetry is in the environment variable paths.

To ensure that pipx has been successfully installed, type in the command:

```
pipx --version
``` 

This should output the version if it has been installed properly. Then, to verify the poetry installation, type in the command:

``` 
poetry --version
```
This will output the poetry version and verify the existence of the poetry CLI.

Then clone the repository and enter the agential directory.

``` 
git clone https://github.com/alckasoc/agential/
```

Finally install all of the packages.

```
poetry install
```

### Verifying Environment Installation

To verify your environment is correctly installed, please run the following commands.

```
make lint
```
This command will execute the pre-made `lint` target in the Makefile, which, internally, uses `mypy`, `black`, and `ruff`.

If this command fails to run, check if Poetry has properly installed by running (same as previous section):
```
poetry --version
```

Next, run `auto_lint`. This will execute the pre-made `auto_lint` target in the Makefile which automatically formats your code with `black` and `ruff`.

```
make auto_lint
```

Finally, run this command:

```
make test
```

The `test` command within the Makefile internally runs `pytest` on unit tests located in the `tests` directory. 

### Setting up the `.env`

To test your implementations, you will most likely need an API key. API keys are kept locally in the `.env` file. At the root directory, create a `.env` file and include your relevant API keys. 

To use them, simply:

```
import os
import dotenv

dotenv.load_dotenv()
api_key = os.getenv("<NAME OF YOUR API KEY>")
```

### Pre-commit Install/Uninstall (Optional)

Pre-commit automatically runs specified checks before each commit to ensure code quality, adherence to best practices, and error prevention. If the checks fail, the commit is rejected until the issues are resolved. 

We have `pre-commit` as a developer-sided code quality checker, but often times you may find it slowing down your development! To uninstall `pre-commit`, run:
```
pre-commit uninstall
```

Then, later, should you choose to use it, run:
```
pre-commit install
```


## 3. 🧭 Navigating the Repository

Head to the `agential` where the project source code is. Within the `cog` directory, you'll find various modules and subdirectories catering to different aspects of the project's functionality.

- `agent`: Agent implementations
- `eval`: Evaluation module
- `functional`: Low-level functions for implementing agents
- `modules`: Submodules with specific functionalities like memory, planning, reflection, and scoring.
- `persona`: Default persona for agents, if applicable
- `prompts`: Agent prompts and a few-shot examples
- `utils`: Utility functions for fetching, formatting, parsing, etc.


Please take a look at the [README](https://github.com/alckasoc/agential/blob/main/README.md) for a well-structured overview of the project!

## 4. ⚒️ What do I work on?

You can start by browsing through our list of [issues](https://github.com/alckasoc/agential/issues) or suggesting your own!

Once you’ve decided on an issue, leave a comment and wait for approval! We don't want multiple people on a single issue unless the issue stresses it! 

If you’re ever in doubt about whether or not a proposed feature aligns with our library, feel free to raise an issue about it and we’ll get back to you promptly!

## 5. ❓  Questions

Feel free to contact [Vincent Tu](https://www.linkedin.com/in/vincent%2Dtu%2D422b18208/), our lead contributor. We're very friendly and welcoming to new contributors, so don't hesitate to reach out! 🙂

## Thank you for your consideration in joining us 😃🤝🙏
