# 🚀 Contributing 

👍🎉 First off, thanks for taking the time to contribute! 🎉👍

Whether it's a bug report, new feature, correction, or additional documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary information to effectively respond to your bug report or contribution.

## 1. 💪 Getting Started

### 🍴 Fork the Repository

To start your journey, you'll need your very own copy of discussion-agents. Think of it as your innovation lab. 🧪

- Navigate to the discussion-agents repository on GitHub.
- In the upper-right corner, click the `Fork` button.


### Create and Switch to a New Branch

Create a new branch for your contribution and make sure to add your name and the purpose of the branch.

```
git checkout -b <your-name>/<branch-purpose>

```
### 👍🎉 Submit a Pull Request (PR)

- Create a Pull Request from your branch to the main repository. Make sure to include a detailed description of your changes and reference any related issues.

### 🤝 Collaborate

- Be responsive to comments and feedback on your PR.
- Make necessary updates as suggested.
- Once your PR is approved, it will be merged into the main repository. 

## Developer Setup

First install [anaconda](https://docs.anaconda.com/free/anaconda/install/windows/) and add to path by going to advanced system settings. Then, launch cmd.

Make sure to install 'make' if it's not installed on your computer. Please follow this [article](https://earthly.dev/blog/makefiles-on-windows/),if `make` is not installed .


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


### How 

## 2. 🧭 How To Navigate

Head to the `discussion_agents` where the project source code is. Within the `cog` directory, you'll find various modules and subdirectories catering to different aspects of the project's functionality.

- `agent`: Agent implementations
- `eval`: Evaluation module
- `functional`: Low-level functions for implementing agents
- `modules`: Submodules with specific functionalities like memory, planning, reflection, and scoring.
- `persona`: Default persona for agents, if applicable
- `prompts`: Agent prompts and a few-shot examples
- `utils`: Utility functions for fetching, formatting, parsing, etc.


Please take a look at the [README](https://github.com/alckasoc/discussion-agents/blob/main/README.md) for a well-structured overview of the project!

## 3. ⚒️ What do I work on?

You can start by browsing through our list of [issues](https://github.com/alckasoc/discussion-agents/issues) or suggesting your own!

Once you’ve decided on an issue, leave a comment and wait for approval! We don't want multiple people on a single issue unless the issue stresses it! 

If you’re ever in doubt about whether or not a proposed feature aligns with our library, feel free to raise an issue about it and we’ll get back to you promptly!

## 4. ❓  Questions/Collaboration

Feel free to contact [Vincent Tu](https://www.linkedin.com/in/vincent%2Dtu%2D422b18208/), our lead contributor. We're very friendly and welcoming to new contributors, so don't hesitate to reach out! 🙂


## 5. 👨‍💻 Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to make participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behaviour that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behaviour by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic address, without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behaviour and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behaviour.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned with this Code of Conduct, or to ban temporarily or permanently any contributor for other behaviours that they deem inappropriate, threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community. Examples of representing a project or community include using an official project e-mail address, posting via an official social media account, or acting as an appointed representative at an online or offline event. Representation of a project may be further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behaviour may be reported by contacting the project maintainer at [@Vincent Tu](https://github.com/alckasoc). All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project maintainer is obligated to maintain confidentiality about the reporter of an incident. Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions as determined by other members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/), version 1.4, available at http://contributor-covenant.org/version/1/4.


## Thank you for your consideration in joining us 😃🤝🙏