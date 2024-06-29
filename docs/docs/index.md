<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

# Welcome to Agential!

## Benchmark Few-shot Examples

| **Benchmarks** | Number of few-shot examples |
| -------------- | --------------------------- |
| **HotpotQA**   |  6                          |
| **FEVER**      |  3                          |
| **TriviaQA**   |  4                          |
| **AmbigNQ**    |  5                          |
| **GSM8k**      |  8                          |
| **SVAMP**      |  7                          |
| **TabMWP**     |  4                          |
| **MBPP**       |  3                          |
| **HumanEval**  |  0                          |
| **ALFWorld**   |                             |
| **WebShop**    |                             |
| **AgentBench** |                             |


## Implementing...



- :octicons-check-16: **:** not tested in the original paper

- :material-check-all: **:** tested in the original paper

| **Methods / Benchmarks** |       HotpotQA       |        FEVER         |       TriviaQA       |       AmbigNQ        |
| ------------------------ | :------------------: | :------------------: | :------------------: | :------------------: |
| ReAct                    | :material-check-all: | :material-check-all: | :octicons-check-16:  | :octicons-check-16:  |
| Reflexion                | :material-check-all: | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| CRITIC                   | :material-check-all: | :octicons-check-16:  | :material-check-all: | :material-check-all: |
| LATS                     |                      |                      |                      |                      |

| **Methods / Benchmarks** |        GSM8k         |        SVAMP         |        TabMWP        |
| ------------------------ | :------------------: | :------------------: | :------------------: |
| ReAct                    | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| Reflexion                | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| CRITIC                   | :material-check-all: | :material-check-all: | :material-check-all: |
| LATS                     |                      |                      |                      |

| **Methods / Benchmarks** |         MBPP         |      HumanEval       |
| ------------------------ | :------------------: | :------------------: |
| ReAct                    | :octicons-check-16:  | :octicons-check-16:  |
| Reflexion                | :material-check-all: | :material-check-all: |
| CRITIC                   | :octicons-check-16:  | :octicons-check-16:  |
| LATS                     |                      |                      |

| **Methods / Benchmarks** | ALFWorld | WebShop | AgentBench |
| ------------------------ | :------: | :-----: | :--------: |
| ReAct                    |          |         |            |
| Reflexion                |          |         |            |
| CRITIC                   |          |         |            |
| LATS                     |          |         |            |

## Experimenting...


| **Methods / Benchmarks** | HotpotQA | FEVER | TriviaQA | AmbigNQ |
| ------------------------ | :------: | :---: | :------: | :-----: |
| ReAct                    |          |       |          |         |
| Reflexion                |          |       |          |         |
| CRITIC                   |          |       |          |         |
| LATS                     |          |       |          |         |

| **Methods / Benchmarks** | GSM8k | SVAMP | TabMWP |
| ------------------------ | :---: | :---: | :----: |
| ReAct                    |       |       |        |
| Reflexion                |       |       |        |
| CRITIC                   |       |       |        |
| LATS                     |       |       |        |

| **Methods / Benchmarks** | MBPP  | HumanEval |
| ------------------------ | :---: | :-------: |
| ReAct                    |       |           |
| Reflexion                |       |           |
| CRITIC                   |       |           |
| LATS                     |       |           |

| **Methods / Benchmarks** | ALFWorld | WebShop | AgentBench |
| ------------------------ | :------: | :-----: | :--------: |
| ReAct                    |          |         |            |
| Reflexion                |          |         |            |
| CRITIC                   |          |         |            |
| LATS                     |          |         |            |

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
