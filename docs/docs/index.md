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

## Types of errors

### CRITIC

Sure! Here's the section on CRITIC errors organized into a table for better readability:

## Types of errors

### CRITIC

| Dataset  | Example | Error Type                         |
|----------|---------|------------------------------------|
| MBPP     | 1       | Logical error                      |
| MBPP     | 2       | Logical error                      |
| MBPP     | 3       | Logical error                      |
| MBPP     | 4       | No error                           |
| MBPP     | 5       | No error                           |
| GSM8K    | 1       | Code efficiency                    |
| GSM8K    | 2       | NameError (var not defined)        |
| GSM8K    | 3       | Logical error                      |
| GSM8K    | 4       | Logical error                      |
| GSM8K    | 5       | Logical error                      |
| SVAMP    | 1       | Logical error                      |
| SVAMP    | 2       | No error                           |
| SVAMP    | 3       | Logical error                      |
| SVAMP    | 4       | No error                           |
| SVAMP    | 5       | No error                           |
| TabMWP   | 1       | Incorrect answer format            |
| TabMWP   | 2       | NameError (var not defined)        |
| TabMWP   | 3       | No error                           |
| TabMWP   | 4       | Incorrect answer format            |
| TabMWP   | 5       | No error                           |
| AMBIGNQ  | 1       | No error                           |
| AMBIGNQ  | 2       | No error                           |
| AMBIGNQ  | 3       | Incorrect answer/answer format     |
| AMBIGNQ  | 4       | No error                           |
| AMBIGNQ  | 5       | Incorrect answer/answer format     |
| HOTPOTQA | 1       | No error                           |
| HOTPOTQA | 2       | No error                           |
| HOTPOTQA | 3       | No error                           |
| HOTPOTQA | 4       | Incorrect answer/answer format     |
| HOTPOTQA | 5       | Incorrect answer/answer format     |
| TRIVIAQA | 1       | Incorrect answer                   |
| TRIVIAQA | 2       | No error                           |
| TRIVIAQA | 3       | Incorrect answer                   |
| TRIVIAQA | 4       | No error                           |
| TRIVIAQA | 5       | No error                           |
| FEVER    | 1       | Incorrect answer                   |
| FEVER    | 2       | Incorrect answer format            |
| FEVER    | 3       | No error                           |
| FEVER    | 4       | No error                           |
| FEVER    | 5       | No error                           |

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
