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
| Self-Refine              | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| ExpeL                    | :material-check-all: | :material-check-all: | :octicons-check-16:  | :octicons-check-16:  |
| LATS                     | :material-check-all: | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |

| **Methods / Benchmarks** |        GSM8k         |        SVAMP         |        TabMWP        |
| ------------------------ | :------------------: | :------------------: | :------------------: |
| ReAct                    | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| Reflexion                | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| CRITIC                   | :material-check-all: | :material-check-all: | :material-check-all: |
| Self-Refine              | :material-check-all: | :octicons-check-16:  | :octicons-check-16:  |
| ExpeL                    | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| LATS                     | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |

| **Methods / Benchmarks** |         MBPP         |      HumanEval       |
| ------------------------ | :------------------: | :------------------: |
| ReAct                    | :octicons-check-16:  | :octicons-check-16:  |
| Reflexion                | :material-check-all: | :material-check-all: |
| CRITIC                   | :octicons-check-16:  | :octicons-check-16:  |
| Self-Refine              | :octicons-check-16:  | :octicons-check-16:  |
| ExpeL                    | :octicons-check-16:  | :octicons-check-16:  |
| LATS                     |                      |                      |

| **Methods / Benchmarks** | ALFWorld | WebShop | AgentBench |
| ------------------------ | :------: | :-----: | :--------: |
| ReAct                    |          |         |            |
| Reflexion                |          |         |            |
| CRITIC                   |          |         |            |
| Self-Refine              |          |         |            |
| ExpeL                    |          |         |            |
| LATS                     |          |         |            |

## Experimenting...

| **Methods / Benchmarks** | HotpotQA | FEVER | TriviaQA | AmbigNQ |
| ------------------------ | :------: | :---: | :------: | :-----: |
| ReAct                    |          |       |          |         |
| Reflexion                |          |       |          |         |
| CRITIC                   |          |       |          |         |
| Self-Refine              |          |       |          |         |
| ExpeL                    |          |       |          |         |
| LATS                     |          |       |          |         |

| **Methods / Benchmarks** | GSM8k | SVAMP | TabMWP |
| ------------------------ | :---: | :---: | :----: |
| ReAct                    |       |       |        |
| Reflexion                |       |       |        |
| CRITIC                   |       |       |        |
| Self-Refine              |       |       |        |
| ExpeL                    |       |       |        |
| LATS                     |       |       |        |

| **Methods / Benchmarks** | MBPP  | HumanEval |
| ------------------------ | :---: | :-------: |
| ReAct                    |       |           |
| Reflexion                |       |           |
| CRITIC                   |       |           |
| Self-Refine              |       |           |
| ExpeL                    |       |           |
| LATS                     |       |           |

| **Methods / Benchmarks** | ALFWorld | WebShop | AgentBench |
| ------------------------ | :------: | :-----: | :--------: |
| ReAct                    |          |         |            |
| Reflexion                |          |         |            |
| CRITIC                   |          |         |            |
| Self-Refine              |          |         |            |
| ExpeL                    |          |         |            |
| LATS                     |          |         |            |

## Types of errors

### CRITIC

Sure! Here's the section on CRITIC errors organized into a table for better readability:

## Types of errors

### CRITIC

Certainly! Here's the section organized with each benchmark in a single row and the error types listed in a numbered format within the same cell:

## Types of errors

### CRITIC, Self-Refine

| Dataset   | Error Types                                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------------------------------ |
| HUMANEVAL | 1. Logical error<br>2. Logical error<br>3. Logical error<br>4. No error<br>5. No error                                   |
| MBPP      | 1. Logical error<br>2. Logical error<br>3. Logical error<br>4. No error<br>5. No error                                   |
| GSM8K     | 1. Code efficiency<br>2. NameError (var not defined)<br>3. Logical error<br>4. Logical error<br>5. Logical error         |
| SVAMP     | 1. Logical error<br>2. No error<br>3. Logical error<br>4. No error<br>5. No error                                        |
| TABMWP    | 1. Incorrect answer format<br>2. NameError (var not defined)<br>3. No error<br>4. Incorrect answer format<br>5. No error |
| AMBIGNQ   | 1. No error<br>2. No error<br>3. Incorrect answer/answer format<br>4. No error<br>5. Incorrect answer/answer format      |
| HOTPOTQA  | 1. No error<br>2. No error<br>3. No error<br>4. Incorrect answer/answer format<br>5. Incorrect answer/answer format      |
| TRIVIAQA  | 1. Incorrect answer<br>2. No error<br>3. Incorrect answer<br>4. No error<br>5. No error                                  |
| FEVER     | 1. Incorrect answer<br>2. Incorrect answer format<br>3. No error<br>4. No error<br>5. No error                           |

### ReflexionCoT, ReflexionReAct

| **Benchmark** | **ReflexionCoT** | **ReflexionReAct** |
| ------------- | ---------------- | ------------------ |
| **HotpotQA**  | 1. Misinterpretation<br>2. Incorrect assumption<br>3. Misinterpretation<br>4. Misinterpretation<br>5. Misinterpretation | 1. Misled action<br>2. Misled action<br>3. Misread context<br>4. Wrong answer<br>5. Logical error |
| **FEVER**     | 1. Insufficient info<br>2. Misinterpretation<br>3. Insufficient info<br>4. Insufficient info<br>5. Misinterpretation | 1. Ignored context<br>2. Insufficient info<br>3. Insufficient info<br>4. Ignore context<br>5. Ignore context |
| **AmbigNQ**   | 1. Knowledge error<br>2. Knowledge error<br>3. Knowledge error<br>4. Misinterpret question<br>5. Knowledge error | 1. Incorrect assumption/Insufficient info<br>2. Insufficient info<br>3. Knowledge error<br>4. Incorrect answer format<br>5. Misread context |
| **TriviaQA**  | 1. Incorrect assumption<br>2. Incorrect assumption<br>3. Incorrect assumption<br>4. Misinterpretation<br>5. Incorrect assumption | 1. Ignore context<br>2. Ignore context<br>3. Ignore context<br>4. Ignore context<br>5. Ignore context |
| **GSM8K**     | 1. Logical error<br>2. Logical error<br>3. Misinterpret question<br>4. Logical error<br>5. Misinterpret question | 1. Logical error/Misinterpret question<br>2. Logical error/Misinterpret question<br>3. Logical error/Re-calculation error<br>4. Logical error/Re-calculation error<br>5. Logical error/Misinterpret question |
| **SVAMP**     | 1. Logical error<br>2. Logical error<br>3. Logical error<br>4. Logical error<br>5. Logical error | 1. Misinterpret question<br>2. Logical error<br>3. Logical error<br>4. Logical error<br>5. Logical error |
| **TabMWP**    | 1. Incorrect operator<br>2. Incorrect operator<br>3. Misinterpret question<br>4. Incorrect operator<br>5. Logical error | 1. Misinterpret question<br>2. Logical error<br>3. Logical error<br>4. Re-calculation error<br>5. Logical error |
| **HumanEval** | 1. Conceptual error<br>2. Logical error<br>3. Logical error<br>4. Logical error<br>5. Logical error | 1. Logical error<br>2. Logical error<br>3. Logical error<br>4. Logical error<br>5. Logical error |
| **MBPP**      | 1. Logical error<br>2. Logical error<br>3. Incorrect function usage<br>4. Logical error<br>5. Logical error | 1. Incorrect function implementation<br>2. Logical error<br>3. Incorrect function usage<br>4. Logical error<br>5. Logical error |

### LATS

| **Benchmark** | **LATS Reflect** | **LATS Value** |
| ------------- | ---------------- | -------------- |
| **HotpotQA**  | 1. Misled action<br>2. Misled action<br>3. Misread context<br>4. Wrong answer<br>5. Logical error | 1. wrong (2)<br>2. right (10)<br>3. wrong (2)<br>4. wrong (5)<br>5. wip (10) |
| **FEVER**     | 1. Ignored context<br>2. Insufficient info<br>3. Insufficient info<br>4. Ignore context<br>5. Ignore context | 1. wrong (2)<br>2. wrong (1)<br>3. wrong (2)<br>4. wip (10)<br>5. right (10) |
| **AmbigNQ**   | 1. Incorrect assumption/Insufficient info<br>2. Insufficient info<br>3. Knowledge error<br>4. Incorrect answer format<br>5. Misread context | 1.  wrong (2)<br>2.  wrong (3)<br>3.  wrong (4)<br>4.  wip (10)<br>5.  right (10) |
| **TriviaQA**  | 1. Ignore context<br>2. Ignore context<br>3. Ignore context<br>4. Ignore context<br>5. Ignore context | 1. wrong (2)<br>2. wrong (2)<br>3. wrong (3)<br>4. wip (10)<br>5. right (10) |
| **GSM8K**     | 1. Logical error/Misinterpret question<br>2. Logical error/Misinterpret question<br>3. Logical error/Re-calculation error<br>4. Logical error/Re-calculation error<br>5. Logical error/Misinterpret question | 1. wrong (1)<br>2. wrong (2)<br>3. wrong (6)<br>4. wip (10)<br>5. right (10) |
| **SVAMP**     | 1. Misinterpret question<br>2. Logical error<br>3. Logical error<br>4. Logical error<br>5. Logical error | 1. wrong (4)<br>2. wrong (3)<br>3. wrong (4)<br>4. wip (10)<br>5. right (10) |
| **TabMWP**    | 1. Misinterpret question<br>2. Logical error<br>3. Logical error<br>4. Re-calculation error<br>5. Logical error | 1. wrong (3)<br>2. wrong (4)<br>3. wrong (3)<br>4. wip (7)<br>5. right (10) |
| **HumanEval** | 1. Logical error<br>2. Logical error<br>3. Logical error<br>4. Logical error<br>5. Logical error | 1. wrong (2)<br>2. wrong (4)<br>3. wrong (3)<br>4. wip (5)<br>5. right (10) |
| **MBPP**      | 1. Incorrect function implementation<br>2. Logical error<br>3. Incorrect function usage<br>4. Logical error<br>5. Logical error | 1. wrong (3)<br>2. wrong (2)<br>3. wrong (3)<br>4. wip (9)<br>5. right (10) |

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
