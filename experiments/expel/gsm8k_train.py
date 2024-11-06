"""Train ExpeL on GSM8K."""

import os
import warnings
import pickle

import numpy as np
import tiktoken

from agential.eval.metrics.classification import EM
from agential.utils.general import safe_execute

warnings.filterwarnings("ignore")

from agential.agents.expel.agent import ExpeL
from agential.agents.expel.memory import ExpeLExperienceMemory, ExpeLInsightMemory
from agential.agents.reflexion.agent import ReflexionReAct
from agential.core.llm import LLM
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

import wandb

wandb.login()
from datasets import load_dataset

from experiments.utils import set_seed

import argparse

parser = argparse.ArgumentParser(description="Train ExpeL")
parser.add_argument(
    "--n_train_samples", type=int, default=-1, help="Number of samples to train"
)
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model")
parser.add_argument(
    "--eval_model", type=str, default="gpt-4o", help="The evaluator model"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_reflections", type=int, default=3, help="Max reflections")
parser.add_argument("--max_trials", type=int, default=3, help="Max trials")
parser.add_argument("--max_steps", type=int, default=6, help="Max steps")
parser.add_argument("--max_tokens", type=int, default=5000, help="Max tokens")
parser.add_argument(
    "--experience_memory_strategy",
    type=str,
    default="task",
    help="Experience memory strategy",
)
parser.add_argument("--embedder", type=str, default="huggingface", help="Embedder")
parser.add_argument(
    "--experiences_path", type=str, default="", help="Experiences path (pkl)"
)
parser.add_argument("--insights_path", type=str, default="", help="Insights path (pkl)")
parser.add_argument(
    "--max_insights", type=int, default=20, help="Max number of insights"
)
parser.add_argument("--leeway", type=int, default=5, help="Leeway")
parser.add_argument(
    "--success_batch_size", type=int, default=8, help="Success batch size"
)
parser.add_argument(
    "--extract_init_insights",
    type=bool,
    default=True,
    help="Boolean to extract initial insights",
)
parser.add_argument("--patience", type=int, default=3, help="Patience")
parser.add_argument(
    "--reflect_strategy", type=str, default="reflexion", help="Reflection strategy"
)
parser.add_argument(
    "--use_dynamic_examples",
    type=bool,
    default=True,
    help="Boolean to use dynamic examples",
)
parser.add_argument(
    "--extract_insights", type=bool, default=True, help="Boolean to extract insights"
)
parser.add_argument("--k_docs", type=int, default=24, help="Number of docs to retrieve")
parser.add_argument("--num_fewshots", type=int, default=6, help="Number of fewshots")
parser.add_argument(
    "--max_fewshot_tokens", type=int, default=1500, help="Max tokens for fewshots"
)
parser.add_argument(
    "--reranker_strategy", type=str, default="none", help="Reranker strategy"
)
args = parser.parse_args()

set_seed(args.seed)
root_dir = "output"
method_name = "expel"
benchmark = "gsm8k"

if __name__ == "__main__":
    data = load_dataset("alckasoc/gsm8k_expel_train_100")["train"]

    n_train_samples = args.n_train_samples
    model = args.model
    eval_model = args.eval_model
    seed = args.seed
    max_reflections = args.max_reflections
    max_trials = args.max_trials
    max_steps = args.max_steps
    max_tokens = args.max_tokens
    experience_memory_strategy = args.experience_memory_strategy
    embedder = args.embedder
    experiences_path = args.experiences_path
    insights_path = args.insights_path
    max_insights = args.max_insights
    leeway = args.leeway
    success_batch_size = args.success_batch_size
    extract_init_insights = args.extract_init_insights
    patience = args.patience
    reflect_strategy = args.reflect_strategy
    use_dynamic_examples = args.use_dynamic_examples
    extract_insights = args.extract_insights
    k_docs = args.k_docs
    num_fewshots = args.num_fewshots
    max_fewshot_tokens = args.max_fewshot_tokens
    reranker_strategy = (
        args.reranker_strategy if args.reranker_strategy != "none" else None
    )

    if experiences_path:
        with open(experiences_path, "rb") as f:
            experiences = pickle.load(f)
    else:
        experiences = []

    if insights_path:
        with open(insights_path, "rb") as f:
            insights = pickle.load(f)
    else:
        insights = []

    embedder_dict = {"huggingface": HuggingFaceEmbeddings}

    output_path = os.path.join(root_dir, benchmark)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    llm = LLM(
        model,
        organization=os.getenv("OPENAI_ORGANIZATION"),
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=seed,
    )

    eval_llm = LLM(
        eval_model,
        organization=os.getenv("OPENAI_ORGANIZATION"),
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=seed,
    )

    try:
        enc = tiktoken.encoding_for_model(args.model)
    except:
        enc = tiktoken.get_encoding("gpt-3.5-turbo")

    reflexion_react_agent = ReflexionReAct(
        llm=llm,
        benchmark=benchmark,
        max_reflections=max_reflections,
        max_trials=max_trials,
        max_steps=max_steps,
        max_tokens=max_tokens,
        enc=enc,
    )

    agent = ExpeL(
        llm=llm,
        benchmark=benchmark,
        reflexion_react_agent=reflexion_react_agent,
        experience_memory=ExpeLExperienceMemory(
            experiences=experiences,
            strategy=experience_memory_strategy,
            embedder=embedder_dict[embedder](),
            encoder=enc,
        ),
        insight_memory=ExpeLInsightMemory(
            insights=insights, max_num_insights=max_insights, leeway=leeway
        ),
        success_batch_size=success_batch_size,
        extract_init_insights=extract_init_insights,
    )

    run = wandb.init(
        project=benchmark,
        entity="agential",
        config={
            "is_training": True,
            "n_train_samples": n_train_samples,
            "model": model,
            "eval_model": eval_model,
            "seed": seed,
            "max_reflections": max_reflections,
            "max_trials": max_trials,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "experience_memory_strategy": experience_memory_strategy,
            "embedder": embedder,
            "experiences_path": experiences_path,
            "max_insights": max_insights,
            "leeway": leeway,
            "success_batch_size": success_batch_size,
            "patience": patience,
            "reflect_strategy": reflect_strategy,
            "use_dynamic_examples": use_dynamic_examples,
            "extract_insights": extract_insights,
            "k_docs": k_docs,
            "num_fewshots": num_fewshots,
            "max_fewshot_tokens": max_fewshot_tokens,
            "reranker_strategy": reranker_strategy,
        },
        group=method_name,
        tags=[
            "is_training=True",
            f"n_train_samples={n_train_samples}",
            f"method={method_name}",
            f"model={model}",
            f"eval_model={eval_model}",
            f"seed={seed}",
            f"max_reflections={max_reflections}",
            f"max_trials={max_trials}",
            f"max_steps={max_steps}",
            f"max_tokens={max_tokens}",
            f"experience_memory_strategy={experience_memory_strategy}",
            f"embedder={embedder}",
            f"experiences_path={experiences_path}",
            f"max_insights={max_insights}",
            f"leeway={leeway}",
            f"success_batch_size={success_batch_size}",
            f"patience={patience}",
            f"reflect_strategy={reflect_strategy}",
            f"use_dynamic_examples={use_dynamic_examples}",
            f"extract_insights={extract_insights}",
            f"k_docs={k_docs}",
            f"num_fewshots={num_fewshots}",
            f"max_fewshot_tokens={max_fewshot_tokens}",
            f"reranker_strategy={reranker_strategy}",
        ],
    )

    eval_table_data = []
    perf_table_data = []
    em_scores = []
    outputs = []

    for idx, instance in enumerate(data):
        if n_train_samples != -1 and idx >= n_train_samples:
            break

        question = instance["question"]
        answer: str = instance["answer"]
        answer = str(float(answer.split("#### ")[-1].strip().replace(",", "")))

        # Inference.
        out = agent.generate(
            question=question,
            key=answer,
            reflect_strategy=reflect_strategy,
            use_dynamic_examples=use_dynamic_examples,
            extract_insights=extract_insights,
            patience=patience,
            k_docs=k_docs,
            num_fewshots=num_fewshots,
            max_fewshot_tokens=max_fewshot_tokens,
            reranker_strategy=reranker_strategy,
        )

        code_str = out.answer.replace("```python", "").replace("```", "").strip()
        pred_answers, _ = safe_execute(code_string=code_str)
        try:
            pred_answer = str(float(pred_answers[0]))
        except:
            pred_answer = "NaN"

        is_correct = int(EM(pred_answer, answer, is_numeric=True))

        # Update scores.
        em_scores.append(is_correct)

        # Update tables.
        eval_table_data.append([question, answer, pred_answer, out.answer, is_correct])
        perf_table_data.append(
            [
                out.total_prompt_tokens,
                out.total_completion_tokens,
                out.total_tokens,
                out.total_prompt_cost,
                out.total_completion_cost,
                out.total_cost,
                out.total_prompt_time,
                out.total_time,
            ]
        )

        # Update outputs.
        outputs.append(out)

        # Log metrics per instance.
        run.log(
            {
                "em": is_correct,
            }
        )

    # Calculate total scores.
    total_em = sum(em_scores) / len(em_scores)

    # Create tables.
    eval_table = wandb.Table(
        data=eval_table_data,
        columns=["question", "answer", "code_answer", "predicted_answer", "EM"],
    )
    perf_columns = [
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "total_prompt_cost (USD)",
        "total_completion_cost (USD)",
        "total_cost (USD)",
        "total_prompt_time (s)",
        "total_time (s)",
    ]
    perf_table = wandb.Table(data=perf_table_data, columns=perf_columns)

    # Save outputs as pkl.
    outputs_save_path = os.path.join(output_path, f"{run.name}.pkl")
    with open(outputs_save_path, "wb") as f:
        pickle.dump(outputs, f)

    # Save ExpeL experience/insights memory as pkl.
    expel_experience_memories_save_path = os.path.join(
        output_path, f"{run.name}-expel-exp-memories.pkl"
    )
    with open(expel_experience_memories_save_path, "wb") as f:
        pickle.dump(agent.strategy.experience_memory.experiences, f)

    expel_insights_memories_save_path = os.path.join(
        output_path, f"{run.name}-expel-insights-memories.pkl"
    )
    with open(expel_insights_memories_save_path, "wb") as f:
        pickle.dump(agent.strategy.insight_memory.insights, f)

    # Save outputs as artifact.
    artifact = wandb.Artifact(name=run.name, type="output")
    artifact.add_file(local_path=outputs_save_path, name="outputs.pkl")

    # Save ExpeL experience/insights memory separately for ease-of-use.
    artifact.add_file(
        local_path=expel_experience_memories_save_path, name="expel-exp-memories.pkl"
    )
    artifact.add_file(
        local_path=expel_insights_memories_save_path, name="expel-insights-memories.pkl"
    )
    artifact.save()

    # Log tables.
    run.log({f"{run.name}_eval": eval_table, f"{run.name}_perf": perf_table})

    # Log all metrics.
    column_averages = np.mean(np.array(perf_table_data, dtype=float), axis=0).tolist()
    column_sums = np.sum(np.array(perf_table_data, dtype=float), axis=0).tolist()
    run.log(
        {
            "total_em": total_em,
            **dict(zip([f"avg_{col}" for col in perf_columns], column_averages)),
            **dict(zip([f"sum_{col}" for col in perf_columns], column_sums)),
        }
    )

    run.finish()
