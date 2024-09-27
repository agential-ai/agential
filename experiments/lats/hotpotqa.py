"""Run LATS on HotpotQA."""
import numpy as np
import tiktoken
from agential.agents.lats.agent import LATS
from agential.eval.metrics.classification import EM, f1, precision, recall
import os
import pickle

import warnings

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from agential.core.llm import LLM

from experiments.utils import set_seed

import wandb
wandb.login()
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description="Run ReflexionReAct experiments.")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model")
parser.add_argument("--eval_model", type=str, default="gpt-4o", help="The evaluator model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--n_samples", type=int, default=5, help="Number of samples")
parser.add_argument("--max_reflections", type=int, default=4, help="Max reflections")
parser.add_argument("--depth_limit", type=int, default=7, help="Depth limit")
parser.add_argument("--max_unique", type=int, default=5, help="Max unique")
parser.add_argument("--cache_value", type=bool, default=True, help="Cache value")
parser.add_argument("--max_iterations", type=int, default=30, help="Max trials")
args = parser.parse_args()

set_seed(args.seed)
root_dir = "output"
method_name = "reflexion_react"
benchmark = "hotpotqa"

if __name__ == '__main__':
    data = load_dataset("alckasoc/hotpotqa_500")['train']

    model = args.model
    eval_model = args.eval_model
    seed = args.seed
    n_samples= args.n_samples
    depth_limit= args.depth_limit
    max_unique= args.max_unique
    cache_value= args.cache_value
    max_reflections= args.max_reflections
    max_iterations= args.max_iterations

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
        seed=seed
    )

    eval_llm = LLM(
        eval_model,
        organization=os.getenv("OPENAI_ORGANIZATION"),
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=seed
    )

    try:
        enc = tiktoken.encoding_for_model(args.model)
    except:
        enc = tiktoken.get_encoding("gpt-3.5-turbo")

    method = LATS(
        llm=llm,
        benchmark=benchmark,
        n_samples=n_samples,
        max_reflections=max_reflections,
        depth_limit=depth_limit,
        max_unique=max_unique,
        cache_value=cache_value
    )

    run = wandb.init(
        project=benchmark, 
        entity="agential",
        config={
            "model": model,
            "eval_model": eval_model,
            "seed": seed,
            "n_samples": n_samples,
            "depth_limit": depth_limit,
            "max_unique": max_unique,
            "cache_value": cache_value,
            "max_reflections": max_reflections,
            "max_iterations": max_iterations,
        },
        group=method_name,
        tags=[f"method={method_name}", 
              f"model={model}",
            f"eval_model={eval_model}",
            f"seed={seed}",
            f"n_samples={n_samples}",
            f"depth_limit={depth_limit}",
            f"max_unique={max_unique}",
            f"cache_value={cache_value}",
            f"max_reflections={max_reflections}",
            f"max_iterations={max_iterations}"],
    )

    eval_table_data = []
    perf_table_data = []
    em_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    outputs = []

    for instance in data:
        question = instance["question"]
        answer = instance["answer"]

        # Inference.
        out = method.generate(
            question=question,
            key=answer,
            max_iterations=max_iterations,

        )

        # Calculate metrics.
        is_correct = int(EM(out.answer, answer, llm_as_judge=True, llm=eval_llm))
        precision_score = precision(out.answer, answer)
        recall_score = recall(out.answer, answer)
        f1_score = f1(out.answer, answer)

        # Update scores.
        em_scores.append(is_correct)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f1_scores.append(f1_score)

        # Update tables.
        eval_table_data.append([question, answer, out.answer, is_correct, precision_score, recall_score, f1_score])
        perf_table_data.append([
            out.total_prompt_tokens, 
            out.total_completion_tokens, 
            out.total_tokens, 
            out.total_prompt_cost,
            out.total_completion_cost,
            out.total_cost,
            out.total_prompt_time,
            out.total_time
        ])

        # Update outputs.
        outputs.append(out)

        # Log metrics.
        run.log({
            "em": is_correct,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        })

    total_em = sum(em_scores) / len(em_scores)
    total_precision = sum(precision_scores) / len(precision_scores)
    total_recall = sum(recall_scores) / len(recall_scores)
    total_f1 = sum(f1_scores) / len(f1_scores)

    eval_table = wandb.Table(data=eval_table_data, columns=["question", "answer", "predicted_answer", "EM", "precision", "recall", "f1"])
    perf_columns = ["total_prompt_tokens", "total_completion_tokens", "total_tokens", "total_prompt_cost", "total_completion_cost", "total_cost", "total_prompt_time", "total_time"]
    perf_table = wandb.Table(data=perf_table_data, columns=perf_columns)

    outputs_save_path = os.path.join(output_path, f"{run.name}.pkl")
    with open(outputs_save_path, 'wb') as f:
        pickle.dump(outputs, f)

    artifact = wandb.Artifact(name=run.name, type="output")
    artifact.add_file(local_path=outputs_save_path, name="outputs.pkl")
    artifact.save()

    run.log({
        f"{run.name}_eval": eval_table,
        f"{run.name}_perf": perf_table
    })

    column_averages = np.mean(np.array(perf_table_data, dtype=float), axis=0).tolist()
    run.log({
        "total_em": total_em,
        "total_precision": total_precision,
        "total_recall": total_recall,
        "total_f1": total_f1,
        **dict(zip(perf_columns, column_averages))
    })

    run.finish()
