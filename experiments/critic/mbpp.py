"""Run Critic on MBPP."""

import numpy as np

from agential.eval.metrics.classification import EM
import os
import pickle
import warnings

from agential.agents.critic.agent import Critic
from agential.utils.general import safe_execute

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from agential.core.llm import LLM

from experiments.utils import set_seed

import wandb

wandb.login()
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description="Run Critic experiments.")
parser.add_argument(
    "--n_eval_samples", type=int, default=-1, help="Number of samples to evaluate"
)
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model")
parser.add_argument(
    "--eval_model", type=str, default="gpt-4o", help="The evaluator model"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--max_interactions", type=int, default=7, help="Maximum number of interactions"
)
parser.add_argument("--use_tool", type=bool, default=True, help="Whether to use tool")
args = parser.parse_args()

set_seed(args.seed)
root_dir = "output"
method_name = "critic"
benchmark = "mbpp"

if __name__ == "__main__":
    data = load_dataset("Sing0402/mbpp")["train"]

    n_eval_samples = args.n_eval_samples
    model = args.model
    eval_model = args.eval_model
    seed = args.seed
    max_interactions = args.max_interactions
    use_tool = args.use_tool

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

    method = Critic(
        llm=llm,
        benchmark=benchmark,
    )

    run = wandb.init(
        project=benchmark,
        entity="agential",
        config={
            "n_eval_samples": n_eval_samples,
            "model": model,
            "eval_model": eval_model,
            "seed": seed,
            "max_interactions": max_interactions,
            "use_tool": use_tool,
        },
        group=method_name,
        tags=[
            f"n_eval_samples={n_eval_samples}",
            f"method={method_name}",
            f"model={model}",
            f"eval_model={eval_model}",
            f"seed={seed}",
            f"max_interactions={max_interactions}",
            f"use_tool={use_tool}",
        ],
    )

    eval_table_data = []
    perf_table_data = []
    em_scores = []
    outputs = []

    for idx, instance in enumerate(data):
        if n_eval_samples != -1 and idx >= n_eval_samples:
            break

        question = instance["prompt"]
        answer: str = "\n".join(
            instance["test_imports"] + [""] + instance["test_list"]
        ).strip()

        # Inference.
        out = method.generate(
            question=question,
            max_interactions=max_interactions,
            use_tool=use_tool,
            additional_keys={"tests": answer},
            critique_additional_keys={"tests": answer},
        )

        code_str = out.answer.replace("```python", "").replace("```", "").strip()
        _, execution_status = safe_execute(
            f"from typing import *\n\n{code_str}\n{answer}"
        )

        is_correct = int(EM(execution_status, "Done", normalize=False))

        # Update scores.
        em_scores.append(is_correct)

        # Update tables.
        eval_table_data.append(
            [question, answer, execution_status, out.answer, is_correct]
        )
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
                "pass@k=1": is_correct,
            }
        )

    # Calculate total scores.
    total_em = sum(em_scores) / len(em_scores)

    # Create tables.
    eval_table = wandb.Table(
        data=eval_table_data,
        columns=["question", "answer", "code_answer", "predicted_answer", "pass@k=1"],
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

    # Save outputs as artifact.
    artifact = wandb.Artifact(name=run.name, type="output")
    artifact.add_file(local_path=outputs_save_path, name="outputs.pkl")
    artifact.save()

    # Log tables.
    run.log({f"{run.name}_eval": eval_table, f"{run.name}_perf": perf_table})

    # Log all metrics.
    column_averages = np.mean(np.array(perf_table_data, dtype=float), axis=0).tolist()
    column_sums = np.sum(np.array(perf_table_data, dtype=float), axis=0).tolist()
    run.log(
        {
            "total_pass@k=1": total_em,
            **dict(zip([f"avg_{col}" for col in perf_columns], column_averages)),
            **dict(zip([f"sum_{col}" for col in perf_columns], column_sums)),
        }
    )

    run.finish()
