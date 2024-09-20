"""Run CoT on TriviaQA."""
from agential.eval.metrics.classification import EM, f1, precision, recall
import os
import json
import pickle

import warnings

from agential.prompting.cot.prompting import CoT
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from agential.core.llm import LLM

from experiments.utils import set_seed

import wandb
wandb.login()
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description="Run CoT experiments.")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_retries", type=int, default=1, help="Number of retries")
parser.add_argument("--warming", type=float, nargs='+', default=[0.0], help="Warming values")
args = parser.parse_args()

set_seed(args.seed)
root_dir = "output"
method_name = "cot"
benchmark = "triviaqa"

if __name__ == '__main__':
    data = load_dataset("alckasoc/triviaqa_500")['train']

    model = args.model
    seed = args.seed
    num_retries = args.num_retries
    warming = args.warming

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

    method = CoT(
        llm=llm,
        benchmark=benchmark,
    )

    run = wandb.init(
        project=benchmark, 
        entity="agential",
        config={
            "model": model,
            "seed": seed,
            "num_retries": num_retries,
            "warming": warming,
        },
        group=method_name,
        tags=[f"method={method_name}", f"model={model}", f"seed={seed}", f"num_retries={num_retries}", f"warming={warming}", "base"],
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
        answers = list(set(instance["answer"]['normalized_aliases']))

        # Inference.
        out = method.generate(
            question=question,
            key=instance['answer']['normalized_value'],
            num_retries=num_retries,
            warming=warming
        )

        # Calculate metrics.
        is_correct = int(any([EM(out.answer, answer) for answer in answers]))
        precision_score = max([precision(out.answer, answer) for answer in answers])
        recall_score = max([recall(out.answer, answer) for answer in answers])
        f1_score = max([f1(out.answer, answer) for answer in answers])

        # Update scores.
        em_scores.append(is_correct)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f1_scores.append(f1_score)

        # Update tables.
        eval_table_data.append([question, str(answers), out.answer, is_correct, precision_score, recall_score, f1_score])
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
    perf_table = wandb.Table(data=perf_table_data, columns=["total_prompt_tokens", "total_completion_tokens", "total_tokens", "total_prompt_cost", "total_completion_cost", "total_cost", "total_prompt_time", "total_time"])

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

    run.log({
        "total_em": total_em,
        "total_precision": total_precision,
        "total_recall": total_recall,
        "total_f1": total_f1,
    })

    run.finish()
