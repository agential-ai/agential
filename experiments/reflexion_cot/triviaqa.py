"""Run ReflexionCoT on TriviaQA."""
import numpy as np
from agential.eval.metrics.classification import EM, f1, fuzzy_EM, llm_as_judge_eval, precision, recall
import os
import pickle

import warnings

from agential.agents.reflexion.agent import ReflexionCoT
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from agential.core.llm import LLM

from experiments.utils import set_seed

import wandb
wandb.login()
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description="Run ReflexionCoT experiments.")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model")
parser.add_argument("--eval_model", type=str, default="gpt-4o", help="The evaluator model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_reflections", type=int, default=3, help="Max reflections")
parser.add_argument("--max_trials", type=int, default=3, help="Max trials")
parser.add_argument("--patience", type=int, default=3, help="Patience")
parser.add_argument("--reflect_strategy", type=str, default="reflexion", help="Reflection strategy")
args = parser.parse_args()

set_seed(args.seed)
root_dir = "output"
method_name = "reflexion_cot"
benchmark = "triviaqa"

if __name__ == '__main__':
    data = load_dataset("alckasoc/triviaqa_500")['train']

    model = args.model
    eval_model = args.eval_model
    seed = args.seed
    max_reflections = args.max_reflections
    max_trials = args.max_trials
    patience = args.patience
    reflect_strategy = args.reflect_strategy

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

    method = ReflexionCoT(
        llm=llm,
        benchmark=benchmark,
        max_reflections=max_reflections,
        max_trials=max_trials,
    )

    run = wandb.init(
        project=benchmark, 
        entity="agential",
        config={
            "model": model,
            "eval_model": eval_model,
            "seed": seed,
            "patience": patience,
            "max_reflections": max_reflections,
            "max_trials": max_trials,
            "reflect_strategy": reflect_strategy,
        },
        group=method_name,
        tags=[f"method={method_name}", f"model={model}", f"eval_model={eval_model}", f"seed={seed}", f"patience={patience}", f"max_reflections={max_reflections}", f"max_trials={max_trials}", f"reflect_strategy={reflect_strategy}"],
    )

    eval_table_data = []
    perf_table_data = []
    em_scores = []
    fuzzy_em_scores = []
    llm_judge_eval_scores = []
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
            reflect_strategy=reflect_strategy,
            patience=patience,
        )

        # Calculate metrics.
        is_correct = int(any([EM(out.answer, answer) for answer in answers]))
        is_correct_fuzzy = int(any([fuzzy_EM(out.answer, answer) for answer in answers]))
        llm_judge_eval_score = int(llm_as_judge_eval(llm=eval_llm, question=question, answer=out.answer, key=answers))
        precision_score = max([precision(out.answer, answer) for answer in answers])
        recall_score = max([recall(out.answer, answer) for answer in answers])
        f1_score = max([f1(out.answer, answer) for answer in answers])

        # Update scores.
        em_scores.append(is_correct)
        fuzzy_em_scores.append(is_correct_fuzzy)
        llm_judge_eval_scores.append(llm_judge_eval_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f1_scores.append(f1_score)

        # Update tables.
        eval_table_data.append([question, str(answers), out.answer, is_correct, is_correct_fuzzy, llm_judge_eval_score, precision_score, recall_score, f1_score])
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

        # Log metrics per instance.
        run.log({
            "em": is_correct,
            "fuzzy_em": is_correct_fuzzy,
            "llm_judge_eval": llm_judge_eval_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        })

    # Calculate total scores.
    total_em = sum(em_scores) / len(em_scores)
    total_em_fuzzy = sum(fuzzy_em_scores) / len(fuzzy_em_scores)
    total_llm_judge_eval = sum(llm_judge_eval_scores) / len(llm_judge_eval_scores)
    total_precision = sum(precision_scores) / len(precision_scores)
    total_recall = sum(recall_scores) / len(recall_scores)
    total_f1 = sum(f1_scores) / len(f1_scores)

    # Create tables.
    eval_table = wandb.Table(data=eval_table_data, columns=["question", "answer", "predicted_answer", "EM", "fuzzy_EM", "llm_judge_eval", "precision", "recall", "f1"])
    perf_columns = ["total_prompt_tokens", "total_completion_tokens", "total_tokens", "total_prompt_cost (USD)", "total_completion_cost (USD)", "total_cost (USD)", "total_prompt_time (s)", "total_time (s)"]
    perf_table = wandb.Table(data=perf_table_data, columns=perf_columns)

    # Save outputs as pkl.
    outputs_save_path = os.path.join(output_path, f"{run.name}.pkl")
    with open(outputs_save_path, 'wb') as f:
        pickle.dump(outputs, f)

    # Save outputs as artifact.
    artifact = wandb.Artifact(name=run.name, type="output")
    artifact.add_file(local_path=outputs_save_path, name="outputs.pkl")
    artifact.save()

    # Log tables.
    run.log({
        f"{run.name}_eval": eval_table,
        f"{run.name}_perf": perf_table
    })

    # Log all metrics.
    column_averages = np.mean(np.array(perf_table_data, dtype=float), axis=0).tolist()
    column_sums = np.sum(np.array(perf_table_data, dtype=float), axis=0).tolist()
    run.log({
        "total_em": total_em,
        "total_em_fuzzy": total_em_fuzzy,
        "total_llm_judge_eval": total_llm_judge_eval,
        "total_precision": total_precision,
        "total_recall": total_recall,
        "total_f1": total_f1,
        **dict(zip([f"avg_{col}" for col in perf_columns], column_averages)),
        **dict(zip([f"sum_{col}" for col in perf_columns], column_sums)),
    })

    run.finish()
