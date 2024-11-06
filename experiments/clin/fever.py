"""Run FEVER on CLIN."""
from agential.agents.clin.agent import CLIN
from agential.agents.clin.memory import CLINMemory
from agential.agents.clin.prompts import CLIN_INSTRUCTION_FEVER, CLIN_META_SUMMARY_INSTRUCTION_FEVER, CLIN_SUMMARY_INSTRUCTION_FEVER
from agential.core.fewshots.fever import FEVER_FEWSHOT_EXAMPLES_REACT
from agential.utils.docstore import DocstoreExplorer
import numpy as np
from agential.eval.metrics.classification import EM, f1, fuzzy_EM, llm_as_judge_eval, precision, recall
import os
import json
import pickle

import tiktoken
import warnings
from langchain_community.docstore.wikipedia import Wikipedia
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from agential.core.llm import LLM

from experiments.utils import set_seed

import wandb
wandb.login()

import argparse

parser = argparse.ArgumentParser(description="Run CLIN experiments.")
parser.add_argument("--n_eval_samples", type=int, default=-1, help="Number of samples to evaluate")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model")
parser.add_argument("--eval_model", type=str, default="gpt-4o", help="The evaluator model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_trials", type=int, default=3, help="Maximum number of trails")
parser.add_argument("--max_steps", type=int, default=6, help="Maximum number of steps")
parser.add_argument("--max_tokens", type=int, default=5000, help="Maximum number of tokens")
parser.add_argument("--k", type=int, default=10, help="Number of meta-summaries to use.")
parser.add_argument("--quadrant", type=str, default="adapt", help="Type of summary to use.")
parser.add_argument("--patience", type=int, default=3, help="Number of trials before early stopping")
parser.add_argument("--memory_path", type=str, default="", help="Memory path (pkl)")
args = parser.parse_args()

set_seed(args.seed)
root_dir = "output"
method_name = "clin"
benchmark = "fever"

if __name__ == '__main__':
    with open("../../data/fever/paper_dev_s42_sample500.json", 'r') as f:
        data = json.load(f)

    n_eval_samples = args.n_eval_samples
    model = args.model
    eval_model = args.eval_model
    seed = args.seed
    max_trials = args.max_trials
    max_steps = args.max_steps
    max_tokens = args.max_tokens
    k = args.k
    quadrant = args.quadrant
    patience = args.patience
    memory_path = args.memory_path

    if memory_path:
        with open(memory_path, 'rb') as f:
            memory = pickle.load(f)
    else:
        memory = {}

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
        enc = tiktoken.encoding_for_model(model)
    except:
        enc = tiktoken.get_encoding("gpt-3.5-turbo")

    method = CLIN(
        llm=llm,
        benchmark=benchmark,
        memory=CLINMemory(
            k=k,
            **memory,
        ),
        # kwargs.
        max_trials=max_trials,
        max_steps=max_steps,
        max_tokens=max_tokens,
        enc=enc,
        docstore=DocstoreExplorer(Wikipedia()),
    )

    run = wandb.init(
        project=benchmark, 
        entity="agential",
        config={
            "is_training": False,
            "n_eval_samples": n_eval_samples,
            "model": model,
            "eval_model": eval_model,
            "seed": seed,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "max_trials": max_trials,
            "k": k,
            "quadrant": quadrant,
            "patience": patience
        },
        group=method_name,
        tags=[
            f"is_training=False",
            f"n_eval_samples={n_eval_samples}",
            f"method={method_name}", 
            f"model={model}", 
            f"eval_model={eval_model}", 
            f"seed={seed}", 
            f"max_trials={max_trials}",
            f"max_steps={max_steps}", 
            f"max_tokens={max_tokens}",
            f"k={k}",
            f"quadrant={quadrant}",
            f"patience={patience}"
        ],
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

    for idx, instance in enumerate(data):
        if n_eval_samples != -1 and idx >= n_eval_samples:
            break
        
        question = instance["claim"]
        answer = instance["label"]

        # Inference.
        out = method.generate(
            question=question,
            key=answer,
            examples=FEVER_FEWSHOT_EXAMPLES_REACT,
            prompt=CLIN_INSTRUCTION_FEVER,
            summary_prompt=CLIN_SUMMARY_INSTRUCTION_FEVER,
            meta_summary_prompt=CLIN_META_SUMMARY_INSTRUCTION_FEVER,
            additional_keys={},
            summary_additional_keys={},
            meta_summary_additional_keys={},
            quadrant=quadrant,
            patience=patience,
        )

        # Calculate metrics.
        is_correct = int(EM(out.answer, answer))
        is_correct_fuzzy = int(fuzzy_EM(out.answer, answer))
        llm_judge_eval_score = int(llm_as_judge_eval(llm=eval_llm, question=question, answer=out.answer, key=answer))
        precision_score = precision(out.answer, answer)
        recall_score = recall(out.answer, answer)
        f1_score = f1(out.answer, answer)

        # Update scores.
        em_scores.append(is_correct)
        fuzzy_em_scores.append(is_correct_fuzzy)
        llm_judge_eval_scores.append(llm_judge_eval_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f1_scores.append(f1_score)

        # Update tables.
        eval_table_data.append([question, answer, out.answer, is_correct, is_correct_fuzzy, llm_judge_eval_score, precision_score, recall_score, f1_score])
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
            "fuzzy_em": is_correct_fuzzy,
            "llm_judge_eval": llm_judge_eval_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        })

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

    # Save CLIN memory as pkl.
    clin_memories_save_path = os.path.join(output_path, f"{run.name}-clin-memories.pkl")
    with open(clin_memories_save_path, 'wb') as f:
        pickle.dump(method.strategy.memory.show_memories(), f)

    # Save outputs as pkl.
    outputs_save_path = os.path.join(output_path, f"{run.name}.pkl")
    with open(outputs_save_path, 'wb') as f:
        pickle.dump(outputs, f)

    # Save CLIN memory for ease-of-use.
    artifact = wandb.Artifact(name=run.name, type="output")
    artifact.add_file(local_path=clin_memories_save_path, name="clin-memories.pkl")

    # Save outputs as artifact.
    artifact.add_file(local_path=outputs_save_path, name="outputs.pkl")
    artifact.save()

    run.log({
        f"{run.name}_eval": eval_table,
        f"{run.name}_perf": perf_table
    })

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
