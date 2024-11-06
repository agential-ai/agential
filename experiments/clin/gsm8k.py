"""Run CLIN on GSM8K."""

import numpy as np
import tiktoken
from agential.agents.clin.agent import CLIN
from agential.agents.clin.memory import CLINMemory
from agential.agents.clin.prompts import CLIN_ADAPT_META_SUMMARY_SYSTEM, CLIN_ADAPT_SUMMARY_SYSTEM, CLIN_INSTRUCTION_GSM8K, CLIN_META_SUMMARY_INSTRUCTION_GSM8K, CLIN_SUMMARY_INSTRUCTION_GSM8K
from agential.core.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_REACT
from agential.utils.docstore import DocstoreExplorer
from agential.eval.metrics.classification import EM
import os
import pickle
import warnings
from langchain_community.docstore.wikipedia import Wikipedia
from agential.utils.general import safe_execute
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from agential.core.llm import LLM

from experiments.utils import set_seed

import wandb
wandb.login()
from datasets import load_dataset

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
benchmark = "gsm8k"

if __name__ == '__main__':
    data = load_dataset("openai/gsm8k", "main")['test']

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
        enc = tiktoken.encoding_for_model(args.model)
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
            "n_eval_samples": n_eval_samples,
            "model": model,
            "eval_model": eval_model,
            "seed": seed,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "max_trials": max_trials,
        },
        group=method_name,
        tags=[
            f"n_eval_samples={n_eval_samples}",
            f"method={method_name}", 
            f"model={model}", 
            f"eval_model={eval_model}", 
            f"seed={seed}", 
            f"max_trials={max_trials}",
            f"max_steps={max_steps}", 
            f"max_tokens={max_tokens}",
            f"memory_k={k}"
        ],
    )

    eval_table_data = []
    perf_table_data = []
    em_scores = []
    outputs = []

    for idx, instance in enumerate(data):
        if n_eval_samples != -1 and idx >= n_eval_samples:
            break
        
        question = instance["question"]
        answer: str = instance["answer"]
        answer = str(float(answer.split("#### ")[-1].strip().replace(",", "")))

        # Inference.
        out = method.generate(
            question=question,
            key=answer,
            examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
            prompt=CLIN_INSTRUCTION_GSM8K,
            summary_prompt=CLIN_SUMMARY_INSTRUCTION_GSM8K,
            meta_summary_prompt=CLIN_META_SUMMARY_INSTRUCTION_GSM8K,
            additional_keys={},
            summary_additional_keys={},
            meta_summary_additional_keys={},
            summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
            meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
            quadrant=quadrant,
            patience=patience
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
        })

    # Calculate total scores.
    total_em = sum(em_scores) / len(em_scores)

    # Create tables.
    eval_table = wandb.Table(data=eval_table_data, columns=["question", "answer", "code_answer", "predicted_answer", "EM"])
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
        **dict(zip([f"avg_{col}" for col in perf_columns], column_averages)),
        **dict(zip([f"sum_{col}" for col in perf_columns], column_sums)),
    })
    
    run.finish()