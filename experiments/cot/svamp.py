"""Run CoT on SVAMP."""

import numpy as np
from agential.eval.metrics.classification import EM
import os
import pickle
import warnings

from agential.prompting.cot.prompting import CoT
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

parser = argparse.ArgumentParser(description="Run CoT experiments.")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model")
parser.add_argument("--eval_model", type=str, default="gpt-4o", help="The evaluator model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_retries", type=int, default=1, help="Number of retries")
parser.add_argument("--warming", type=float, nargs='+', default=[0.0], help="Warming values")
args = parser.parse_args()

set_seed(args.seed)
root_dir = "output"
method_name = "cot"
benchmark = "svamp"

if __name__ == '__main__':
    
    data = load_dataset("tongyx361/svamp")['test']

    model = args.model
    eval_model = args.eval_model
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

    eval_llm = LLM(
        eval_model,
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
            "eval_model": eval_model,
            "seed": seed,
            "num_retries": num_retries,
            "warming": warming,
        },
        group=method_name,
        tags=[f"method={method_name}", f"model={model}", f"eval_model={eval_model}", f"seed={seed}", f"num_retries={num_retries}", f"warming={warming}"],
    )



    eval_table_data = []
    perf_table_data = []
    em_scores = []
    outputs = []

    for instance in data:
        question = instance['Body'] + " " + instance['Question']
        answer = str(float(instance["Answer"]))

         # Inference.
        out = method.generate(
            question=question,
            key=answer,
            num_retries=num_retries,
            warming=warming
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
