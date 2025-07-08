import argparse
import os
import pickle
import sys
import wandb
import ast
import json
from experiments.utils import set_seed
from experiments.experiment_utils import (
    METHOD_REGISTRY,
    BENCHMARK_REGISTRY,
    validate_method_benchmark_compatibility,
    get_benchmark_type,
    construct_complex_question,
    construct_complex_key,
    evaluate_answer,
    get_benchmark_data,
    get_agent_instance,
    generate_agent_response,
)

def parse_dict_arg(arg):
    if not arg:
        return {}
    try:
        return ast.literal_eval(arg)
    except Exception:
        try:
            return json.loads(arg)
        except Exception:
            raise ValueError(f"Could not parse argument as dict: {arg}")

def main():
    parser = argparse.ArgumentParser(description="Run a single method on a single benchmark.")
    parser.add_argument("--method", type=str, required=True, help="Method to run (required)")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark to run (required)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="The model")
    parser.add_argument("--eval_model", type=str, default="gpt-4.1-mini", help="The evaluator model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="agential", help="wandb project name")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--list_methods", action="store_true", help="List available methods")
    parser.add_argument("--list_benchmarks", action="store_true", help="List available benchmarks")
    parser.add_argument("--agent_hyperparams", type=str, default=None, help="Agent hyperparameters as a dict string or JSON")
    parser.add_argument("--generate_params", type=str, default=None, help="Generate parameters as a dict string or JSON")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the chunk to run (0-based)")
    parser.add_argument("--n_chunks", type=int, default=10, help="Number of chunks to split the data into (only used if --use_chunks)")
    parser.add_argument("--use_chunks", action="store_true", help="If set, split data into chunks and run only on chunk_idx")
    args = parser.parse_args()
    n_chunks = args.n_chunks
    chunk_idx = args.chunk_idx

    # Handle list commands
    if args.list_methods:
        print("Available methods:")
        for method, info in METHOD_REGISTRY.items():
            print(f"  {method}: {info['class']} from {info['module']}")
        return

    if args.list_benchmarks:
        print("Available benchmarks:")
        for benchmark, info in BENCHMARK_REGISTRY.items():
            print(f"  {benchmark} ({info['type']}): {info['description']}")
        return

    # Validate inputs
    method = args.method.strip()
    benchmark = args.benchmark.strip()
    if method not in METHOD_REGISTRY:
        print(f"Invalid method: {method}")
        print("Available methods:", list(METHOD_REGISTRY.keys()))
        print("Use --list_methods to see all available methods")
        sys.exit(1)
    if benchmark not in BENCHMARK_REGISTRY:
        print(f"Invalid benchmark: {benchmark}")
        print("Available benchmarks:", list(BENCHMARK_REGISTRY.keys()))
        print("Use --list_benchmarks to see all available benchmarks")
        sys.exit(1)
    if not validate_method_benchmark_compatibility(method, benchmark):
        print(f"Method {method} does not support benchmark {benchmark}")
        print(f"Supported benchmarks for {method}:", METHOD_REGISTRY[method]["supported_benchmarks"])
        sys.exit(1)

    # Parse agent hyperparams and generate params
    agent_hyperparams = parse_dict_arg(args.agent_hyperparams)
    generate_params = parse_dict_arg(args.generate_params)
    print(f"Agent hyperparams: {agent_hyperparams}")
    print(f"Generate params: {generate_params}")

    set_seed(args.seed)
    wandb.login()

    print(f"\n=== Running {method} on {benchmark} ===")
    benchmark_info = BENCHMARK_REGISTRY[benchmark]
    print(f"Benchmark type: {benchmark_info['type']}")
    print(f"Description: {benchmark_info['description']}")
    
    # Load data
    try:
        data = get_benchmark_data(benchmark)
        print(f"Loaded {len(data)} samples")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Split data into chunks if requested
    if args.use_chunks:
        if not (0 <= chunk_idx < n_chunks):
            print(f"chunk_idx must be in [0, {n_chunks-1}], got {chunk_idx}")
            sys.exit(1)
        chunk_size = (len(data) + n_chunks - 1) // n_chunks  # ceil division
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(data))
        data_chunk = data[start_idx:end_idx]
        print(f"Running chunk {chunk_idx+1}/{n_chunks}: examples {start_idx} to {end_idx-1} (total {len(data_chunk)})")
    else:
        data_chunk = data
        print(f"Running on entire dataset: {len(data_chunk)} examples")

    output_path = os.path.join(args.output_dir, method, benchmark)
    os.makedirs(output_path, exist_ok=True)

    # Create LLMs
    from agential.core.llm import LLM
    llm = LLM(
        args.model,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=args.seed,
    )

    # Evaluation LLM for LLM-as-judge metrics
    eval_llm = None
    if benchmark_info["requires_llm_judge"]:
        eval_llm = LLM(
            args.eval_model,
            temperature=0,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=args.seed,
        )

    # Create agent (pass agent_hyperparams)
    try:
        agent = get_agent_instance(method, benchmark, llm, agent_hyperparams)
    except Exception as e:
        print(f"Failed to create agent: {e}")
        sys.exit(1)

    run = wandb.init(
        project=args.wandb_project,
        entity="agential",
        config={
            "model": args.model,
            "eval_model": args.eval_model,
            "seed": args.seed,
            "method": method,
            "benchmark": benchmark,
            "benchmark_type": benchmark_info["type"],
        },
        group=method,
        tags=[
            f"method={method}",
            f"model={args.model}",
            f"eval_model={args.eval_model}",
            f"seed={args.seed}",
            f"benchmark_type={benchmark_info['type']}",
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

    for idx, instance in enumerate(data_chunk):
        print(f"\n=== Processing Example {idx+1}/{len(data_chunk)} ===")
        
        # Handle complex question and key construction
        if benchmark_info.get("complex_key", False):
            question = construct_complex_question(instance, benchmark)  # type: ignore
            key = construct_complex_key(instance, benchmark)  # type: ignore
        else:
            question = instance[benchmark_info["question_field"]]  # type: ignore
            key = instance[benchmark_info["key_field"]]  # type: ignore
        
        # Generate response
        import time
        print(f"Question: {question}")
        start_time = time.time()
        out = generate_agent_response(agent, method, benchmark, question, key, generate_params)
        elapsed = out["metrics"].get("total_time", time.time() - start_time)
        
        # Evaluate with comprehensive metrics
        eval_results = evaluate_answer(benchmark, out["answer"], key, eval_llm, question)
        
        # Update scores
        em_scores.append(eval_results["em"])
        fuzzy_em_scores.append(eval_results["fuzzy_em"])
        llm_judge_eval_scores.append(eval_results["llm_judge_eval"])
        precision_scores.append(eval_results["precision"])
        recall_scores.append(eval_results["recall"])
        f1_scores.append(eval_results["f1"])
        
        # Update tables
        eval_table_data.append([
            question,
            str(key),
            out["answer"],
            eval_results["em"],
            eval_results["fuzzy_em"],
            eval_results["llm_judge_eval"],
            eval_results["precision"],
            eval_results["recall"],
            eval_results["f1"],
        ])
        perf_table_data.append([
            out["metrics"]["total_tokens"],
            out["metrics"]["total_cost"],
            out["metrics"]["total_time"],
        ])
        outputs.append(out)
        
        # Log metrics per instance
        run.log({
            "em": eval_results["em"],
            "fuzzy_em": eval_results["fuzzy_em"],
            "llm_judge_eval": eval_results["llm_judge_eval"],
            "precision": eval_results["precision"],
            "recall": eval_results["recall"],
            "f1": eval_results["f1"],
        })
        
        status = "✓" if eval_results["em"] else "✗"
        # Print a neat vertical table for each example
        print(f"\n--- Example {idx+1} ---")
        print(f"Status     : {status}")
        print(f"EM         : {eval_results['em']}")
        print(f"Fuzzy EM   : {eval_results['fuzzy_em']}")
        if benchmark_info["requires_llm_judge"]:
            print(f"LLM Judge  : {eval_results['llm_judge_eval']}")
        print(f"Precision  : {eval_results['precision']:.3f}")
        print(f"Recall     : {eval_results['recall']:.3f}")
        print(f"F1         : {eval_results['f1']:.3f}")
        print(f"Elapsed    : {elapsed:.2f}s")
        print(f"Tokens     : {out['metrics']['total_tokens']}")
        print(f"Cost       : ${out['metrics']['total_cost']:.4f}")
        print(f"Latency    : {out['metrics']['total_time']:.2f}s")
        print(f"Question   : {question}")
        print(f"Predicted  : {out['answer']}")
        print(f"Label      : {key}")
        print("----------------------")

    # Calculate total scores
    total_em = sum(em_scores) / len(em_scores)
    total_em_fuzzy = sum(fuzzy_em_scores) / len(fuzzy_em_scores)
    total_llm_judge_eval = sum(llm_judge_eval_scores) / len(llm_judge_eval_scores)
    total_precision = sum(precision_scores) / len(precision_scores)
    total_recall = sum(recall_scores) / len(recall_scores)
    total_f1 = sum(f1_scores) / len(f1_scores)
    
    # Extract performance metrics
    tokens_list = [row[0] for row in perf_table_data]
    costs_list = [row[1] for row in perf_table_data]
    times_list = [row[2] for row in perf_table_data]
    
    # Calculate totals and averages
    total_tokens = sum(tokens_list)
    total_cost = sum(costs_list)
    total_time = sum(times_list)
    avg_tokens = total_tokens / len(perf_table_data)
    avg_cost = total_cost / len(perf_table_data)
    avg_time = total_time / len(perf_table_data)
    
    # Calculate additional statistics
    import numpy as np
    tokens_std = np.std(tokens_list)
    costs_std = np.std(costs_list)
    times_std = np.std(times_list)
    
    tokens_min = min(tokens_list)
    tokens_max = max(tokens_list)
    costs_min = min(costs_list)
    costs_max = max(costs_list)
    times_min = min(times_list)
    times_max = max(times_list)
    
    # Additional useful statistics
    tokens_median = np.median(tokens_list)
    costs_median = np.median(costs_list)
    times_median = np.median(times_list)
    
    # Percentiles for better distribution understanding
    tokens_p25 = np.percentile(tokens_list, 25)
    tokens_p75 = np.percentile(tokens_list, 75)
    tokens_p95 = np.percentile(tokens_list, 95)
    costs_p25 = np.percentile(costs_list, 25)
    costs_p75 = np.percentile(costs_list, 75)
    costs_p95 = np.percentile(costs_list, 95)
    times_p25 = np.percentile(times_list, 25)
    times_p75 = np.percentile(times_list, 75)
    times_p95 = np.percentile(times_list, 95)
    


    # Create tables
    eval_table = wandb.Table(
        data=eval_table_data,
        columns=[
            "question",
            "answer",
            "predicted_answer",
            "EM",
            "fuzzy_EM",
            "llm_judge_eval",
            "precision",
            "recall",
            "f1",
        ],
    )
    perf_columns = [
        "total_tokens",
        "total_cost (USD)",
        "total_time (s)",
    ]
    perf_table = wandb.Table(data=perf_table_data, columns=perf_columns)
    
    outputs_save_path = os.path.join(output_path, f"{run.name or 'run'}.pkl")
    with open(outputs_save_path, "wb") as f:
        pickle.dump(outputs, f)
    artifact = wandb.Artifact(name=run.name or 'run', type="output")
    artifact.add_file(local_path=outputs_save_path, name="outputs.pkl")
    artifact.save()
    
    # Log tables and metrics
    run.log({
        f"{run.name or 'run'}_eval": eval_table,
        f"{run.name or 'run'}_perf": perf_table,
        "total_em": total_em,
        "total_em_fuzzy": total_em_fuzzy,
        "total_llm_judge_eval": total_llm_judge_eval,
        "total_precision": total_precision,
        "total_recall": total_recall,
        "total_f1": total_f1,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "total_time": total_time,
        "avg_tokens": avg_tokens,
        "avg_cost": avg_cost,
        "avg_time": avg_time,
        "tokens_std": tokens_std,
        "costs_std": costs_std,
        "times_std": times_std,
        "tokens_min": tokens_min,
        "tokens_max": tokens_max,
        "costs_min": costs_min,
        "costs_max": costs_max,
        "times_min": times_min,
        "times_max": times_max,
        "tokens_median": tokens_median,
        "costs_median": costs_median,
        "times_median": times_median,
        "tokens_p25": tokens_p25,
        "tokens_p75": tokens_p75,
        "tokens_p95": tokens_p95,
        "costs_p25": costs_p25,
        "costs_p75": costs_p75,
        "costs_p95": costs_p95,
        "times_p25": times_p25,
        "times_p75": times_p75,
        "times_p95": times_p95,
        "total_tasks": len(perf_table_data),
        "accuracy": total_em,  # Use EM as primary accuracy metric
    })
    run.finish()

    # Print summary metrics
    print("\nExperiment Summary:")
    print("------------------")
    print(f"Method: {method}")
    print(f"Benchmark: {benchmark} ({benchmark_info['type']})")
    if args.use_chunks:
        print(f"Total tasks: {len(perf_table_data)} (chunk {chunk_idx+1}/{n_chunks})")
    else:
        print(f"Total tasks: {len(perf_table_data)} (entire dataset)")
    print(f"EM Accuracy: {total_em:.2%} ({sum(em_scores)}/{len(em_scores)})")
    print(f"Fuzzy EM Accuracy: {total_em_fuzzy:.2%} ({sum(fuzzy_em_scores)}/{len(fuzzy_em_scores)})")
    if benchmark_info["requires_llm_judge"]:
        print(f"LLM Judge Accuracy: {total_llm_judge_eval:.2%} ({sum(llm_judge_eval_scores)}/{len(llm_judge_eval_scores)})")
    print(f"Precision: {total_precision:.3f}")
    print(f"Recall: {total_recall:.3f}")
    print(f"F1 Score: {total_f1:.3f}")
    print(f"\nPerformance Metrics:")
    print(f"{'Metric':<12} {'Total':<12} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8} {'P25':<8} {'P75':<8} {'P95':<8}")
    print("-" * 100)
    print(f"{'Tokens':<12} {total_tokens:<12} {avg_tokens:<8.1f} {tokens_std:<8.1f} {tokens_min:<8} {tokens_max:<8} {tokens_median:<8.1f} {tokens_p25:<8.1f} {tokens_p75:<8.1f} {tokens_p95:<8.1f}")
    print(f"{'Cost ($)':<12} {total_cost:<12.6f} {avg_cost:<8.6f} {costs_std:<8.6f} {costs_min:<8.6f} {costs_max:<8.6f} {costs_median:<8.6f} {costs_p25:<8.6f} {costs_p75:<8.6f} {costs_p95:<8.6f}")
    print(f"{'Latency (s)':<12} {total_time:<12.2f} {avg_time:<8.2f} {times_std:<8.2f} {times_min:<8.2f} {times_max:<8.2f} {times_median:<8.2f} {times_p25:<8.2f} {times_p75:<8.2f} {times_p95:<8.2f}")

if __name__ == "__main__":
    main() 