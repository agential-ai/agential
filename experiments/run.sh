#!/bin/bash

# Usage: ./run_all_methods.sh [extra args]
# Example: ./run_all_methods.sh --n_eval_samples 5 --model gpt-4o

METHODS=(cot)
BENCHMARKS=(fever)
# METHODS=(react self_refine cot clin reflexion expel lats critic)
# BENCHMARKS=(hotpotqa fever ambignq triviaqa gsm8k svamp tabmwp humaneval mbpp)

for BENCHMARK in "${BENCHMARKS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    echo "\n==============================="
    echo "Running $METHOD on $BENCHMARK"
    echo "===============================\n"
    python run_all.py \
      --method "$METHOD" \
      --benchmark "$BENCHMARK" \
      --wandb_project "$BENCHMARK" \
      --agent_hyperparams "{'max_interactions': 1, 'patience': 1, 'truncate_length': -1, 'verbose': False}" \
      --generate_params "{}" \
      "$@"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
      echo "Error running $METHOD on $BENCHMARK. Exiting."
      exit $STATUS
    fi
  done
done

echo "\nAll methods completed for all benchmarks: ${BENCHMARKS[*]}" 