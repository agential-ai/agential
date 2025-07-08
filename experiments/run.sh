#!/bin/bash

# Usage: ./run_all_methods.sh [extra args]
# Example: ./run_all_methods.sh --n_eval_samples 5 --model gpt-4o

python run_all.py \
    --method lats \
    --benchmark hotpotqa \
    --wandb_project hotpotqa \
    --agent_hyperparams "{'n_samples': 3, 'max_reflections': 3, 'depth_limit': 5, 'max_unique': 4, 'verbose': False}" \
    --generate_params "{}" \

python run_all.py \
    --method reflexion \
    --benchmark fever \
    --wandb_project fever \
    --agent_hyperparams "{'verbose': False}" \
    --generate_params "{}" \

python run_all.py \
    --method lats \
    --benchmark ambignq \
    --wandb_project ambignq \
    --agent_hyperparams "{'n_samples': 3, 'max_reflections': 3, 'depth_limit': 5, 'max_unique': 4, 'verbose': False}" \
    --generate_params "{}" \

python run_all.py \
    --method lats \
    --benchmark triviaqa \
    --wandb_project triviaqa \
    --agent_hyperparams "{'n_samples': 3, 'max_reflections': 3, 'depth_limit': 5, 'max_unique': 4, 'verbose': False}" \
    --generate_params "{}" \

python run_all.py \
    --method lats \
    --benchmark gsm8k \
    --wandb_project gsm8k \
    --agent_hyperparams "{'n_samples': 3, 'max_reflections': 3, 'depth_limit': 5, 'max_unique': 4, 'verbose': False}" \
    --generate_params "{}" \

python run_all.py \
    --method lats \
    --benchmark svamp \
    --wandb_project svamp \
    --agent_hyperparams "{'n_samples': 3, 'max_reflections': 3, 'depth_limit': 5, 'max_unique': 4, 'verbose': False}" \
    --generate_params "{}" \

python run_all.py \
    --method lats \
    --benchmark tabmwp \
    --wandb_project tabmwp \
    --agent_hyperparams "{'n_samples': 3, 'max_reflections': 3, 'depth_limit': 5, 'max_unique': 4, 'verbose': False}" \
    --generate_params "{}" \

python run_all.py \
    --method lats \
    --benchmark humaneval \
    --wandb_project humaneval \
    --agent_hyperparams "{'n_samples': 3, 'max_reflections': 3, 'depth_limit': 5, 'max_unique': 4, 'verbose': False}" \
    --generate_params "{}" \

python run_all.py \
    --method lats \
    --benchmark mbpp \
    --wandb_project mbpp \
    --agent_hyperparams "{'n_samples': 3, 'max_reflections': 3, 'depth_limit': 5, 'max_unique': 4, 'verbose': False}" \
    --generate_params "{}" \
