# Training
python hotpotqa_train.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "adapt" \
    --patience 3

python fever_train.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "adapt" \
    --patience 3

python ambignq_train.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "adapt" \
    --patience 3

python triviaqa_train.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "adapt" \
    --patience 3

python gsm8k_train.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "adapt" \
    --patience 3

python mbpp_train.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "adapt" \
    --patience 3

hotpotqa_train_run = ""
fever_train_run = ""
ambignq_train_run = ""
triviaqa_train_run = ""
gsm8k_train_run = ""
mbpp_train_run = ""

# Base runs
python hotpotqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/hotpotqa/${hotpotqa_train_run}-clin-memories.pkl"

python fever.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/fever/${fever_train_run}-clin-memories.pkl"

python ambignq.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/ambignq/${ambignq_train_run}-clin-memories.pkl"

python triviaqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/triviaqa/${triviaqa_train_run}-clin-memories.pkl"

python gsm8k.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/gsm8k/${gsm8k_train_run}-clin-memories.pkl"

python svamp.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/gsm8k/${gsm8k_train_run}-clin-memories.pkl"

python tabmwp.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/gsm8k/${gsm8k_train_run}-clin-memories.pkl"

python humaneval.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/mbpp/${mbpp_train_run}-clin-memories.pkl"

python mbpp.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o-mini" \
    --seed 42 \
    --max_trials 3 \
    --max_steps 6 \
    --max_tokens 5000 \
    --k 10 \
    --quadrant "gen_task" \
    --patience 3 \
    --memory_path "output/mbpp/${mbpp_train_run}-clin-memories.pkl"


###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_benchmark = "hotpotqa"
memory_path = ""

# Sweep max_trials (3,5,7)
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_trials 5 --max_steps 6 --max_tokens 5000 --k 10 --quadrant "gen_task" --patience 3 --memory_path ${memory_path}
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_trials 7 --max_steps 6 --max_tokens 5000 --k 10 --quadrant "gen_task" --patience 3 --memory_path ${memory_path}

# Sweep max_steps (4,6,8)
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_trials 3 --max_steps 4 --max_tokens 5000 --k 10 --quadrant "gen_task" --patience 3 --memory_path ${memory_path}
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_trials 3 --max_steps 8 --max_tokens 5000 --k 10 --quadrant "gen_task" --patience 3 --memory_path ${memory_path}

# Sweep k (5,10,15)
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_trials 3 --max_steps 6 --max_tokens 5000 --k 5 --quadrant "gen_task" --patience 3 --memory_path ${memory_path}
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_trials 3 --max_steps 6 --max_tokens 5000 --k 15 --quadrant "gen_task" --patience 3 --memory_path ${memory_path}

# Sweep quadrant
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_trials 3 --max_steps 6 --max_tokens 5000 --k 10 --quadrant "gen_env" --patience 3 --memory_path ${memory_path}
