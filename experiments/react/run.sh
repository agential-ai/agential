# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python gsm8k.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python svamp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python tabmwp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python humaneval.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python mbpp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000

###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_benchmark = "hotpotqa"

# max_steps=6
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 7500
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 10000

# max_steps=4
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 4 --max_tokens 5000
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 4 --max_tokens 7500
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 4 --max_tokens 10000

# max_steps=8
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 8 --max_tokens 5000
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 8 --max_tokens 7500
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 8 --max_tokens 10000

# max_steps=10
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 10 --max_tokens 5000
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 10 --max_tokens 7500
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 10 --max_tokens 10000
