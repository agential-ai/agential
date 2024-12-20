# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python gsm8k.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python svamp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python tabmwp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python humaneval.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python mbpp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"

###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_benchmark = "hotpotqa"

# max_reflections=2
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 1 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 1 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 1 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 5 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 5 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

# max_reflections=3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

# max_reflections=5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 1 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 1 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 1 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 5 --patience 3 --reflect_strategy "last_attempt"
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 5 --patience 3 --reflect_strategy "last_attempt_and_reflexion"
