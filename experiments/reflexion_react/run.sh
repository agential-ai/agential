# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6
python gsm8k.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6
python svamp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6
python tabmwp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6
python humaneval.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6
python mbpp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6

###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_benchmark = "hotpotqa"

# max_trials=1, max_steps=4
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 4 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 4 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 4 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

# max_trials=1, max_steps=6
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 6 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 6 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 6 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

# max_trials=1, max_steps=8
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 8 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 8 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --max_steps 8 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

# max_trials=3, max_steps=4
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 4 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 4 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 4 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

# max_trials=3, max_steps=6
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

# max_trials=3, max_steps=8
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 8 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 8 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 8 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

# max_trials=5, max_steps=4
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 4 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 4 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 4 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

# max_trials=5, max_steps=6
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 6 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 6 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 6 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

# max_trials=5, max_steps=8
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 8 --reflect_strategy "reflexion" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 8 --reflect_strategy "last_attempt" --max_reflections 3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --max_steps 8 --reflect_strategy "last_attempt_and_reflexion" --max_reflections 3

###############################################################
# Sweep over max_reflections
###############################################################

# max_reflections=2
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 3 --reflect_strategy "reflexion" --max_steps 6

# max_reflections=3
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --reflect_strategy "reflexion" --max_steps 6

# max_reflections=4
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 4 --max_trials 3 --reflect_strategy "reflexion" --max_steps 6

# max_reflections=5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 3 --reflect_strategy "reflexion" --max_steps 6