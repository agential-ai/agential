# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"

###############################################################
# Hyperparameter Sweep
###############################################################

# max_reflections=2
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 1 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 1 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 1 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 5 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 2 --max_trials 5 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

# max_reflections=3
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 1 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

# max_reflections=5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 1 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 1 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 1 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 5 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_reflections 5 --max_trials 5 --patience 3 --reflect_strategy "last_attempt_and_reflexion"
