# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 5000

###############################################################
# Hyperparameter Sweep
###############################################################

# max_steps=6
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 7500
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 6 --max_tokens 10000

# max_steps=4
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 4 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 4 --max_tokens 7500
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 4 --max_tokens 10000

# max_steps=8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 8 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 8 --max_tokens 7500
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 8 --max_tokens 10000

# max_steps=10
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 10 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 10 --max_tokens 7500
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_steps 10 --max_tokens 10000
