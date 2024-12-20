# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0
python gsm8k.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0
python svamp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0
python tabmwp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0
python humaneval.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0
python mbpp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0

###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_benchmark = "hotpotqa"

# num_retries=1, warming=[0.0, 0.5]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0 0.5

# num_retries=1, warming=[0.0, 0.3, 0.7, 1.0]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 1 --warming 0.0 0.3 0.7 1.0

# num_retries=2, warming=[0.0]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 2 --warming 0.0

# num_retries=2, warming=[0.0, 0.5]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 2 --warming 0.0 0.5

# num_retries=2, warming=[0.0, 0.3, 0.7, 1.0]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 2 --warming 0.0 0.3 0.7 1.0

# num_retries=3, warming=[0.0]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 3 --warming 0.0

# num_retries=3, warming=[0.0, 0.5]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 3 --warming 0.0 0.5

# num_retries=3, warming=[0.0, 0.3, 0.7, 1.0]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 3 --warming 0.0 0.3 0.7 1.0

# num_retries=4, warming=[0.0]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 4 --warming 0.0

# num_retries=4, warming=[0.0, 0.5]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 4 --warming 0.0 0.5

# num_retries=4, warming=[0.0, 0.3, 0.7, 1.0]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 4 --warming 0.0 0.3 0.7 1.0

# num_retries=5, warming=[0.0]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 5 --warming 0.0

# num_retries=5, warming=[0.0, 0.5]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 5 --warming 0.0 0.5

# num_retries=5, warming=[0.0, 0.3, 0.7, 1.0]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --num_retries 5 --warming 0.0 0.3 0.7 1.0