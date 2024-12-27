# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3
python gsm8k.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3
python svamp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3
python tabmwp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3
python humaneval.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3
python mbpp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 3 --max_reflections 3 --depth_limit 5 --max_unique 4 --cache_values True --max_iterations 3

###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_benchmark = "hotpotqa"

# Combination sweep n_samples=[5,7,9,11] and depth_limit=[7,8,9]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 7 --max_reflections 4 --depth_limit 7 --max_unique 5 --cache_values True --max_iterations 5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 9 --max_reflections 4 --depth_limit 7 --max_unique 5 --cache_values True --max_iterations 5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 11 --max_reflections 4 --depth_limit 7 --max_unique 5 --cache_values True --max_iterations 5

python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 7 --max_reflections 4 --depth_limit 8 --max_unique 5 --cache_values True --max_iterations 5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 9 --max_reflections 4 --depth_limit 8 --max_unique 5 --cache_values True --max_iterations 5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 11 --max_reflections 4 --depth_limit 8 --max_unique 5 --cache_values True --max_iterations 5

python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 7 --max_reflections 4 --depth_limit 9 --max_unique 5 --cache_values True --max_iterations 5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 9 --max_reflections 4 --depth_limit 9 --max_unique 5 --cache_values True --max_iterations 5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 11 --max_reflections 4 --depth_limit 9 --max_unique 5 --cache_values True --max_iterations 5

# max_reflections=[5,6]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 5 --max_reflections 5 --depth_limit 7 --max_unique 5 --cache_values True --max_iterations 5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 5 --max_reflections 6 --depth_limit 7 --max_unique 5 --cache_values True --max_iterations 5

# max_unique=[5,6,7]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 5 --max_reflections 6 --depth_limit 7 --max_unique 6 --cache_values True --max_iterations 5
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 5 --max_reflections 6 --depth_limit 7 --max_unique 7 --cache_values True --max_iterations 5

# max_iterations=[6,7]
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 5 --max_reflections 6 --depth_limit 7 --max_unique 5 --cache_values True --max_iterations 6
python ${hparam_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --n_samples 5 --max_reflections 6 --depth_limit 7 --max_unique 5 --cache_values True --max_iterations 7



