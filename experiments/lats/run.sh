# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --n_samples 5 --max_reflections 4 --depth_limit 7 --max_unique 5 --cache_value True --max_iterations 5
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --n_samples 5 --max_reflections 4 --depth_limit 7 --max_unique 5 --cache_value True --max_iterations 5
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --n_samples 5 --max_reflections 4 --depth_limit 7 --max_unique 5 --cache_value True --max_iterations 5
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --n_samples 5 --max_reflections 4 --depth_limit 7 --max_unique 5 --cache_value True --max_iterations 5

###############################################################
# Hyperparameter Sweep
###############################################################
