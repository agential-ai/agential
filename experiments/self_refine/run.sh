# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 3
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 3
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 3
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 3

###############################################################
# Hyperparameter Sweep
###############################################################

# fewshot_type=cot
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 3 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 3 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 3 --num_results 10
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 5 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 5 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 5 --num_results 10
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 7 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 7 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "cot" --max_interactions 7 --num_results 10

# fewshot_type=direct
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 3 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 3 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 3 --num_results 10
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 5 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 5 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 5 --num_results 10
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 7 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 7 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "direct" --max_interactions 7 --num_results 10

# fewshot_type=react
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 3 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 3 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 3 --num_results 10
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 5 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 5 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 5 --num_results 10
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 7 --num_results 5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 7 --num_results 8
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --fewshot_type "react" --max_interactions 7 --num_results 10
