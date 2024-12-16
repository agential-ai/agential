# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 3
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 3
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 3
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 3

###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_qa_benchmark = "hotpotqa"

# fewshot_type=cot
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 7
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 7
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "cot" --max_interactions 7

# fewshot_type=direct
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 7
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 7
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "direct" --max_interactions 7

# fewshot_type=react
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 3
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 5
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 7
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 7
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --fewshot_type "react" --max_interactions 7

hparam_math_code_benchmark = "gsm8k"

# fewshot_type=cot
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7

# fewshot_type=direct
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7

# fewshot_type=react
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3 
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3 
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 3 
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 
