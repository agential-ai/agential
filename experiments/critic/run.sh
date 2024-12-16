# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 7 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 7 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 7 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 7 --use_tool True
python gsm8k.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python svamp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python tabmwp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python humaneval.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python mbpp.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True

###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_qa_benchmark = "hotpotqa"

# fewshot_type = "cot"
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "cot" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "cot" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "cot" --max_interactions 10 --use_tool True

python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "cot" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "cot" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "cot" --max_interactions 10 --use_tool True

python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 10 --use_tool True

# fewshot_type = "direct"
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "direct" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "direct" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "direct" --max_interactions 10 --use_tool True

python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "direct" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "direct" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "direct" --max_interactions 10 --use_tool True

python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 10 --use_tool True

# fewshot_type = "react"
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "react" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "react" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 5  --fewshot_type "react" --max_interactions 10 --use_tool True

python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "react" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "react" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 8  --fewshot_type "react" --max_interactions 10 --use_tool True

python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 5 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool True
python ${hparam_qa_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 10 --use_tool True

hparam_math_code_benchmark = "gsm8k"

# fewshot_type = "cot"
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True

python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True

python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True

# fewshot_type = "direct"
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True

python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True

python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True

# fewshot_type = "react"
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True

python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True

python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 5 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 7 --use_tool True
python ${hparam_math_code_benchmark}.py --model "gpt-3.5-turbo" --eval_model "gpt-4o-mini" --seed 42 --max_interactions 10 --use_tool True
