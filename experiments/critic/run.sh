# Base runs
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 7 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 7 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 7 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 7 --use_tool True

###############################################################
# Hyperparameter Sweep
###############################################################

# fewshot_type=cot
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 10 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 10 --use_tool True

# fewshot_type=direct
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 10 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 10 --use_tool True

# fewshot_type=react
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 10 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 10 --use_tool True
