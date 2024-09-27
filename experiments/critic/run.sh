# Base runs
python hotpotqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --evidence_length 400 \
    --num_results 8 \
    --fewshot_type "cot" \
    --max_interactions 7 \
    --use_tool True

python fever.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --evidence_length 400 \
    --num_results 8 \
    --fewshot_type "cot" \
    --max_interactions 7 \
    --use_tool True

python ambignq.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --evidence_length 400 \
    --num_results 8 \
    --fewshot_type "cot" \
    --max_interactions 7 \
    --use_tool True

python triviaqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --evidence_length 400 \
    --num_results 8 \
    --fewshot_type "cot" \
    --max_interactions 7 \
    --use_tool True

###############################################################
# HotpotQA, EVERYTHING BELOW HASN'T BEEN RAN
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

###############################################################
# FEVER
###############################################################

# fewshot_type=cot
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 10 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 10 --use_tool True

# fewshot_type=direct
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 10 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 10 --use_tool True

# fewshot_type=react
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 10 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 10 --use_tool True

###############################################################
# AmbigNQ
###############################################################

# fewshot_type=cot
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 10 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 10 --use_tool True

# fewshot_type=direct
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 10 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 10 --use_tool True

# fewshot_type=react
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 10 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 10 --use_tool True

###############################################################
# TriviaQA
###############################################################

# fewshot_type=cot
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 10 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 10 --use_tool True

# fewshot_type=direct
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 10 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 10 --use_tool True

# fewshot_type=react
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 10 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 10 --use_tool True
