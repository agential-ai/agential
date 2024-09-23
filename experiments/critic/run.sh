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