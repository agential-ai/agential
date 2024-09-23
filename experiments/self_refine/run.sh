python hotpotqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \ 
    --seed 42 \
    --patience 1 \
    --fewshot_type "cot" \
    --max_interactions 3

python fever.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \ 
    --seed 42 \
    --patience 1 \
    --fewshot_type "cot" \
    --max_interactions 3

python ambignq.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \ 
    --seed 42 \
    --patience 1 \
    --fewshot_type "cot" \
    --max_interactions 3

python triviaqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \ 
    --seed 42 \
    --patience 1 \
    --fewshot_type "cot" \
    --max_interactions 3