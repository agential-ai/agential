python hotpotqa.py \
    --model "gpt-3.5-turbo" \
    --seed 42 \
    --max_steps 6 \
    --max_tokens 5000

python fever.py \
    --model "gpt-3.5-turbo" \
    --seed 42 \
    --max_steps 6 \
    --max_tokens 5000

python ambignq.py \
    --model "gpt-3.5-turbo" \
    --seed 42 \
    --max_steps 6 \
    --max_tokens 5000

python triviaqa.py \
    --model "gpt-3.5-turbo" \
    --seed 42 \
    --max_steps 6 \
    --max_tokens 5000