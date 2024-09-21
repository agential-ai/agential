python hotpotqa.py \
    --model "gpt-3.5-turbo" \
    --seed 42 \
    --num_retries 1 \
    --warming 0.0

python fever.py \
    --model "gpt-3.5-turbo" \
    --seed 42 \
    --num_retries 1 \
    --warming 0.0

python ambignq.py \
    --model "gpt-3.5-turbo" \
    --seed 42 \
    --num_retries 1 \
    --warming 0.0

python triviaqa.py \
    --model "gpt-3.5-turbo" \
    --seed 42 \
    --num_retries 1 \
    --warming 0.0