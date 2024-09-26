# Base runs
python hotpotqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 1 \
    --reflect_strategy "reflexion"

python fever.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 1 \
    --reflect_strategy "reflexion"

python ambignq.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 1 \
    --reflect_strategy "reflexion"

python triviaqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 1 \
    --reflect_strategy "reflexion"
