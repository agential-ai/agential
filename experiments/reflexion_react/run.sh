# Base runs
python hotpotqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 1 \
    --reflect_strategy "reflexion"\
    --max_steps 6 \
    --max_tokens 5000

python fever.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 1 \
    --reflect_strategy "reflexion"\
    --max_steps 6 \
    --max_tokens 5000

python ambignq.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 1 \
    --reflect_strategy "reflexion"\
    --max_steps 6 \
    --max_tokens 5000

python triviaqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 1 \
    --reflect_strategy "reflexion"\
    --max_steps 6 \
    --max_tokens 5000


# reflect_strategy [ last_attempt , last_attempt_and_reflexion ] 
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 1 --reflect_strategy "last_attempt" --max_steps 6 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 1 --reflect_strategy "last_attempt_and_reflexion" --max_steps 6 --max_tokens 5000

# max_steps [ 6 , 10 , 12 ]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 1 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 1 --reflect_strategy "reflexion" --max_steps 10 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 1 --reflect_strategy "reflexion" --max_steps 12 --max_tokens 5000


# max_trials [5, 7]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 5 --patience 1 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 7 --patience 1 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 5000

# max_reflections [5, 7]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 5 --max_trials 3 --patience 1 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 7 --max_trials 3 --patience 1 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 5000

# patience [2, 3]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 2 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 5000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 5000

#max_tokens [10000 ,15000]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 2 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 10000
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "reflexion" --max_steps 6 --max_tokens 15000
