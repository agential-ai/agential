# Base runs
python hotpotqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 3 \
    --reflect_strategy "reflexion"

python fever.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 3 \
    --reflect_strategy "reflexion"

python ambignq.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 3 \
    --reflect_strategy "reflexion"

python triviaqa.py \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --patience 3 \
    --reflect_strategy "reflexion"


###############################################################
# HotpotQA
###############################################################

# reflect_strategy [last_attempt, last_attempt_and_reflexion] 
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

# max_trials [5, 7]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 7 --patience 3 --reflect_strategy "reflexion"

# max_reflections [5, 7]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 7 --max_trials 3 --patience 3 --reflect_strategy "reflexion"

# patience [2, 4]
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 2 --reflect_strategy "reflexion"
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 4 --reflect_strategy "reflexion"

###############################################################
# FEVER, EVERYTHING BELOW HASN'T BEEN RAN 
###############################################################

# reflect_strategy [last_attempt, last_attempt_and_reflexion] 
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

# max_trials [5, 7]
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 7 --patience 3 --reflect_strategy "reflexion"

# max_reflections [5, 7]
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 7 --max_trials 3 --patience 3 --reflect_strategy "reflexion"

# patience [2, 4]
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 2 --reflect_strategy "reflexion"
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 4 --reflect_strategy "reflexion"

###############################################################
# AmbigNQ
###############################################################

# reflect_strategy [last_attempt, last_attempt_and_reflexion] 
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

# max_trials [5, 7]
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 7 --patience 3 --reflect_strategy "reflexion"

# max_reflections [5, 7]
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 7 --max_trials 3 --patience 3 --reflect_strategy "reflexion"

# patience [2, 4]
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 2 --reflect_strategy "reflexion"
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 4 --reflect_strategy "reflexion"

###############################################################
# TriviaQA
###############################################################

# reflect_strategy [last_attempt, last_attempt_and_reflexion] 
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt"
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 3 --reflect_strategy "last_attempt_and_reflexion"

# max_trials [5, 7]
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 5 --patience 3 --reflect_strategy "reflexion"
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 7 --patience 3 --reflect_strategy "reflexion"

# max_reflections [5, 7]
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 5 --max_trials 3 --patience 3 --reflect_strategy "reflexion"
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 7 --max_trials 3 --patience 3 --reflect_strategy "reflexion"

# patience [2, 4]
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 2 --reflect_strategy "reflexion"
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --patience 4 --reflect_strategy "reflexion"
