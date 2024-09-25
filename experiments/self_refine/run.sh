# Base runs
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

# fewshot_type="direct", max_interactions=3
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 3
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 3
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 3
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 3

# fewshot_type="react", max_interactions=3
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 3
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 3
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 3
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 3

# fewshot_type="cot", max_interactions=1
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 1
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 1
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 1
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 1

# fewshot_type="direct", max_interactions=1
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 1
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 1
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 1
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 1

# fewshot_type="react", max_interactions=1
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 1
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 1
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 1
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 1

# fewshot_type="cot", max_interactions=2
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 2
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 2
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 2
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 2

# fewshot_type="direct", max_interactions=2
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 2
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 2
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 2
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 2


# fewshot_type="react", max_interactions=2
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 2
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 2
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 2
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 2

# fewshot_type="cot", max_interactions=5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 5
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 5
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 5
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 5

# fewshot_type="direct", max_interactions=5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 5
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 5
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 5
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 5

# fewshot_type="react", max_interactions=5
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 5
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 5
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 5
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 5

# fewshot_type="cot", max_interactions=7
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 7
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 7
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 7
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "cot" --max_interactions 7

# fewshot_type="direct", max_interactions=7
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 7
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 7
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 7
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "direct" --max_interactions 7

# fewshot_type="react", max_interactions=7
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 7
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 7
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 7
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --patience 1 --fewshot_type "react" --max_interactions 7
