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

# Fewshot Type: cot
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 3 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 3 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 5 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 9 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 9 --use_tool False

# Fewshot Type: direct, CONTINUE EXPERIMENTS BELOW
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 3 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 3 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "direct" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "direct" --max_interactions 5 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 9 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 9 --use_tool False

# Fewshot Type: react
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 3 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 3 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "react" --max_interactions 5 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "react" --max_interactions 5 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool False
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 9 --use_tool True
python hotpotqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 9 --use_tool False

# Fewshot Type: cot
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 3 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 3 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 5 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 9 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 9 --use_tool False

# Fewshot Type: direct
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 3 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 3 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "direct" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "direct" --max_interactions 5 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 9 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 9 --use_tool False

# Fewshot Type: react
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 3 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 3 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "react" --max_interactions 5 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "react" --max_interactions 5 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool False
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 9 --use_tool True
python fever.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 9 --use_tool False

# Fewshot Type: cot
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 3 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 3 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 5 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 9 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 9 --use_tool False

# Fewshot Type: direct
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 3 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 3 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "direct" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "direct" --max_interactions 5 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 9 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 9 --use_tool False

# Fewshot Type: react
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 3 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 3 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "react" --max_interactions 5 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "react" --max_interactions 5 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool False
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 9 --use_tool True
python ambignq.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 9 --use_tool False

# Fewshot Type: cot
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 3 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 3 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "cot" --max_interactions 5 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "cot" --max_interactions 7 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 9 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "cot" --max_interactions 9 --use_tool False

# Fewshot Type: direct
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 3 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 3 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "direct" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "direct" --max_interactions 5 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "direct" --max_interactions 7 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 9 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "direct" --max_interactions 9 --use_tool False

# Fewshot Type: react
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 3 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 3 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "react" --max_interactions 5 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 8 --fewshot_type "react" --max_interactions 5 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 10 --fewshot_type "react" --max_interactions 7 --use_tool False
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 9 --use_tool True
python triviaqa.py --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --evidence_length 400 --num_results 5 --fewshot_type "react" --max_interactions 9 --use_tool False

