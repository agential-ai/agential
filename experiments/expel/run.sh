# Training
python hotpotqa_train.py \
    --n_train_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python fever_train.py \
    --n_train_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python ambignq_train.py \
    --n_train_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python triviaqa_train.py \
    --n_train_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python gsm8k_train.py \
    --n_train_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python tabmwp_train.py \
    --n_train_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python mbpp_train.py \
    --n_train_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

hotpotqa_train_run = "cerulean-waterfall-2"
fever_train_run = ""
ambignq_train_run = ""
triviaqa_train_run = ""
gsm8k_train_run = ""
tabmwp_train_run = ""
mbpp_train_run = ""

# Base runs
python hotpotqa.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/hotpotqa/${hotpotqa_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/hotpotqa/${hotpotqa_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python fever.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/fever/${fever_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/fever/${fever_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python ambignq.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/ambignq/${ambignq_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/ambignq/${ambignq_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python triviaqa.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/triviaqa/${triviaqa_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/triviaqa/${triviaqa_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python gsm8k.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/gsm8k/${gsm8k_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/gsm8k/${gsm8k_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python svamp.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/gsm8k/${gsm8k_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/gsm8k/${gsm8k_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python tabmwp.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/tabmwp/${tabmwp_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/tabmwp/${tabmwp_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python humaneval.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/mbpp/${mbpp_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/mbpp/${mbpp_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"

python mbpp.py \
    --n_eval_samples -1 \
    --model "gpt-3.5-turbo" \
    --eval_model "gpt-4o" \
    --seed 42 \
    --max_reflections 3 \
    --max_trials 3 \
    --max_steps 6 \
    --experience_memory_strategy "task" \
    --embedder "huggingface" \
    --experiences_path "output/mbpp/${mbpp_train_run}-expel-exp-memories.pkl" \
    --insights_path "output/mbpp/${mbpp_train_run}-expel-insights-memories.pkl" \
    --max_insights 20 \
    --leeway 5 \
    --success_batch_size 8 \
    --extract_init_insights True \
    --patience 3 \
    --reflect_strategy "reflexion" \
    --use_dynamic_examples True \
    --extract_insights True \
    --k_docs 24 \
    --num_fewshots 6 \
    --max_fewshot_tokens 3000 \
    --reranker_strategy "none"


###############################################################
# [Optional] Hyperparameter Sweep
###############################################################

hparam_benchmark = "hotpotqa"
experiences_path = ""
insights_path = ""

#  max_insights = [10, 15, 25, 30], reranker_strategy = "none" 
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 10 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 15 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 25 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 30 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"

# max_insights = [10, 15, 25, 30], reranker_strategy = "task" 
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 10 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "task"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 15 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "task"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "task"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 25 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "task"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 30 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "task"

# max_insights = [10, 15, 25, 30], reranker_strategy = "thought" 
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 10 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "thought"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 15 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "thought"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "thought"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 25 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "thought"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 30 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "thought"

# max_insights = [10, 15, 25, 30]  reranker_strategy = "length" 
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 10 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "length"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 15 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "length"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "length"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 25 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "length"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 30 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "length"


# experience memory strategy = ["action", "thought", "step"]
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "action" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "thought" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "step" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"

# use dynamic examples = True
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples False --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"

# k-docs = [19, 21, 27, 29]  
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 19 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 21 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 27 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 29 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"

# num_fewshots = [5, 6, 7]
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 5 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 7 --max_fewshot_tokens 3000 --reranker_strategy "none"

# leeway = [4, 5, 6] 
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 4 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 5 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
python ${hparam_benchmark}.py --n_eval_samples -1 --model "gpt-3.5-turbo" --eval_model "gpt-4o" --seed 42 --max_reflections 3 --max_trials 3 --max_steps 6 --experience_memory_strategy "task" --embedder "huggingface" --experiences_path ${experiences_path} --insights_path ${insights_path} --max_insights 20 --leeway 6 --success_batch_size 8 --extract_init_insights True --patience 3 --reflect_strategy "reflexion" --use_dynamic_examples True --extract_insights True --k_docs 24 --num_fewshots 6 --max_fewshot_tokens 3000 --reranker_strategy "none"
