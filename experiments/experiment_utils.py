"""
Utility functions for running experiments across different benchmarks and methods.
"""

import importlib
from typing import List, Dict, Any
from datasets import load_dataset

# Registry for method modules and config names
METHOD_REGISTRY = {
    "react": {
        "module": "agential.methods.react",
        "class": "ReAct",
        "config": "BENCHMARK_CONFIG",
        "supported_benchmarks": ["hotpotqa", "fever", "ambignq", "triviaqa", "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"],
    },
    "self_refine": {
        "module": "agential.methods.self_refine",
        "class": "SelfRefine",
        "config": "SELF_REFINE_BENCHMARK_CONFIG",
        "supported_benchmarks": ["hotpotqa", "fever", "ambignq", "triviaqa", "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"],
    },
    "cot": {
        "module": "agential.methods.cot",
        "class": "CoT",
        "config": "COT_BENCHMARK_CONFIG",
        "supported_benchmarks": ["hotpotqa", "fever", "ambignq", "triviaqa", "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"],
    },
    "clin": {
        "module": "agential.methods.clin",
        "class": "CLIN",
        "config": "CLIN_BENCHMARK_CONFIG",
        "supported_benchmarks": ["hotpotqa", "fever", "ambignq", "triviaqa", "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"],
    },
    "reflexion": {
        "module": "agential.methods.reflexion",
        "class": "Reflexion",
        "config": "BENCHMARK_CONFIG",
        "supported_benchmarks": ["hotpotqa", "fever", "ambignq", "triviaqa", "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"],
    },
    "expel": {
        "module": "agential.methods.expel",
        "class": "ExpeL",
        "config": "EXPEL_BENCHMARK_CONFIG",
        "supported_benchmarks": ["hotpotqa", "fever", "ambignq", "triviaqa", "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"],
    },
    "lats": {
        "module": "agential.methods.lats",
        "class": "LATS",
        "config": "BENCHMARK_CONFIG",
        "supported_benchmarks": ["hotpotqa", "fever", "ambignq", "triviaqa", "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"],
    },
    "critic": {
        "module": "agential.methods.critic",
        "class": "Critic",
        "config": "CRITIC_BENCHMARK_CONFIG",
        "supported_benchmarks": ["hotpotqa", "fever", "ambignq", "triviaqa", "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"],
    },
    "standard": {
        "module": "agential.methods.standard",
        "class": "Standard",
        "config": "BENCHMARK_CONFIG",
        "supported_benchmarks": [
            "hotpotqa", "fever", "ambignq", "triviaqa",
            "gsm8k", "svamp", "tabmwp", "humaneval", "mbpp"
        ],
    },
}

# Comprehensive benchmark registry with metadata
BENCHMARK_REGISTRY = {
    "hotpotqa": {
        "dataset": ("Sing0402/hotpotqa_200", "train"),
        "type": "qa",
        "description": "Multi-hop question answering",
        "requires_llm_judge": True,
        "key_field": "answer",
        "question_field": "question",
    },
    "fever": {
        "dataset": ("Sing0402/fever_200", "train"),
        "type": "qa",
        "description": "Fact extraction and verification",
        "requires_llm_judge": False,
        "key_field": "answer",
        "question_field": "question",
    },
    "ambignq": {
        "dataset": ("Sing0402/ambignq_200", "train"),
        "type": "qa",
        "description": "Ambiguous question answering",
        "requires_llm_judge": True,
        "key_field": "annotations",
        "question_field": "question",
        "complex_key": True,
    },
    "triviaqa": {
        "dataset": ("Sing0402/triviaqa", "train"),
        "type": "qa",
        "description": "Trivia question answering",
        "requires_llm_judge": True,
        "key_field": "answer",
        "question_field": "question",
        "complex_key": True,
    },
    "gsm8k": {
        "dataset": ("Sing0402/gsm8k_200", "train"),
        "type": "math",
        "description": "Grade school math word problems",
        "requires_llm_judge": False,
        "key_field": "answer",
        "question_field": "question",
        "complex_key": True,
    },
    "svamp": {
        "dataset": ("Sing0402/svamp_200", "train"),
        "type": "math",
        "description": "Simple arithmetic word problems",
        "requires_llm_judge": False,
        "key_field": "Answer",
        "question_field": "Body",
        "question_suffix": "Question",
        "complex_key": True,
    },
    "tabmwp": {
        "dataset": ("Sing0402/tabmwp_200", "train"),
        "type": "math",
        "description": "Table-based math word problems",
        "requires_llm_judge": False,
        "key_field": "answer",
        "question_field": "question",
        "table_field": "table",
        "complex_key": True,
    },
    "humaneval": {
        "dataset": ("openai/openai_humaneval", "test"),
        "type": "code",
        "description": "Human evaluation of code generation",
        "requires_llm_judge": False,
        "key_field": "test",
        "question_field": "prompt",
        "entry_point_field": "entry_point",
        "complex_key": True,
    },
    "mbpp": {
        "dataset": ("Sing0402/mbpp", "train"),
        "type": "code",
        "description": "Mostly Basic Python Problems",
        "requires_llm_judge": False,
        "key_field": "test_list",
        "question_field": "prompt",
        "test_imports_field": "test_imports",
        "complex_key": True,
    },
}

# Example answer keys for each benchmark (for quick tests/eval)
BENCHMARK_EXAMPLES = {
    "hotpotqa": ("Which book is the most popular in the world?", "The Bible"),
    "fever": ("Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", "SUPPORTS"),
    "ambignq": ("When did the simpsons first air on television?", "1989"),
    "triviaqa": ("Which American-born Sinclair won the Nobel Prize for Literature in 1930?", "Sinclair Lewis"),
    "gsm8k": ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "-9867630"),
    "svamp": ("There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. How big is each group of bananas?", "145"),
    "tabmwp": ("Read the following table regarding 'Bowling Scores' and then write Python code to answer a question:\n\nName | Score\nAmanda | 117\nSam | 236\nIrma | 144\nMike | 164\n\nQuestion: Some friends went bowling and kept track of their scores. How many more points did Mike score than Irma?", "20"),
    "humaneval": ("from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\ncheck(has_close_elements)"),
    "mbpp": ("Write a python function to find the first repeated character in a given string.", 'assert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"'),
}

# Evaluation functions for each benchmark type
from agential.eval.classification import (
    EM,
    f1,
    fuzzy_EM,
    llm_as_judge_eval,
    precision,
    recall,
)
from agential.utils.general import safe_execute

def validate_method_benchmark_compatibility(method: str, benchmark: str) -> bool:
    """Validate if a method supports a given benchmark."""
    if method not in METHOD_REGISTRY:
        return False
    if benchmark not in BENCHMARK_REGISTRY:
        return False
    return benchmark in METHOD_REGISTRY[method]["supported_benchmarks"]

def get_benchmark_type(benchmark: str) -> str:
    """Get the type of a benchmark (qa, math, code)."""
    if benchmark not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return BENCHMARK_REGISTRY[benchmark]["type"]

def extract_complex_key(key_data: Any, benchmark: str) -> List[str]:
    """Extract answers from complex key structures."""
    if benchmark == "ambignq":
        # Handle ambignq annotations structure
        answers = []
        for ann in key_data:
            if ann["type"] == "singleAnswer":
                answers.extend(ann["answer"])
            else:
                for qa_pair in ann["qaPairs"]:
                    answers.extend(qa_pair["answer"])
        return list(set(answers))
    elif benchmark == "triviaqa":
        # Handle triviaqa answer structure
        return list(set(key_data["normalized_aliases"]))
    elif benchmark == "gsm8k":
        # Handle gsm8k answer processing
        answer = str(float(str(key_data).split("#### ")[-1].strip().replace(",", "")))
        return [answer]
    elif benchmark == "humaneval":
        # Handle humaneval answer construction
        # This will be handled in construct_complex_key with instance data
        return [str(key_data)]
    elif benchmark == "mbpp":
        # Handle mbpp answer construction
        # This will be handled in construct_complex_key with instance data
        return [str(key_data)]
    else:
        # Default: return as single answer
        return [str(key_data)]

def construct_complex_question(instance: Any, benchmark: str) -> str:
    """Construct questions from complex data structures."""
    if benchmark == "svamp":
        # Handle svamp question construction
        body = instance[BENCHMARK_REGISTRY[benchmark]["question_field"]]
        question_suffix = instance[BENCHMARK_REGISTRY[benchmark]["question_suffix"]]
        return f"{body} {question_suffix}"
    elif benchmark == "tabmwp":
        # Handle tabmwp question construction
        question = instance[BENCHMARK_REGISTRY[benchmark]["question_field"]]
        table = instance[BENCHMARK_REGISTRY[benchmark]["table_field"]]
        return f"Read the following table regarding and then write Python code to answer a question:\n\n{table}\n\nQuestion: {question}"
    else:
        # Default: return simple question field
        return instance[BENCHMARK_REGISTRY[benchmark]["question_field"]]

def construct_complex_key(instance: Any, benchmark: str) -> str:
    """Construct keys from complex data structures."""
    if benchmark == "humaneval":
        # Handle humaneval key construction
        test = instance[BENCHMARK_REGISTRY[benchmark]["key_field"]]
        entry_point = instance[BENCHMARK_REGISTRY[benchmark]["entry_point_field"]]
        return f"{test}\ncheck({entry_point})"
    elif benchmark == "mbpp":
        # Handle mbpp key construction
        test_imports = instance[BENCHMARK_REGISTRY[benchmark]["test_imports_field"]]
        test_list = instance[BENCHMARK_REGISTRY[benchmark]["key_field"]]
        return "\n".join(test_imports + [""] + test_list).strip()
    else:
        # Default: return simple key field
        return instance[BENCHMARK_REGISTRY[benchmark]["key_field"]]

def evaluate_math_answer(answer: str, key: str) -> bool:
    """Evaluate math answers by executing the code and comparing numeric results."""
    try:
        code_str = answer.replace("```python", "").replace("```", "").strip()
        code_with_imports = f"from typing import *\n{code_str}"
        code_answer, execution_status = safe_execute(code_with_imports)
        if code_answer and len(code_answer) > 0:
            numeric_answer = str(code_answer[0])
            return EM(numeric_answer, key, is_numeric=True)
        else:
            return False
    except Exception:
        return False

def evaluate_code_answer(answer: str, key: str, benchmark: str) -> bool:
    """Evaluate code answers using safe_execute."""
    try:
        code_str = answer.replace("```python", "").replace("```", "").strip()
        if benchmark == "humaneval" or benchmark == "mbpp":
            _, execution_status = safe_execute(f"from typing import *\n\n{code_str}\n{key}")
            return EM(execution_status, "Done", normalize=False)
        else:
            return False
    except Exception as e:
        print(f"Warning: Code evaluation failed: {e}")
        return False

def evaluate_answer(benchmark: str, answer: str, key: str, eval_llm=None, question: str = "") -> dict:
    """Evaluate answer with comprehensive metrics based on benchmark type."""
    if not answer or answer.strip() == "":
        return {
            "em": 0,
            "fuzzy_em": 0,
            "llm_judge_eval": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    benchmark_type = get_benchmark_type(benchmark)
    
    if benchmark_type == "qa":
        # Handle complex key structures for QA benchmarks
        if BENCHMARK_REGISTRY[benchmark].get("complex_key", False):
            # Extract multiple answers from complex structure
            answers = extract_complex_key(key, benchmark)
            # Use any() for EM and fuzzy_EM, max() for precision/recall/f1
            is_correct = int(any([EM(answer, ans) for ans in answers]))
            is_correct_fuzzy = int(any([fuzzy_EM(answer, ans) for ans in answers]))
            precision_score = max([precision(answer, ans) for ans in answers])
            recall_score = max([recall(answer, ans) for ans in answers])
            f1_score = max([f1(answer, ans) for ans in answers])
        else:
            # Simple single answer evaluation
            is_correct = int(EM(answer, key))
            is_correct_fuzzy = int(fuzzy_EM(answer, key))
            precision_score = precision(answer, key)
            recall_score = recall(answer, key)
            f1_score = f1(answer, key)
        
        # LLM as judge evaluation
        llm_judge_score = 0
        if eval_llm and question and BENCHMARK_REGISTRY[benchmark]["requires_llm_judge"]:
            if BENCHMARK_REGISTRY[benchmark].get("complex_key", False):
                # Pass the extracted answers list to LLM judge
                answers = extract_complex_key(key, benchmark)
                llm_judge_score = int(llm_as_judge_eval(
                    llm=eval_llm, question=question, answer=answer, key=answers
                ))
            else:
                llm_judge_score = int(llm_as_judge_eval(
                    llm=eval_llm, question=question, answer=answer, key=key
                ))
        
        return {
            "em": is_correct,
            "fuzzy_em": is_correct_fuzzy,
            "llm_judge_eval": llm_judge_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        }
    
    elif benchmark_type == "math":
        # Math benchmarks: use numeric EM (safe_execute + EM)
        if benchmark == "gsm8k":
            key = str(float(key.split("#### ")[-1].strip().replace(",", "")))
        is_correct = int(evaluate_math_answer(answer, key))
        return {
            "em": is_correct,
            "fuzzy_em": is_correct,  # Same as EM for numeric
            "llm_judge_eval": is_correct,  # Same as EM for numeric
            "precision": float(is_correct),
            "recall": float(is_correct),
            "f1": float(is_correct),
        }
    
    elif benchmark_type == "code":
        # Code benchmarks: use safe_execute + EM (pass@1)
        is_correct = int(evaluate_code_answer(answer, key, benchmark))
        return {
            "em": is_correct,
            "fuzzy_em": is_correct,  # Same as EM for code
            "llm_judge_eval": is_correct,  # Same as EM for code
            "precision": float(is_correct),
            "recall": float(is_correct),
            "f1": float(is_correct),
        }
    
    else:
        # Default to fuzzy EM for unknown benchmark types
        is_correct_fuzzy = int(fuzzy_EM(answer, key))
        return {
            "em": is_correct_fuzzy,
            "fuzzy_em": is_correct_fuzzy,
            "llm_judge_eval": is_correct_fuzzy,
            "precision": float(is_correct_fuzzy),
            "recall": float(is_correct_fuzzy),
            "f1": float(is_correct_fuzzy),
        }

def get_benchmark_data(benchmark: str) -> List[Any]:
    """Load and validate benchmark data."""
    if benchmark not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    benchmark_info = BENCHMARK_REGISTRY[benchmark]
    dataset_name, split = benchmark_info["dataset"]
    
    try:
        dataset = load_dataset(dataset_name)[split]  # type: ignore
        data = list(dataset)
        
        # Validate that required fields exist
        required_fields = [benchmark_info["question_field"], benchmark_info["key_field"]]
        if data:
            sample = data[0]
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                raise ValueError(f"Missing required fields in dataset: {missing_fields}")
        
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset for benchmark {benchmark}: {e}")

def get_agent_instance(method: str, benchmark: str, llm, overrides: dict = {}) -> Any:
    """Create and return an agent instance for the given method and benchmark."""
    if not validate_method_benchmark_compatibility(method, benchmark):
        raise ValueError(f"Method {method} does not support benchmark {benchmark}")
    method_info = METHOD_REGISTRY[method]
    try:
        method_module = importlib.import_module(method_info["module"])
        AgentClass = getattr(method_module, method_info["class"])
        # Get config and validate benchmark support
        config = getattr(method_module, method_info["config"])
        if benchmark not in config:
            raise ValueError(f"Benchmark {benchmark} not found in {method} config")
        # Use overrides for agent init if provided
        if overrides:
            return AgentClass(llm, benchmark, **overrides)
        else:
            return AgentClass(llm, benchmark)
    except Exception as e:
        raise RuntimeError(f"Failed to create agent for method {method}: {e}")

def generate_agent_response(agent: Any, method: str, benchmark: str, question: str, key: str, overrides: dict = {}) -> Dict[str, Any]:
    """Generate response using the agent with method-agnostic param overrides."""
    params = {}
    # Use method defaults if present
    if method in METHOD_REGISTRY and "generate_params" in METHOD_REGISTRY[method]:
        params.update(METHOD_REGISTRY[method]["generate_params"])
    # Always set question
    params["question"] = question
    # Let overrides/config specify any additional keys (including additional_keys, refine_additional_keys, etc.)
    if overrides:
        params.update(overrides)
    return agent.generate(**params, additional_keys={"tests": key}) 