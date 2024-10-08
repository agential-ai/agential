"""Run ExpeL on HotpotQA."""

import os
import warnings

import tiktoken

warnings.filterwarnings('ignore')

from agential.agents.expel.agent import ExpeL
from agential.agents.expel.memory import ExpeLExperienceMemory, ExpeLInsightMemory
from agential.agents.reflexion.agent import ReflexionReAct
from agential.core.llm import LLM
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

import wandb
wandb.login()
from datasets import load_dataset

from experiments.utils import set_seed

import argparse

parser = argparse.ArgumentParser(description="Run Standard experiments.")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model")
parser.add_argument("--eval_model", type=str, default="gpt-4o", help="The evaluator model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_reflections", type=int, default=3, help="Max reflections")
parser.add_argument("--max_trials", type=int, default=3, help="Max trials")
parser.add_argument("--patience", type=int, default=3, help="Patience")
parser.add_argument("--reflect_strategy", type=str, default="reflexion", help="Reflection strategy")
parser.add_argument("--max_steps", type=int, default=6, help="Max steps")
parser.add_argument("--max_tokens", type=int, default=5000, help="Max tokens")
parser.add_argument("--experience_memory_strategy", type=str, default="task", help="Experience memory strategy")
parser.add_argument("--embedder", type=str, default="huggingface", help="Embedder")
args = parser.parse_args()

set_seed(args.seed)
root_dir = "output"
method_name = "expel"
benchmark = "hotpotqa"

if __name__ == '__main__':
    data = load_dataset("alckasoc/hotpotqa_500")['train']

    model = args.model
    eval_model = args.eval_model
    seed = args.seed
    max_reflections = args.max_reflections
    max_trials = args.max_trials
    patience = args.patience
    reflect_strategy = args.reflect_strategy
    max_steps = args.max_steps
    max_tokens = args.max_tokens
    experience_memory_strategy = args.experience_memory_strategy
    embedder = args.embedder

    embedder_dict = {
        "huggingface": HuggingFaceEmbeddings
    }

    output_path = os.path.join(root_dir, benchmark)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    llm = LLM(
        model, 
        organization=os.getenv("OPENAI_ORGANIZATION"), 
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=seed
    )

    eval_llm = LLM(
        eval_model,
        organization=os.getenv("OPENAI_ORGANIZATION"),
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=seed
    )

    try:
        enc = tiktoken.encoding_for_model(args.model)
    except:
        enc = tiktoken.get_encoding("gpt-3.5-turbo")

    reflexion_react_agent = ReflexionReAct(
        llm=llm,
        benchmark=benchmark,
        max_reflections=max_reflections,
        max_trials=max_trials,
        max_steps=max_steps,
        max_tokens=max_tokens,
        enc=enc,
    )

    agent = ExpeL(
        llm=llm,
        benchmark=benchmark,
        reflexion_react_agent=reflexion_react_agent,
        experience_memory=ExpeLExperienceMemory(
            experiences=[],
            strategy=experience_memory_strategy,
            embedder=embedder_dict[embedder](),
            encoder=enc
        ),
        insight_memory=ExpeLInsightMemory(
            max_num_insights=20,
            leeway=5
        ),
        success_batch_size=8
    )