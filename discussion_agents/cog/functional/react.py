"""Functional module for ReAct."""
from typing import Any, Tuple, Dict
from tiktoken import Encoding
from langchain.prompts import PromptTemplate
from discussion_agents.cog.prompts.react import REACT_INSTRUCTION, WEBTHINK_SIMPLE6
from discussion_agents.utils.parse import remove_newline
from langchain.agents.react.base import DocstoreExplorer


def _build_agent_prompt(question: str, scratchpad: str) -> PromptTemplate:
    prompt = PromptTemplate.from_template(REACT_INSTRUCTION).format(
        examples=WEBTHINK_SIMPLE6,
        question=question,
        scratchpad=scratchpad
    )
    return prompt

def _prompt_agent(llm: Any, question: str, scratchpad: str) -> str:
    prompt = _build_agent_prompt(question=question, scratchpad=scratchpad)
    out = llm(prompt)
    return remove_newline(out)

def _is_halted(
    finished: bool,
    step_n: int, 
    max_steps: int, 
    question: str, 
    scratchpad: str, 
    max_tokens: int,
    enc: Encoding
) -> bool:
    over_max_steps = step_n > max_steps
    over_token_limit = len(enc.encode(_build_agent_prompt(question=question, scratchpad=scratchpad))) > max_tokens
    return finished or over_max_steps or over_token_limit


def react_think(llm: Any, question: str, scratchpad: str) -> str:
    scratchpad += f"\nThought:"
    thought = _prompt_agent(
        llm=llm, 
        question=question, 
        scratchpad=scratchpad
    ).split("Action")[0]
    scratchpad += " " + thought
    return scratchpad

def react_act(llm: Any, question: str, scratchpad: str) -> Tuple[str, str]:
    scratchpad += f"\nAction:"
    action = _prompt_agent(
        llm=llm, 
        question=question, 
        scratchpad=scratchpad
    ).split("Observation")[0]
    scratchpad += " " + action
    return scratchpad, action

def react_observe(
    action_type: str, 
    query: str, 
    scratchpad: str, 
    step_n: int, 
    docstore: DocstoreExplorer
) -> Dict[str, Any]:
    scratchpad += f"\nObservation {step_n}: "
    
    out = {
        "scratchpad": scratchpad,
        "answer": None,
        "finished": False,
        "step_n": step_n
    }

    if action_type.lower() == "finish":
        out["answer"] = query
        out["finished"] = True
        out["scratchpad"] += f"{query}"
    elif action_type.lower() == "search":
        try:
            out["scratchpad"] += remove_newline(docstore.search(query))
        except Exception:
            out["scratchpad"] += f"Could not find that page, please try again."
        
    elif action_type.lower() == "lookup":
        try:
            out["scratchpad"] += remove_newline(docstore.lookup(query))
        except ValueError:
            out["scratchpad"] += f"The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
    else:
        out["scratchpad"] += "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."

    out["step_n"] += 1

    return out