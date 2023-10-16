"""Planning module for Generative Agents."""
from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory
from langchain.schema.language_model import BaseLanguageModel

from discussion_agents.utils.parse import parse_list

def generate_broad_plan(
    instruction: str,
    lifestyle: str,
    name: str,
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
) -> List[str]:
    prompt = PromptTemplate.from_template(
        "{instruction}\n"
        + "In general, {name}'s lifestyle: {lifestyle}\n"
        + "Provide each step on a new line. "
        + "Here is {name}'s plan in broad-strokes:\n"
        + "1) "
    )
    chain = LLMChain(llm=llm, llm_kwargs=llm_kwargs, prompt=prompt, memory=memory)
    result = parse_list(chain.run(instruction=instruction, name=name, lifestyle=lifestyle).strip())
    result = [s.split(")")[-1].rstrip(",.").strip() for s in result]

    return result


def update_status(
    previous_steps: List[str],
    plan_step: str,
    name: str,
    status: str,
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
) -> str:
    previous_steps = "\n".join(previous_steps)

    plan_prompt = PromptTemplate.from_template(
        previous_steps
        + "\n"
        + "Given the statements above, "
        + "is there anything that {name} should remember as they plan for:\n"
        + plan_step
        + "Write the response from {name}'s perspective."
    )
    chain = LLMChain(llm=llm, llm_kwargs=llm_kwargs, prompt=plan_prompt, memory=memory)
    plan_result = chain.run(name=name).strip()

    thought_prompt = PromptTemplate.from_template(
        previous_steps
        + "\n"
        + "Given the statements above, how might we summarize "
        + "{name}'s thoughts about the plan up till now?\n\n"
        + "Write the response from {name}'s perspective."
    )
    chain = LLMChain(
        llm=llm, llm_kwargs=llm_kwargs, prompt=thought_prompt, memory=memory
    )
    thought_result = chain.run(name=name).strip()

    status_prompt = PromptTemplate.from_template(
        "{name}'s status from the previous step: "
        + "{status}\n\n"
        + "{name}'s thoughts at the end of the previous step: "
        + (plan_result + " " + thought_result).replace("\n", "")
        + "\n\n"
        + "Given the above, write {name}'s status "
        + "that reflects {name}'s "
        + "thoughts at the end of the previous step. "
        + "Write this in third-person talking about {name}."
        + "Follow this format below:\nStatus: <new status>"
    )
    chain = LLMChain(
        llm=llm, llm_kwargs=llm_kwargs, prompt=status_prompt, memory=memory
    )
    status = chain.run(name=name, status=status).strip()

    return status


def update_broad_plan(
    instruction: str, 
    name: str,
    plan: List[str],
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
) -> List[str]:
    plan = "\n".join(plan)

    daily_plan_req_prompt = PromptTemplate.from_template(
        "Instruction: {instruction}\n"
        + "Here is {name}'s plan in broad-strokes:\n"
        + plan
        + "Are there any updates or changes to be made to this plan? "
        + "Update the plan to incorporate the changes. "
        + "If there are no changes to be made, simply return the same plan."
        + "Follow this format for the plan:\n"
        + "1) <text>\n2) <text>\n3) ..."
    )
    daily_plan_req_kwargs = dict(
        instruction=instruction,
        name=name
    )
    chain = LLMChain(
        llm=llm, llm_kwargs=llm_kwargs, prompt=daily_plan_req_prompt, memory=memory
    )
    result = parse_list(chain.run(**daily_plan_req_kwargs).strip())
    result = [s.split(")")[-1].rstrip(",.").strip() for s in result]

    return result


def generate_refined_plan(
    instruction: str, 
    plan: List[str],
    name: str,
    llm: BaseLanguageModel,
    memory: BaseMemory,
    k: int = 1,
    llm_kwargs: Dict[str, Any] = {"max_tokens": 3000, "temperature": 0.8}
) -> List[str]:
    plan_format = ""
    for i, step in enumerate(plan):
        plan_format += f"{i}. {step}\n"
        plan_format += f"Substep: <Fill in>\n"
    plan_format = plan_format[:-1]

    prompt = PromptTemplate.from_template(
        "Instruction: \n{instruction}\n\n"
        + "Plan format: \n"
        + "{plan_format}\n\n"
        + "Given the instruction and the current plan above, "
        + "for each planning step, fill in 'Substep:', "
        + "which describes in detail what the substeps for that given planning step are. "
        + "If a planning step does not require a substep(s), then do not specify any substep. "
        + "Return the entire plan with the included substeps. Keep the steps descriptive, but concise."
        + "Output format example: \n"
        + "1) <first step>\n"
        + "1.1) <first step, first substep>\n"
        + "1.2) <first step, second substep>\n"
    )
    chain = LLMChain(llm=llm, llm_kwargs=llm_kwargs, prompt=prompt, memory=memory)

    results = []
    for _ in range(k):
        result = chain.run(instruction=instruction, plan_format=plan_format).strip()
        results.append(result)

    if k == 1:
        results = results[0]
    else:
        plans = [f"Plan {i}:\n{result}\n\n" for i, result in enumerate(results)]

        prompt = PromptTemplate.from_template(
            "Instruction: \n{instruction}\n\n"
            + "Below are {k} different plans for the instruction above. "
            + "Consolidate them into 1 plan."
            + "{plans}"
        )
        chain = LLMChain(llm=llm, llm_kwargs=llm_kwargs, prompt=prompt, memory=memory)
        results = chain.run(instruction=instruction, k=k, plans=plans).strip()

    result = parse_list(results)
    result = [s.split(")")[-1].rstrip(",.").strip() for s in result]

    return results
