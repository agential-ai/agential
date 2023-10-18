"""Planning module for Generative Agents."""
from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory
from langchain.schema.language_model import BaseLanguageModel

from discussion_agents.utils.parse import parse_list

def generate_broad_plan(
    instruction: str,
    summary: str,
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
) -> List[str]:
    prompt = PromptTemplate.from_template(
        "Below is a summary of you."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Provide each step on a new line. "
        + "Here is your plan for the instruction in broad-strokes:\n"
        + "1) "
    )
    chain = LLMChain(llm=llm, llm_kwargs=llm_kwargs, prompt=prompt, memory=memory)
    result = parse_list(chain.run(instruction=instruction, summary=summary).strip())
    result = [s.split(")")[-1].rstrip(",.").strip() for s in result]

    return result


def update_status(
    instruction: str, 
    previous_steps: List[str],
    plan_step: str,
    summary: str,
    status: str,
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
) -> str:
    previous_steps = "\n".join(previous_steps)

    plan_prompt = PromptTemplate.from_template(
        "Below is a summary of you."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Previous steps for the above instruction: {previous_steps}\n"
        + "Given the statements above, "
        + "is there anything that you should remember as you plan for: {plan_step}\n"
        + "Write the response from your perspective."
    )
    chain = LLMChain(llm=llm, llm_kwargs=llm_kwargs, prompt=plan_prompt, memory=memory)
    plan_result = chain.run(
        summary=summary, 
        instruction=instruction, 
        previous_steps=previous_steps,
        plan_step=plan_step
    ).strip()

    thought_prompt = PromptTemplate.from_template(
        "Below is a summary of you."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Previous steps for the above instruction: {previous_steps}\n"
        + "Given the statements above, how might we summarize "
        + "your thoughts about the plan up till now?\n"
        + "Write the response from your perspective."
    )
    chain = LLMChain(
        llm=llm, llm_kwargs=llm_kwargs, prompt=thought_prompt, memory=memory
    )
    thought_result = chain.run(
        summary=summary,
        instruction=instruction,
        previous_steps=previous_steps
    ).strip()

    plan_and_thought = (plan_result + " " + thought_result).replace("\n", "")

    status_prompt = PromptTemplate.from_template(
        "Below is a summary of you."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Your status from the previous step: {status}\n\n"
        + "Your thoughts at the end of the previous step: {plan_and_thought}\n\n"
        + "Given the above, write a status "
        + "that reflects your status at the end of the previous step. "
        + "Write this in third-person talking about yourself."
        + "Follow this format below:\nStatus: <new status>"
    )
    chain = LLMChain(
        llm=llm, llm_kwargs=llm_kwargs, prompt=status_prompt, memory=memory
    )
    status = chain.run(
        summary=summary, 
        instruction=instruction, 
        status=status,
        plan_and_thought=plan_and_thought
    ).strip()

    return status


def update_broad_plan(
    instruction: str, 
    summary: str,
    plan: List[str],
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
) -> List[str]:
    daily_plan_req_prompt = PromptTemplate.from_template(
        "Below is a summary of you."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Here is your current plan in broad-strokes: \n{plan}"
        + "Are there any updates or changes to be made to this plan? "
        + "Update the plan to incorporate the changes. "
        + "If there are no changes to be made, simply return the same plan."
        + "Follow this format for the plan:\n"
        + "1) <text>\n2) <text>\n3) ..."
    )
    chain = LLMChain(
        llm=llm, llm_kwargs=llm_kwargs, prompt=daily_plan_req_prompt, memory=memory
    )
    broad_plan = chain.run(
        summary=summary,
        instruction=instruction,
        plan="\n".join(plan)
    ).strip()
    broad_plan = parse_list(broad_plan)
    broad_plan = [s.split(")")[-1].rstrip(",.").strip() for s in broad_plan]

    return broad_plan


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
