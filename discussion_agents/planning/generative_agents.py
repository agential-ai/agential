"""Planning module for Generative Agents."""
from typing import List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from discussion_agents.core.base import BaseCore
from discussion_agents.utils.parse import parse_numbered_list


def generate_broad_plan(
    instruction: str,
    summary: str,
    core: BaseCore,
) -> List[str]:
    prompt = PromptTemplate.from_template(
        "Below is a summary of you."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Provide each step on a new line. "
        + "Follow this format for the plan:\n"
        + "1) <text>\n"
        + "2) <text>\n"
        + "3) ...\n"
        + "Here is your plan for the instruction in broad-strokes:\n"
        + "1) "
    )
    chain = LLMChain(llm=core.llm, llm_kwargs=core.llm_kwargs, prompt=prompt)
    result = parse_numbered_list(
        chain.run(summary=summary, instruction=instruction).strip()
    )

    return result


def update_status(
    instruction: str,
    previous_steps: List[str],
    plan_step: str,
    summary: str,
    status: str,
    core: BaseCore,
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
    chain = LLMChain(llm=core.llm, llm_kwargs=core.llm_kwargs, prompt=plan_prompt)
    plan_result = chain.run(
        summary=summary,
        instruction=instruction,
        previous_steps=previous_steps,
        plan_step=plan_step,
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
    chain = LLMChain(llm=core.llm, llm_kwargs=core.llm_kwargs, prompt=thought_prompt)
    thought_result = chain.run(
        summary=summary, instruction=instruction, previous_steps=previous_steps
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
    chain = LLMChain(llm=core.llm, llm_kwargs=core.llm_kwargs, prompt=status_prompt)
    status = chain.run(
        summary=summary,
        instruction=instruction,
        status=status,
        plan_and_thought=plan_and_thought,
    ).strip()

    return status


def generate_refined_plan(
    instruction: str,
    previous_steps: List[str],
    plan_step: str,
    summary: str,
    core: BaseCore,
    k: int = 1,
    # llm_kwargs: Dict[str, Any] = {"max_tokens": 3000, "temperature": 0.8},
) -> List[str]:
    previous_steps = "\n".join(previous_steps)

    prompt = PromptTemplate.from_template(
        "Below is a summary of you."
        + "{summary}\n\n"
        + "Instruction: \n{instruction}\n\n"
        + "Previous steps in the plan for the above instruction: \n"
        + "{previous_steps}\n\n"
        + "Given the instruction and the steps taken thus far above, "
        + "should you generate more detailed substeps for the current step: {plan_step}?\n"
        + "If no substeps are required, simply generate <NO_SUBSTEPS_REQUIRED>."
        + "If substeps are required, return the  Keep the steps descriptive and concise."
        + "Output format example: \n"
        + "1) <first substep>\n"
        + "2) <second substep>\n"
        + "3) <third substep>\n"
    )
    chain = LLMChain(llm=core.llm, llm_kwargs=core.llm_kwargs, prompt=prompt)

    results = []
    for _ in range(k):
        result = chain.run(
            summary=summary, 
            instruction=instruction, 
            previous_steps=previous_steps,
            plan_step=plan_step
        ).strip()
        results.append(result)

    if k == 1:
        results = results[0]
    else:
        plans = [f"Sub-Plan {i}:\n{result}\n\n" for i, result in enumerate(results)]

        prompt = PromptTemplate.from_template(
            "Instruction: \n{instruction}\n\n"
            + "Previous steps in the plan for the above instruction: \n"
            + "{previous_steps}\n\n"
            + "The current plan step is: {plan_step}\n\n"
            + "Below are {k} different sub-plans for the instruction above. "
            + "Consolidate them into 1 plan such that the plan best answers the instruction."
            + "{plans}"
        )
        chain = LLMChain(llm=core.llm, llm_kwargs=core.llm_kwargs, prompt=prompt)
        results = chain.run(
            instruction=instruction, 
            previous_steps=previous_steps,
            plan_step=plan_step,
            k=k, 
            plans=plans
        ).strip()

    result = parse_numbered_list(results)

    return results
