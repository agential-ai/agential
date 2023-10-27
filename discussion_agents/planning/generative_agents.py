"""Planning module for Generative Agents.

Original Generative Agents planning module:
https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/plan.py 

The original Generative Agents planning architecture involved 
temporally-aware prompts and was made with simulating daily behavior 
in mind. Planning, here, is a generalized, instructional adaptation of the 
original generative Agents planning cognitive module.

The methods below are capable of recursively generating a broad -> detailed
plan while keeping track of the agent's status.

A list of changes on how this implementation deviates from the original paper:
- no temporal component
    - no wake up hour generation
    - no track of the day (new day or first day)
    - all planning components exclude timestamps (or day)
    - planning is irrespective of time
- simplified internal planning component
    - (our implementation): `plan_req` dict tracks the broad and refined plan of the agent and their `status`
    - compacting the following components from the original paper:
        - scratch memory currently (what the agent is currently doing) -> status
        - daily_req, daily_plan_req, f_daily_schedule, f_daily_schedule_hourly_org -> plan_req
    - `generate_hourly_schedule` in the original paper generates refined schedules until a threshold is met
    - (our implementation): multiple refined plans for a step are generated and combined via an LLM 
- planning is done by step, iteratively and not altogether at once 
"""
from typing import List

from langchain.prompts import PromptTemplate

from discussion_agents.core.base import BaseCore
from discussion_agents.utils.parse import parse_numbered_list


def generate_broad_plan(
    instruction: str,
    summary: str,
    core: BaseCore,
) -> List[str]:
    """Generate a broad plan based on provided instruction and agent summary.

    This function generates a broad plan in response to a given instruction and summary.
    It utilizes the provided BaseCore to create a prompt template and use it with an
    LLMChain to generate the plan.

    Args:
        instruction (str): The instruction to create a plan for.
        summary (str): A summary of your agent's relevant characteristics.
        core (BaseCore): The agent's core component used for generating plan.

    Returns:
        List[str]: A list of steps representing the generated broad plan.

    Example:
        instruction = "Prepare for a business presentation."
        summary = "..."
        core = BaseCore(...)  # Initialize with the necessary components.
        broad_plan = generate_broad_plan(instruction, summary, core)
    """
    prompt = PromptTemplate.from_template(
        "Below is a summary of your characteristics."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Provide each step on a new line. "
        + "Follow this format for the plan:\n"
        + "1) <text>\n"
        + "2) <text>\n"
        + "3) ...\n"
        + "Devise the plan according to your characteristics. "
        + "Here is your plan for the instruction in broad-strokes:\n"
        + "1) "
    )
    chain = core.chain(prompt=prompt)
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
    """Update the status of a plan step in response to provided information.

    This function takes an instruction, a list of previous steps, the current plan step,
    a summary, and a status update to incorporate into the plan. It uses the provided
    BaseCore to facilitate the update.

    Args:
        instruction (str): The original instruction related to the plan.
        previous_steps (List[str]): A list of previously generated plan steps.
        plan_step (str): The new/current plan step.
        summary (str): A summary of the agent's relevant characteristics.
        status (str): The status to update for the plan step.
        core (BaseCore): The core component used for plan update.

    Returns:
        str: A string representing the updated plan with the new status.

    Example:
        instruction = "Prepare for a business presentation."
        previous_steps = ["1) Research market trends.", "2) Create a product demo."]
        plan_step = "3) Prepare presentation slides."
        summary = "..."
        status = "Product demo requires Figma design and Notion"
        core = BaseCore(...)  # Initialize with the necessary components.
        updated_status = update_status(
            instruction, previous_steps, plan_step, summary, status, core
        )
    """
    previous_steps = "\n".join(previous_steps)

    plan_prompt = PromptTemplate.from_template(
        "Below is a summary of your characteristics."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Previous steps for the above instruction: {previous_steps}\n"
        + "Given the statements above, "
        + "is there anything that you should remember as you plan for: {plan_step}\n"
        + "Write the response from your perspective."
    )
    chain = core.chain(prompt=plan_prompt)
    plan_result = chain.run(
        summary=summary,
        instruction=instruction,
        previous_steps=previous_steps,
        plan_step=plan_step,
    ).strip()

    thought_prompt = PromptTemplate.from_template(
        "Below is a summary of your characteristics."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Previous steps for the above instruction: {previous_steps}\n"
        + "Given the statements above, how might we summarize "
        + "your thoughts about the plan up till now?\n"
        + "Write the response from your perspective."
    )
    chain = core.chain(prompt=thought_prompt)
    thought_result = chain.run(
        summary=summary, instruction=instruction, previous_steps=previous_steps
    ).strip()

    plan_and_thought = (plan_result + " " + thought_result).replace("\n", "")

    status_prompt = PromptTemplate.from_template(
        "Below is a summary of your characteristics."
        + "{summary}\n\n"
        + "Instruction: {instruction}\n"
        + "Your status from the previous step: {status}\n\n"
        + "Your thoughts at the end of the previous step: {plan_and_thought}\n\n"
        + "Given the above, write a status "
        + "that reflects your status at the end of the previous step. "
        + "Write this in third-person talking about yourself."
        + "Follow this format below:\nStatus: <new status>"
    )
    chain = core.chain(prompt=status_prompt)
    status = chain.run(
        summary=summary,
        instruction=instruction,
        status=status,
        plan_and_thought=plan_and_thought,
    ).strip()

    return status


def generate_refined_plan_step(
    instruction: str,
    previous_steps: List[str],
    plan_step: str,
    summary: str,
    core: BaseCore,
    k: int = 1,
    # llm_kwargs: Dict[str, Any] = {"max_tokens": 3000, "temperature": 0.8},
) -> List[str]:
    """Generate a refined plan by incorporating new plan substep(s) given the current plan step and previous steps.

    This function takes an original instruction, a list of previous plan steps, a new/current plan step,
    a summary, and a BaseCore component to refine the current step in the existing plan.

    Args:
        instruction (str): The original instruction related to the plan.
        previous_steps (List[str]): A list of previously generated plan steps.
        plan_step (str): The new/current plan step.
        summary (str): A summary of relevant characteristics or context.
        core (BaseCore): The agent's core component.
        k (int, optional): The number of alternative refined plans to generate. Default is 1.

    Returns:
        List[str]: A list of steps representing the refined plan, including the new step.

    Example:
        instruction = "Prepare for a business presentation."
        previous_steps = ["1) Research market trends.", "2) Create a product demo."]
        plan_step = "3) Prepare presentation slides."
        summary = "Key points: product features, market analysis, competition."
        core_instance = BaseCore(...)  # Initialize with the necessary components.
        refined_steps = generate_refined_plan(
            instruction, previous_steps, plan_step, summary, core_instance, k=2
        )
    """
    previous_steps = "\n".join(previous_steps)

    prompt = PromptTemplate.from_template(
        "Below is a summary of your characteristics."
        + "{summary}\n\n"
        + "Instruction: \n{instruction}\n\n"
        + "Previous steps in the plan for the above instruction: \n"
        + "{previous_steps}\n\n"
        + "Given the instruction and the steps taken thus far above, "
        + "should you generate more detailed substeps for the current step: {plan_step}?\n"
        + "If no substeps are required, simply generate <NO_SUBSTEPS_REQUIRED>. "
        + "If the instruction is unclear, simply generate <NO_SUBSTEPS_REQUIRED>. "
        + "If substeps are required, return the list of substeps. "
        + "Keep the steps descriptive and concise."
        + "Output format example: \n"
        + "1) <first substep>\n"
        + "2) <second substep>\n"
        + "3) <third substep>\n"
    )
    chain = core.chain(prompt=prompt)

    results = []
    for _ in range(k):
        result = chain.run(
            summary=summary,
            instruction=instruction,
            previous_steps=previous_steps,
            plan_step=plan_step,
        ).strip()
        results.append(result)

    if k == 1:
        results = results[0]
    else:
        # Filter out results with no substeps required.
        results = [
            result for result in results if "<NO_SUBSTEPS_REQUIRED>" not in result
        ]
        if not results:
            return ["<NO_SUBSTEPS_REQUIRED>"]

        plans = [f"Sub-Plan {i}:\n{result}\n\n" for i, result in enumerate(results)]

        prompt = PromptTemplate.from_template(
            "Instruction: \n{instruction}\n\n"
            + "Previous steps in the plan for the above instruction: \n"
            + "{previous_steps}\n\n"
            + "The current plan step is: {plan_step}\n\n"
            + "Below are {k} different sub-plans for the instruction above. "
            + "Combine them into 1 plan such that the plan best answers the instruction."
            + "{plans}\n\n"
            + "Output format example: \n"
            + "1) <first substep>\n"
            + "2) <second substep>\n"
            + "3) <third substep>\n"
        )
        chain = core.chain(prompt=prompt)
        results = chain.run(
            instruction=instruction,
            previous_steps=previous_steps,
            plan_step=plan_step,
            k=k,
            plans=plans,
        ).strip()

    results = parse_numbered_list(results)

    return results
