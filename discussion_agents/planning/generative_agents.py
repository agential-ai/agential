"""Planning module for Generative Agents."""
from typing import List, Optional, Dict, Any
import string
import random
from datetime import datetime, timedelta

from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BaseMemory, BaseRetriever

from langchain.prompts import PromptTemplate


from discussion_agents.utils.parse import parse_list
from discussion_agents.utils.format import format_memories_detail
from discussion_agents.utils.fetch import fetch_memories
from discussion_agents.utils.constants import HOUR_STR

def generate_daily_req(
    current_day: datetime, 
    summary: str,
    lifestyle: str,
    name: str, 
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
    wake_up_hour: Optional[int] = 8
) -> List[str]:
    prompt = PromptTemplate.from_template(
        "{summary}\n"
        + "In general, {name}'s lifestyle: {lifestyle}\n"
        + "Today is {current_day}. "
        + "Here is {name}'s plan today in broad-strokes "
        + "(with the time of the day. e.g., have lunch at 12:00 pm, watch TV from 7 to 8 pm): "
        + "1) wake up and complete the morning routine at {wake_up_hour}, "
        + "2)\n"
        + "Provide each step on a new line.\n"
    )
    chain = LLMChain(
        llm=llm,
        llm_kwargs=llm_kwargs,
        prompt=prompt,
        memory=memory
    )
    kwargs = dict(
        summary=summary,
        lifestyle=lifestyle,
        name=name,
        current_day=current_day.strftime("%A %B %d"),
        wake_up_hour=wake_up_hour,
    )
    result = parse_list(chain.run(**kwargs).strip())
    result = [
        f"1) wake up and complete the morning routine at {wake_up_hour}:00 am"
    ] + result
    result = [s.split(")")[-1].rstrip(",.").strip() for s in result]

    return result

def update_status(
    current_day: datetime,
    name: str,
    status: str,
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
    memory_retriever: BaseRetriever,
):
    current_day_str = current_day.strftime("%A %B %d, %Y")
    focal_points = [
        f"{name}'s plan for {current_day_str}.",
        f"Important recent events for {name}'s life.",
    ]

    relevant_context = []
    for focal_point in focal_points:
        fetched_memories = fetch_memories(memory_retriever, focal_point)
        relevant_context.append(
            format_memories_detail(fetched_memories)
        )
    relevant_context = "\n".join(relevant_context)

    plan_prompt = PromptTemplate.from_template(
        relevant_context
        + "\n"
        + "Given the statements above, "
        + "is there anything that {name} should remember as they plan for"
        + " *{current_day_str}*? "
        + "If there is any scheduling information, be as specific as possible "
        + "(include date, time, and location if stated in the statement)\n\n"
        + "Write the response from {name}'s perspective."
    )
    chain = LLMChain(
        llm=llm,
        llm_kwargs=llm_kwargs,
        prompt=plan_prompt,
        memory=memory
    )
    plan_kwargs = dict(name=name, current_day_str=current_day_str)
    plan_result = chain.run(**plan_kwargs).strip()

    thought_prompt = PromptTemplate.from_template(
        relevant_context
        + "\n"
        + "Given the statements above, how might we summarize "
        + "{name}'s feelings about their days up to now?\n\n"
        + "Write the response from {name}'s perspective."
    )
    chain = LLMChain(
        llm=llm,
        llm_kwargs=llm_kwargs,
        prompt=thought_prompt,
        memory=memory
    )
    thought_kwargs = dict(name=name)
    thought_result = chain.run(**thought_kwargs).strip()

    status_prompt = PromptTemplate.from_template(
        "{name}'s status from "
        + "{previous_day}:\n"
        + "{status}\n\n"
        + "{name}'s thoughts at the end of "
        + "{previous_day}:\n"
        + (plan_result + " " + thought_result).replace("\n", "")
        + "\n\n"
        + "It is now {current_day}. "
        + "Given the above, write {name}'s status for "
        + "{current_day} that reflects {name}'s "
        + "thoughts at the end of {previous_day}. "
        + "Write this in third-person talking about {name}."
        + "If there is any scheduling information, be as specific as possible "
        + "(include date, time, and location if stated in the statement).\n\n"
        + "Follow this format below:\nStatus: <new status>"
    )
    status_kwargs = dict(
        name=name,
        previous_day=(current_day - timedelta(days=1)).strftime("%A %B %d, %Y"),
        status=status,
        current_day=current_day.strftime("%A %B %d, %Y"),
    )
    chain = LLMChain(
        llm=llm,
        llm_kwargs=llm_kwargs,
        prompt=status_prompt,
        memory=memory
    )
    status = chain.run(**status_kwargs).strip()

    return status

def update_daily_plan_req(
    current_day: datetime,
    name: str,
    summary: str,
    llm: BaseLanguageModel,
    llm_kwargs: Dict[str, Any],
    memory: BaseMemory,
):
    daily_plan_req_prompt = PromptTemplate.from_template(
        summary
        + "\n"
        + "Today is {current_day}. "
        + "Here is {name}'s plan today in broad-strokes "
        + "(with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).\n\n"
        + "Follow this format (the list should have 4~6 items but no more):\n"
        + "1. wake up and complete the morning routine at <time>, 2. ..."
    )
    daily_plan_req_kwargs = dict(
        current_day=current_day.strftime("%A %B %d, %Y"), name=name
    )
    chain = LLMChain(
        llm=llm,
        llm_kwargs=llm_kwargs,
        prompt=daily_plan_req_prompt,
        memory=memory
    )
    daily_plan_req = (
        chain
        .run(**daily_plan_req_kwargs)
        .strip()
        .replace("\n", " ")
    )

    return daily_plan_req

def generate_hourly_schedule(
    n_m1_activity: List[str], 
    curr_hour_str: str, 
    current_day: datetime,
    name: str,
    daily_req: str,
    summary: str,
    llm: BaseLanguageModel,
    memory: BaseMemory,
) -> str:
    current_day = current_day.strftime("%A %B %d, %Y")
    schedule_format = ""
    for i in HOUR_STR:
        schedule_format += f"[{current_day} -- {i}]"
        schedule_format += f" Activity: [Fill in]\n"
    schedule_format = schedule_format[:-1]

    intermission_str = f"Here's the originally intended hourly breakdown of"
    intermission_str += f" {name}'s schedule today: "
    for count, i in enumerate(daily_req):
        intermission_str += f"{str(count+1)}) {i}, "
    intermission_str = intermission_str[:-2]

    prior_schedule = "\n"
    for count, i in enumerate(n_m1_activity):
        prior_schedule += f"[{current_day} --"
        prior_schedule += f" {HOUR_STR[count]}] Activity:"
        prior_schedule += f" {name}"
        prior_schedule += f" is {i}\n"

    prompt = PromptTemplate.from_template(
        schedule_format
        + "==="
        + summary
        + "\n"
        + prior_schedule
        + "\n"
        + intermission_str
        + "\n"
        + "[{current_day}"
        + " -- {curr_hour_str}] Activity:"
        + " {name} is"
    )
    prompt_kwargs = dict(
        current_day=current_day, curr_hour_str=curr_hour_str, name=name
    )
    llm_kwargs = dict(
        max_tokens=50,
        temperature=0.5,
    )
    chain = LLMChain(
        llm=llm,
        llm_kwargs=llm_kwargs,
        prompt=prompt,
        memory=memory
    )
    result = (
        chain
        .run(**prompt_kwargs, stop="\n")
        .rstrip(".")
        .strip()
    )
    return result

def generate_hourly_schedule_k(
    current_day: datetime, 
    name: str,
    daily_req: str,
    summary: str,
    llm: BaseLanguageModel,
    memory: BaseMemory,
    k: int = 3,  # Diversity count.
    wake_up_hour: Optional[int] = 8,
) -> List[str]:
    n_m1_activity = []
    for i in range(k):
        if len(set(n_m1_activity)) < 5:  # Number of unique activities < 5.
            n_m1_activity = []
            for curr_hour_str in HOUR_STR:
                if wake_up_hour > 0:
                    n_m1_activity += ["sleeping"]
                    wake_up_hour -= 1
                else:
                    n_m1_activity += [
                        generate_hourly_schedule(
                            n_m1_activity=n_m1_activity, 
                            curr_hour_str=curr_hour_str, 
                            current_day=current_day,
                            name=name,
                            daily_req=daily_req,
                            summary=summary,
                            llm=llm,
                            memory=memory,
                        )
                    ]

    # Step 1. Compressing the hourly schedule to the following format:
    # The integer indicates the number of hours. They should add up to 24.
    # [['sleeping', 6], ['waking up and starting her morning routine', 1],
    # ['eating breakfast', 1], ['getting ready for the day', 1],
    # ['working on her painting', 2], ['taking a break', 1],
    # ['having lunch', 1], ['working on her painting', 3],
    # ['taking a break', 2], ['working on her painting', 2],
    # ['relaxing and watching TV', 1], ['going to bed', 1], ['sleeping', 2]]
    _n_m1_hourly_compressed = []
    prev, prev_count = None, 0
    for i in n_m1_activity:
        if i != prev:
            prev_count = 1
            _n_m1_hourly_compressed += [[i, prev_count]]
            prev = i
        elif _n_m1_hourly_compressed:
            _n_m1_hourly_compressed[-1][1] += 1

    # Step 2. Expand to min scale (from hour scale)
    # [['sleeping', 360], ['waking up and starting her morning routine', 60],
    # ['eating breakfast', 60],..
    n_m1_hourly_compressed = []
    for task, duration in _n_m1_hourly_compressed:
        n_m1_hourly_compressed += [[task, duration * 60]]

    return n_m1_hourly_compressed
















# def long_term_planning(
#     new_day: str, 
#     current_day: datetime, 
#     summary: str,
#     lifestyle: str,
#     name: str, 
#     status: str,
#     llm: BaseLanguageModel,
#     memory: GenerativeAgentMemory,
#     memory_retriever: TimeWeightedVectorStoreRetriever,
#     llm_kwargs: Dict[str, Any],
#     wake_up_hour: int = 8,
# ):
#     # When it is a new day, we start by creating the daily_req of the persona.
#     # Note that the daily_req is a list of strings that describe the persona's
#     # day in broad strokes.
#     if new_day == "First day":
#         # Bootstrapping the daily plan for the start of then generation:
#         # if this is the start of generation (so there is no previous day's
#         # daily requirement, or if we are on a new day, we want to create a new
#         # set of daily requirements.
#         daily_req = generate_daily_req(
#             current_day=current_day, 
#             wake_up_hour=wake_up_hour,
#             summary=summary,
#             lifestyle=lifestyle,
#             name=name, 
#             llm=llm,
#             llm_kwargs=llm_kwargs,
#             memory=memory,
#             wake_up_hour=wake_up_hour,
#         )
#     elif new_day == "New day":
#         status, daily_plan_req = update_status_and_daily_plan_req(
#             current_day=current_day,
#             name=name,
#             status=status,
#             summary=summary,
#             llm=llm,
#             llm_kwargs=llm_kwargs,
#             memory=memory,
#             memory_retriever=memory_retriever,
#         )

#     # Based on the daily_req, we create an hourly schedule for the persona,
#     # which is a list of todo items with a time duration (in minutes) that
#     # add up to 24 hours.
#     f_daily_schedule = generate_hourly_schedule_top_k(
#         current_day=current_day, 
#         wake_up_hour=wake_up_hour,
#         name=name,
#         daily_req=daily_req,
#         summary=summary,
#         llm=llm,
#         memory=memory,
#     )


#     if new_day == "First day":
#         return daily_req, f_daily_schedule
#     return status, daily_plan_req, f_daily_schedule




    # Added March 4 -- adding plan to the memory.
    # thought = (
    #     f"This is {self.name}'s plan for {current_day.strftime('%A %B %d, %Y')}:"
    # )
    # for i in self.daily_req:
    #     thought += f" {i},"
    # thought = thought[:-1] + "."

    # self.memory.save_context(
    #     {},
    #     {
    #         self.memory.add_memory_key: thought,
    #         self.memory.now_key: current_day,
    #     },
    # )