"""Unit tests for ReAct functional module."""

import tiktoken

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.functional.react import (
    _build_agent_prompt,
    _is_halted,
    _prompt_agent,
)
from agential.cog.prompts.react import HOTPOTQA_FEWSHOT_EXAMPLES


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt function."""
    prompt = _build_agent_prompt(
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES,
        max_steps=1,
    )

    gt_out = (
        "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n"
        "(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n"
        "(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.\n"
        "(3) Finish[answer], which returns the answer and finishes the task.\n"
        "You have a maximum of 1 steps.\n\n"
        "Here are some examples:\n"
        "Question: Serianna is a band of what genre that combines elements of heavy metal and hardcore punk?\n"
        "Thought 1: Let's search the question in google\n"
        "Action 1: Search[Serianna is a band of what genre that combines elements of heavy metal and hardcore punk? site: wikipedia.org]\n"
        "Observation 1: [Metalcore - Wikipedia] Metalcore is a fusion music genre that combines elements of extreme metal and hardcore punk.\n"
        "Thought 2: The evidence suggests that metalcore is a genre that combines elements of extreme metal and hardcore punk.\n"
        "Action 2: Search[Serianna is a band of metalcore genre. site: wikipedia.org\n"
        "Observation 2: [Serianna - Wikipedia] Serianna was a metalcore band from Madison, Wisconsin. The band formed in 2006...\n"
        "Thought 3: The evidence suggests Serianna is a metalcore band.\n"
        "Action 3: Finish[Metalcore]\n"
        "Question: Which band was formed first, Helium or Jack's Mannequin?\n"
        "Thought 1: Let's search the question in google\n"
        "Action 1: Search[Which band was formed first, Helium or Jack's Mannequin?]\n"
        "Observation 1: [Jack's Mannequin - Wikipedia] Jack's Mannequin was an American rock band formed in 2004, hailing from Orange County, California.\n"
        "Thought 2: The evidence shows that Jack's Mannequin is a band formed in 2004. We then find out when the band Helium was formed.\n"
        "Action 2: Search[When was the band 'Helium' formed?]\n"
        "Observation 2: [] Helium / Active from 1992\n"
        "Thought 3: The evidence shows that Helium was formed in 1992. Jack's Mannequin was formed in 2004. 1992 (Helium) < 2004 (Jack's Mannequin), so Helium was formed first.\n"
        "Action 3: Finish[Helium]\n"
        "Question: What year did Maurice win the award given to the 'player judged most valuable to his team' in the NHL?\n"
        "Thought 1: Let's search the question in google:\n"
        "Action 1: Search[What year did Maurice win the award given to the 'player judged most valuable to his team' in the NHL? site: wikipedia.org]\n"
        "Observation 1: [List of National Hockey League awards - Wikipedia] Awarded to the 'player judged most valuable to his team'. The original trophy was donated to the league by Dr. David A. Hart, father of coach Cecil Hart.\n"
        "Thought 2: The evidence does not provide information about what the award is and Maurice won the award in which year. We can change the search query.\n"
        "Action 2: Search[What year did Maurice win the award of most valuable player in the NHL?]\n"
        "Observation 2: [NHL Maurice Richard Trophy Winners] Award presented to top goal-scorer annually since 1999. It honors Richard, the first player in League history to score 50 goals in 50 games, 50 goals in a...\n"
        "Thought 3: The evidence mention Richard won NHL Trophy, but does not mention if it is for most valuable players.\n"
        "Action 3: Search[When Maurice Richard win the NHL's most valuable player?]\n"
        "Observation 3: [Maurice Richard - Wikipedia] He won the Hart Trophy as the NHL's most valuable player in 1947, played in 13 All-Star Games and was named to 14 post-season NHL All-Star teams, eight on the first team.\n"
        "Thought 4: The evidence shows that Maurice Richard won the Hart Trophy as the NHL's most valuable player in 1947.\n"
        "Action 4: Finish[1947]\n"
        "Question: Are John H. Auer and Jerome Robbins both directors?\n"
        "Thought 1: Let's search the question in google\n"
        "Action 1: Search[Are John H. Auer and Jerome Robbins both directors?]\n"
        "Observation 1: [A history of Jerome Robbins at PNB - Pacific Northwest Ballet] Robbins retained the title of associate artistic director until approximately 1963, ... Ballets: USA, from the late 1950s to the late 1960s.\n"
        "Thought 2: The evidence suggests Jerome Robbins is a director. We then need to verify if John H. Auer is a director.\n"
        "Action 2: Search[Is John H. Auer a director? site: wikipedia.org]\n"
        "Observation 2: [John H. Auer - Wikipedia] Auer was a Hungarian-born child actor who, on coming to the Americas in 1928, became a movie director and producer, initially in Mexico but, from the early 1930s, in Hollywood.\n"
        "Thought 3: The evidence suggests that John H. Auer is an actor, director and producer. Therefore, both John H. Auer and Jerome Robbins are directors.\n"
        "Action 3: Finish[Yes]\n"
        "Question: Which artist did Anthony Toby 'Tony' Hiller appear with that liked showering himself (and others) with confetti?\n"
        "Thought 1: Let's search the question in google\n"
        "Action 1: Search[Which artist did Anthony Toby Tony Hiller appear with that liked showering himself (and others) with confetti?]\n"
        "Observation 1: [Untitled] Without you: The tragic story of Badfinger|Dan Matovina, The Military Orchid and Other Novels|Jocelyn Brooke, Looking at Lisp (Micro computer books)|Tony...\n"
        "Thought 2: The evidence does not provide any useful information about the question. We need to find out who is the artist that liked showering himself (and others) with confetti.\n"
        "Action 2: Search[Which artist liked showering himself (and others) with confetti?]\n"
        "Observation 2: [Rip Taylor - Wikipedia] Charles Elmer 'Rip' Taylor Jr. was an American actor and comedian, known for his exuberance and flamboyant personality, including his wild moustache, toupee, and his habit of showering himself (and others)\n"
        "Thought 3: The evidence suggests that the artist that liked showering himself is Charles Elmer 'Rip' Taylor Jr. We can further check if Rip Taylor appeared with Anthony Toby 'Tony' Hiller.\n"
        "Action 3: Search[Which artist appeared with Anthony Toby 'Tony' Hiller?]\n"
        "Observation 3: [Tony Hiller - Wikipedia] He was best known for writing and/or producing hits for Brotherhood of Man, including 'United We Stand' (1970) and 'Save Your Kisses for Me' (1976). Biography [edit]\n"
        "Thought 4: The evidence does not mention the artist.\n"
        "Action 4: Search[Did Tony Hiller appear with Rip Taylor?]\n"
        "Observation 4: [Tony Hiller - Wikipedia] The Hiller Brothers appeared with many performers of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea...\n"
        "Thought 5: The evidence shows that Tony Hiller appeared with Rip Taylor.\n"
        "Action 5: Finish[Rip Taylor]\n"
        "(END OF EXAMPLES)\n\n"
        "Question: "
    )
    print(prompt)

    assert isinstance(prompt, str)
    assert prompt == gt_out

    gt_out = "  examples 1"
    out = _build_agent_prompt(
        question="",
        scratchpad="",
        examples="examples",
        max_steps=1,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert out == gt_out


def test__prompt_agent() -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES,
        max_steps=1,
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test with custom prompt template string.
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES,
        max_steps=1,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert isinstance(out, str)
    assert out == "1"


def test__is_halted() -> None:
    """Test _is_halted function."""
    gpt3_5_turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Test when finish is true.
    assert _is_halted(
        True,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES,
        10,
        100,
        gpt3_5_turbo_enc,
    )

    # Test when step_n exceeds max_steps.
    assert _is_halted(
        False,
        11,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES,
        10,
        100,
        gpt3_5_turbo_enc,
    )

    # Test when encoded prompt exceeds max_tokens.
    assert _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES,
        10,
        10,
        gpt3_5_turbo_enc,
    )

    # Test when none of the conditions for halting are met.
    assert not _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES,
        10,
        100000,
        gpt3_5_turbo_enc,
    )

    # Test edge case when step_n equals max_steps.
    assert _is_halted(
        False,
        10,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES,
        10,
        100,
        gpt3_5_turbo_enc,
    )

    # Test edge case when encoded prompt equals max_tokens.
    assert _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES,
        10,
        1603,
        gpt3_5_turbo_enc,
    )

    # Test with custom prompt template string.
    assert not _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES,
        10,
        1603,
        gpt3_5_turbo_enc,
        "{question} {scratchpad} {examples} {max_steps}",
    )
