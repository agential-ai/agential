"""Testing Generative Agents memory scoring method(s)."""
import os
import re

import dotenv

from langchain.llms.huggingface_hub import HuggingFaceHub

from discussion_agents.scoring.generative_agents import score_memories_importance

dotenv.load_dotenv(".env")
huggingface_hub_api_key = os.getenv("HUGGINGFACE_HUB_API_KEY")

llm = HuggingFaceHub(
    repo_id="gpt2",
    huggingfacehub_api_token=huggingface_hub_api_key,
)


def test_score_memories_importance():
    """Test score_memories_importance."""
    importance_weight = 0.15

    # Testing different outputs.

    # Test single memory.
    memory_contents = (
        "Today I didn't eat any lunch or dinner and I'm starving for some chipotle."
    )
    # Queried 10 times.
    outs = ["2", "3", "5", "5", "3;", "3", "1.", "5", "4"]

    for scores in outs:
        # The rest of the code within the function.
        scores = "".join(scores.split(";")).split(";")
        scores_list = [float(x) / 10 * importance_weight for x in scores]
        assert len(scores_list) == 1
        assert type(scores_list[0]) is float
        assert scores_list[0] <= 1 and scores_list[0] >= 0

    # Test multiple memories.
    memory_contents = "Tommie remembers his dog, Bruno, from when he was a kid; Tommie feels tired from driving so far"
    scores = "3; 5"
    scores = re.findall(r"\d+", scores)
    scores_list = [float(x) / 10 * importance_weight for x in scores]
    for score in scores_list:
        assert score >= 0 and score <= 1
        assert type(score) is float

    # Testing the entire function.

    scores = score_memories_importance(
        memory_contents="Some memory.",
        llm=llm,
        importance_weight=importance_weight,
    )
