"""LATS general strategy."""

import re

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.docstore.wikipedia import Wikipedia

from agential.cog.lats.functional import (
    _build_failed_trajectory_format,
    _build_reflection_format,
    _prompt_agent,
    _prompt_reflection,
    _prompt_value,
    get_unique_trajectories,
)
from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSReActOutput, LATSSimulationOutput
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.eval.em import EM
from agential.llm.llm import BaseLLM
from agential.utils.docstore import DocstoreExplorer
from agential.utils.general import get_token_cost_time
from agential.utils.parse import remove_newline