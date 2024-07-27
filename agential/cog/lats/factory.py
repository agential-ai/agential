"""LATS prompts and fewshot examples selector."""


from agential.base.factory import BaseFactory
from agential.cog.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType

from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.lats.strategies.qa import (
    LATSFEVERStrategy,
    LATSHotQAStrategy,
    LATSTriviaQAStrategy,
    LATSAmbigNQStrategy
)