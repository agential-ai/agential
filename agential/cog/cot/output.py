"""CoT structured output module."""

from pydantic import Field

from agential.cog.base.output import BaseOutput
from agential.llm.llm import Response


class CoTOutput(BaseOutput):
    """Critic Pydantic output class.

    Attributes:
        additional_info (Response): The llm response information.
    """

    additional_info: Response = Field(..., description="The llm response information.")
