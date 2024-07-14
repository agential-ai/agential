"""ExpeL Agent strategies for QA."""


from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.expel.strategies.base import ExpeLBaseStrategy

class ExpeLQAStrategy(ExpeLBaseStrategy):
    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)

    def get_dynamic_examples(self):
        pass

    def gather_experience(self):
        pass

    def extract_insights(self):
        pass

    def update_insights(self):
        pass

