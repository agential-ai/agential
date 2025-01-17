"""
def __init__(
        self,
        llm: BaseLLM,
        testing: bool = False,
    ) -> None:
        self.llm = llm
        self.testing = testing

    @abstractmethod
    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:

    @abstractmethod
    def reset(
        self, 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
"""