from typing import Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.base import BaseAgent
from agential.cog.strategies.strategy_factory import CriticStrategyFactory

from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_POT_INSTRUCTION_TEST_HUMANEVAL,
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT_TEST,
)

class CriticAgent(BaseAgent):
    """CRITIC Agent.

    Attributes:
        llm (BaseChatModel): An instance of a language model used for generating initial answers
            and critiques.
        mode (str): The CRITIC agent's mode. Can be "qa", "math", or "code".
    """

    def __init__(
        self,
        llm: BaseChatModel,
        mode: str,
        **strategy_kwargs
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm
        self.mode = mode

        self.strategy = CriticStrategyFactory().get_strategy(self.mode, llm=self.llm, **strategy_kwargs)

    def generate(
        self,
        question: str,
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt: str = CRITIC_INSTRUCTION_HOTPOTQA,
        critique_examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        critique_prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        additional_keys: Dict[str, str] = {},
        critique_additional_keys: Dict[str, str] = {},
        max_interactions: int = 7,
        use_search_tool: bool = True,
        use_interpreter_tool: bool = True,
        reset: bool = True,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Generates an answer that is refined with search results.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the initial answer. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_COT.
            prompt (str): The instruction template used to prompt the language model for the initial answer. Defaults to CRITIC_INSTRUCTION_HOTPOTQA.
            critique_examples (str): Few-shot examples to guide the language model in generating critiques. Defaults to HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC.
            critique_prompt (str): The instruction template for generating critiques. Defaults to CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA.
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique_prompt. Defaults to {}.
            max_interactions (int): The maximum number of critique cycles. Defaults to 7.
            use_search_tool (bool): Only for "qa" mode. Flag to decide whether to use the search tool for evidence gathering. Defaults to True.
            use_interpreter_tool (bool): Only for "math" or "code" mode. Flag to decide whether to use the interpreter tool for code execution. Defaults to True.
            **kwargs: Additional parameters for flexibility.
            
        Returns:
            List[Dict[str, str]]: A list of dictionaries.
                "qa" mode:
                    - Each dictionary contains an "answer" and "critique". Optionally, a
                    dictionary may include the search "query" and "search_result", and the final dictionary includes the final "revised_answer".
                "math" mode:
                    - Each dictionary contains "code" and "critique". Optionally, a dictionary may include
                    the "execution_status" and "code_answer" if use_interpreter_tool is True. If the critic
                    improves the solution, then the dictionary will have an "improved_code" key.
        """
        if reset:
            self.reset()

        out = []

        # Initial answer generation
        answer = self.strategy.generate(question, examples, prompt, additional_keys)

        critique = ""
        for idx in range(max_interactions):
            critique, external_tool_info = self.strategy.generate_critique(
                idx=idx,
                question=question, 
                examples=critique_examples, 
                answer=answer, 
                critique=critique,
                prompt=critique_prompt, 
                additional_keys=critique_additional_keys, 
                use_search_tool=use_search_tool,
                max_interactions=max_interactions,
                **kwargs
            )

            out.append(self.strategy.create_output_dict(answer, critique, external_tool_info))

            if self.strategy.halting_condition(critique):
                break

            # Update answer for the next iteration.
            answer = self.strategy.update_answer_based_on_critique(
                question=question, 
                examples=critique_examples, 
                answer=answer, 
                critique=critique,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
                **kwargs
            )

        return out

    def reset(self):
        self.strategy.reset()