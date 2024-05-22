from typing import Dict, Optional
from agential.cog.functional.critic import _prompt_agent, _prompt_critique
from agential.cog.strategies.critic.base import CriticBaseStrategy
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

class QAStrategy(CriticBaseStrategy):
    def __init__(self, llm, search: Optional[GoogleSerperAPIWrapper] = None, evidence_length: int = 400, num_results: int = 8):
        self.llm = llm
        self.search = search
        self.evidence_length = evidence_length
        self.num_results = num_results

        self._query_history = []
        self._evidence_history = set()
        self._halt = False

    def generate(
        self, 
        question: str, 
        examples: str, 
        prompt: str, 
        additional_keys: Dict[str, str],
    ) -> str:
        return _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            additional_keys=additional_keys,
            prompt=prompt,
        )

    def generate_critique(
        self, 
        idx: int,
        question: str, 
        examples: str, 
        answer: str, 
        critique: str,
        prompt: str, 
        additional_keys: Dict[str, str], 
        use_search_tool: bool, 
        max_interactions: int,
        **kwargs
    ):
        external_tool_info = {}
        new_critique = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=critique,
            additional_keys=additional_keys,
            prompt=prompt,
        ).split("> Evidence: ")[0]

        if "> Search Query: " in new_critique:
            _, search_query = new_critique.split("> Search Query:")[:2]
            search_query = search_query.split("\n")[0].strip()

            search_result, context = self.handle_search_query(
                idx, 
                question, 
                search_query, 
                use_search_tool, 
                max_interactions, 
                **kwargs
            )
            new_critique = f"{critique}\n{new_critique}{context}"
            if not use_search_tool:
                search_result = _prompt_critique(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    answer=answer,
                    critique=new_critique,
                    additional_keys=additional_keys,
                    prompt=prompt,
                ).split("> Evidence: ")[0]
                new_critique = f"{critique}\n{new_critique}{search_result.strip()}"
            external_tool_info['search_query'] = search_query
            external_tool_info['search_result'] = search_result
        else:
            if "most possible answer: " not in new_critique:
                new_critique = f"{critique}\n{new_critique}\nLet's give the most possible answer.\n\nQuestion: {question}\nHere's "
                new_critique = _prompt_critique(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    answer=answer,
                    critique=new_critique,
                    additional_keys=additional_keys,
                    prompt=prompt,
                ).split("> Evidence: ")[0]

            new_critique = new_critique.split("most possible answer: ")[-1].strip()
            self._halt = True

        return new_critique, external_tool_info

    def create_output_dict(self, answer: str, critique: str, external_tool_info: Dict[str, str]) -> Dict[str, str]:
        output_dict = {"answer": answer if not self._halt else critique, "critique": critique, **external_tool_info}
        return output_dict
    
    def update_answer_based_on_critique(
        self, 
        question: str, 
        examples: str, 
        answer: str, 
        critique: str, 
        prompt: str, 
        additional_keys: Dict[str, str], 
        external_tool_info: Dict[str, str],
        **kwargs
    ) -> str:
        return answer

    def halting_condition(self, critique: str) -> bool:
        if self._halt:
            self._halt = False
            return True
        return False

    def reset(self):
        self._query_history = []
        self._evidence_history = set()
        self._halt = False

    def handle_search_query(self, idx, question, search_query, use_search_tool, max_interactions, **kwargs):
        evidence_length = kwargs.get('evidence_length', self.evidence_length)
        num_results = kwargs.get('num_results', self.num_results)
        
        if use_search_tool:
            if not self.search:
                raise ValueError("Search tool is required but not provided.")
            
            self._query_history.append(search_query)
            count = self._query_history.count(search_query)
            start = count if count < num_results else num_results - 1

            for k in range(start, num_results):
                search_result = self.search.results(search_query, num_results=k)[-1]
                if "snippet" in search_result and search_result['snippet'] not in self._evidence_history:
                    self._evidence_history.add(search_result['snippet'])
                    break

            if "title" not in search_result and "snippet" not in search_result:
                context = f"""> Evidence: [] No results found\n\n"""
            else:
                context = f"""> Evidence: [{search_result['title']}] {search_result['snippet'][:evidence_length]}\n\n"""
            if idx == max_interactions - 2:
                context += f"Let's give the most possible answer.\n\nQuestion: {question}\nHere's "
        else:
            search_result = {}
            context = """> Evidence: """
        return search_result, context