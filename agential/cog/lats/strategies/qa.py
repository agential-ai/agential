"""LATS Agent strategies for QA."""

from typing import Dict
from agential.eval.em import EM
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.lats.functional import (
    generate_prompt, 
    upward_traversal, 
    get_samples, 
    get_unique_trajectories,
    _prompt_reflection
)
from agential.cog.lats.functional import Node
from agential.cog.lats.prompts import HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT, LATS_REFLECT_INSTRUCTION_HOTPOTQA
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline

from langchain_community.docstore.wikipedia import Wikipedia


class LATSQAStrategy(LATSBaseStrategy):

    def __init__(self, llm ,docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),):
        super().__init__(llm)
        self.failed_trajectories = []
        self.reflection_map = []
        self.docstore = docstore

    def generate(self):
        pass


    def generate_(self , node , prompt_sample):
        traversed_nodes = upward_traversal(node)
        prompt = node.question
        trajectory = generate_prompt(traversed_nodes)
        additional_keys = Dict[str, str] = {}
        sampled_actions = get_samples(prompt, trajectory, f"Thought {node.depth + 1}: ", additional_keys, n, prompt_sample=prompt_sample, stop="Observation")
        
        unique_states = []  # Store unique states here
        for action in sampled_actions:

            thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
            action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)


            
            if action_line:
                action_type = action_line.split('[')[0] if '[' in action_line else action_line
                action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""
            

            unique_states.append((thought_line, action_type , action_param))
        
        return unique_states


    def generate_observation(self, unique_states, node , key):
        node_dict = {}

        for (thought_line , action_type, query) in unique_states:
            reward = 0
            done = False

            if action_type.lower() == "finish":
                obs = query
                if EM(query, key):
                    reward = 1
                done = True
                # done add to Node
            elif action_type.lower() == "search":
                try:
                    search_result = self.docstore.search(query)
                    obs = remove_newline(search_result)
                except Exception:
                    obs = "Could not find that page, please try again."
            elif action_type.lower() == "lookup":
                try:
                    lookup_result = self.docstore.lookup(query)
                    obs = remove_newline(lookup_result)
                except ValueError:
                    obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
            else:
                obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
        
            # Update the new state dictionary
            new_state = {
                'thought': thought_line,
                'action': f"{action_type}[{query}]",
                'observation': obs,
            }

            new_node = Node(state=new_state, question=node.question, parent=node)

            new_node.is_terminal = reward == 1 or done
            new_node.reward = reward
            new_node.depth = node.depth + 1
    
            unique_key = f"{thought_line}::{new_state['action']}"            
            
            node_dict[unique_key] = new_node  # Add this state to unique_states

            if new_node.is_terminal and reward == 0:
                traversed_nodes = upward_traversal(new_node)
                self.failed_trajectories.append({'trajectory': traversed_nodes, 'final_answer': f"{action_type.lower()}[{query}]"})

        return list(unique_states.values())  # Return unique nodes as a list


    def select_node(self):
        pass

    def expand_node(self):
        pass

    def evaluate_node(self):
        pass

    def simulate_node(self):
        pass

    def backpropagate_node(self):
        pass

    def reflect_condition(self, unique_trajectories):
        return len(unique_trajectories) > len(self.reflection_map) and len(unique_trajectories) < 4

    def reflect(self, question, additional_keys):
        unique_trajectories = get_unique_trajectories(self.failed_trajectories)

        reflection_mapping = []
        for trajectory in unique_trajectories:
            reflection = _prompt_reflection(
                self.llm,
                question=question,
                examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
                trajectory=trajectory, 
                prompt=LATS_REFLECT_INSTRUCTION_HOTPOTQA, 
                additional_keys=additional_keys
            )

            reflection_mapping.append({
                'question': question,
                'trajectory': trajectory,
                'reflection': reflection
            })

        self.reflection_map = reflection_mapping

    def reset(self):
        pass
