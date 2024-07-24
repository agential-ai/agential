"""LATS Agent strategies for QA."""
import logging

from typing import Any, Dict
from agential.agential.eval.em import EM
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.lats.functional import generate_prompt, upward_traversal, get_samples, collect_trajectory
from agential.cog.lats.memory import Node
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline

from langchain_community.docstore.wikipedia import Wikipedia


class LATSQAStrategy(LATSBaseStrategy):

    def __init__(self, llm ,docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),):
        super().__init__(llm)
        self.failed_traj = []
        self.docstore = docstore


    def generate(self , node , prompt_sample):
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
        new_state = node.state.copy()  # Make a copy of the parent node's state

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
            new_state['thought'] = thought_line
            new_state['action'] = f"{action_type}[{query}]"

            new_state['observation'] = obs

            new_node = Node(state=new_state, question=node.question, parent=node)

            new_node.is_terminal = reward == 1 or done
            new_node.reward = reward
            new_node.depth = node.depth + 1
    
            unique_key = f"{thought_line}::{new_state['action']}"            
            
            node_dict[unique_key] = new_node  # Add this state to unique_states


            if new_node.is_terminal and reward == 0:
                trajectory = collect_trajectory(new_node)
                #print(trajectory)
                #if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                self.failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{query}]"})

        return list(unique_states.values())  # Return unique nodes as a list





        for action in sampled_actions:

            thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
            action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

            # Use thought and action to form a unique key
            unique_key = f"{thought_line}::{action_line}"
            
            if unique_key in unique_states:
                continue  # Skip if this state already exists

            
            if action_line:
                action_type = action_line.split('[')[0] if '[' in action_line else action_line
                action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

                obs, r, done, info = env.step(f"{action_type.lower()}[{action_param}]")

                # Update the new state dictionary
                new_state = {
                    'thought': thought_line,
                    'action': action_line,
                    'observation': obs
                }

                new_node = Node(state=new_state, question=node.question, parent=node)
                new_node.is_terminal = r == 1 or done
                new_node.reward = r
                new_node.depth = node.depth + 1

                unique_states[unique_key] = new_node  # Add this state to unique_states

                if new_node.is_terminal and r == 0:
                    traversed_nodes = upward_traversal(new_node)
                    self.failed_trajectories.append({'trajectory': traversed_nodes, 'final_answer': f"{action_type.lower()}[{action_param}]"})

        return list(unique_states.values())
    
    def env_step(self,action_type):
        if action_type.lower() == "finish":
            self._answer = query
            self._finished = True
            obs = query
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)




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

    def reflect_node(self):
        pass

    def reset(self):
        pass
