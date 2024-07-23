"""LATS's memory implementations.

Original Paper: https://arxiv.org/pdf/2310.04406
Paper Repository: https://github.com/lapisrocks/LanguageAgentTreeSearch
"""

import numpy as np

from agential.cog.lats.functional import expand_node, select_node

class Node:
    def __init__(self, state, question, parent=None):
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0

    def uct(self):
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def add_child(self, child):
        self.children.append(child)

class LATSMemory:  # TODO: inherit from BaseMemory
    def __init__(self, question):
        self.root = Node(None, question, parent=None)

    # TODO: should have a function to find a node 
    def expand_node(self, node, prompt_sample, task, n_generate_sample, depth_limit):
        children_nodes = expand_node(
            node=node,
            prompt_sample=prompt_sample,
            task=task,
            n_generate_sample=n_generate_sample,
            depth_limit=depth_limit
        )
        for child in children_nodes:
            node.add_child(child)

    def select_node(self, node):
        return select_node(node)

    # def expand_node(node, prompt_sample, task, n_generate_sample, depth_limit=7):
    #     if node.depth >= depth_limit:
    #         node.is_terminal = True
    #         return []
    #     children_nodes = generate_new_states(node, prompt_sample, task, n_generate_sample)
    #     return children_nodes

    # def select_node(node: Node):
    #     while node and node.children:
    #         terminal_children = [child for child in node.children if child.is_terminal]

    #         if len(terminal_children) == len(node.children):
    #             if node.parent:  
    #                 node.parent.children.remove(node)
    #             node = node.parent  
    #             continue  

    #         for child in terminal_children:
    #             if child.reward == 1:
    #                 return child
            
    #         node = max([child for child in node.children if not child.is_terminal], key=lambda child: child.uct(), default=None)

    #     return node