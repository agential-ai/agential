"""Functional module for Language Agent Tree Search (LATS)."""

import numpy as np
import openai
import requests


def _build_standard_prompt():
    pass

def _prompt_standard():
    pass

def _build_cot_prompt():
    pass

def _prompt_cot():
    pass

def _build_reflection_prompt():
    pass

def _prompt_reflection():
    pass

def _build_value_prompt():
    pass

def _prompt_value():
    pass


















global reflection_map
global failed_trajectories
reflection_map = []
failed_trajectories = []

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
    
    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, thought={self.state['thought']}, action={self.state['action']}, observation={self.state['observation']})"

def upward_traversal(node):
    nodes = []
    while node:
        nodes.append(node)
        node = node.parent
    return list(reversed(nodes))

def generate_prompt(node):
    traversed_nodes = upward_traversal(node)  # From root to current node.
    trajectory = []

    for node in traversed_nodes:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth > 0:
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))

    return '\n'.join(trajectory)


def select_node(node: Node):
    while node and node.children:
        terminal_children = [child for child in node.children if child.is_terminal]

        if len(terminal_children) == len(node.children):
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  

        for child in terminal_children:
            if child.reward == 1:
                return child
        
        node = max([child for child in node.children if not child.is_terminal], key=lambda child: child.uct(), default=None)

    return node
    
def node_trajectory_to_text(trajectory):
    formatted_lines = []
    for node in trajectory:
        if node.depth > 0:
            if node.state['thought']:
                formatted_lines.append(f"Thought {node.depth}: {node.state['thought']}")
            if node.state['action']:
                formatted_lines.append(f"Action {node.depth}: {node.state['action']}")
            if node.state['observation']:
                formatted_lines.append(f"Observation {node.depth}: {node.state['observation']}")

    return '\n'.join(formatted_lines)

def get_unique_trajectories(failed_trajectories, num=5):
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get('final_answer')
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj['trajectory']))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories

# Function to generate GPT completions
def gpt(prompt, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    outputs = []
    
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
    
    return outputs

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    global failed_trajectories
    global reflection_map
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    if len(unique_trajectories) > len(reflection_map) and len(unique_trajectories) < 4:
        print("generating reflections")
        reflection_map = task.generate_self_reflection(unique_trajectories, x)
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, reflection_map)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

def generate_new_states(node, args, task, n):
    global failed_trajectories
    prompt = node.question + generate_prompt(node)
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, prompt_sample=args.prompt_sample, stop="Observation")
    tried_actions = []
    
    unique_states = {}  # Store unique states here
    for action in sampled_actions:
        new_state = node.state.copy()  # Make a copy of the parent node's state

        thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue  # Skip if this state already exists

        tried_actions.append(action_line)
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")

            # Update the new state dictionary
            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = obs

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            new_node.depth = node.depth + 1

            unique_states[unique_key] = new_node  # Add this state to unique_states

            if new_node.is_terminal and r == 0:
                traversed_nodes = upward_traversal(new_node)
                failed_trajectories.append({'trajectory': traversed_nodes, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # Return unique nodes as a list


def expand_node(node, args, task, depth_limit=7):
    if node.depth >= depth_limit:
        node.is_terminal = True
        return []
    children_nodes = generate_new_states(node, args, task, args.n_generate_sample)
    return children_nodes

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    global reflection_map
    global failed_trajectories
    
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    value_prompt = task.value_prompt_wrap(x, y, unique_trajectories, reflection_map)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def evaluate_node(node, args, task):
    child_prompts = [child.question + generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        
    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    for i, child in enumerate(node.children):
        child.value = votes[i] 

    return sum(votes) / len(votes) if votes else 0

def rollout(node, args, task, idx, max_depth=4):
    depth = node.depth
    n = 5
    rewards = [0]
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, n)

        for state in new_states:
            if state.is_terminal:
                return state.reward, state
                
        child_prompts = [child.question + generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        #new_state = new_state[0]
        while len(values) == 0:
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        max_value_index = values.index(max(values))
        rewards.append(max(values))
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            rewards = [-1]
    
    return sum(rewards) / len(rewards), node

def backpropagate(node, value):
    while node:
        node.visits += 1
        if node.is_terminal:
            if node.reward == 0:
                node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
        else:
            node.value = (node.value * (node.visits - 1) + value) / node.visits

        node = node.parent

def collect_all_nodes(node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(collect_all_nodes(child))
        return nodes