"""ExpeL training components."""

import random
from typing import List, Dict, Optional
from discussion_agents.cog.agent.reflexion import ReflexionReActAgent


# Q1: Should this experience be a part of the ExpeLAgent class?
# Q2: Should this Experience Gathering Section be a function or a class?

def gather_experience(
    reflexion_react_agent: ReflexionReActAgent,
    questions: List[str],
    keys: List[str],
    strategy: Optional[str] = "reflexion",
) -> Dict[str, List]:
    experiences = {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": []
    }
    for idx, (question, key) in enumerate(zip(questions, keys)):
        trajectory = reflexion_react_agent.generate(
            question=question, key=key, strategy=strategy, reset=True
        )

        experiences["idxs"].append(idx)
        experiences["questions"].append(question)
        experiences["keys"].append(key)
        experiences["trajectories"].append(trajectory)
        experiences["reflections"].append(reflexion_react_agent.reflector.reflections)
        
    return experiences

def categorize_experiences(experiences: Dict[str, List]) -> Dict[str, List]:
    count_dict = {
        "compare": [],
        "success": [],
        "fail": []
    }

    for idx in experiences["idxs"]:  # Index for a particular task.
        trajectory = experiences["trajectories"][idx]
        trials_are_correct = [trial[0] for trial in trajectory]  # (is_correct, answer, output)[0]

        # Success.
        if all(trials_are_correct) and len(trials_are_correct) == 1:  # If success @ first trial, then stop generation.
            count_dict["success"].append(idx)
        # Compare.
        elif trials_are_correct[-1]:  # If fail(s), then succeeds, then only last trial is True.
            count_dict["compare"].append(idx)
        # Fail.
        elif not all(trials_are_correct):  # All trials failed, then fail case.
            count_dict["fail"].append(idx)
        else:
            raise ValueError(f"Unhandled scenario for trajectory at index {idx}.")

    return count_dict

def get_folds(categories: Dict[str, List], n_instances: int, n_folds: int = 2) -> Dict[str, List]:
    folds = {fold: [] for fold in range(n_folds)}

    # Assign labels for 'compare', 'success', and  'fail'.
    for _, indices in categories.items():
        random.shuffle(indices)
        for count, idx in enumerate(indices):
            folds[count % n_folds].append(idx)

    # Each fold is a validation set. Take the difference to get the training set of each fold.
    folds = {fold: set(list(range(n_instances))).difference(values) for fold, values in folds.items()}

    return folds