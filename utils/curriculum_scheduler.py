import numpy as np
import random

def curriculum_scheduler(current_step, total_steps):
    """
    Returns the selected class based on the current step using
    cosine and sine functions for a non-linear change in probabilities.

    :param current_step: The current step number.
    :param total_steps: Total number of steps.
    :return: The selected class.
    """
    # Adjusting the phase and scale of the cosine and sine functions
    class1_prob = (np.cos(np.pi * current_step / total_steps) + 1) / 2
    class2_prob = np.sin(np.pi * current_step / total_steps)
    class3_prob = (np.cos(np.pi * current_step / total_steps - np.pi) + 1) / 2

    # Ensure total probability is 1
    total_prob = class1_prob + class2_prob + class3_prob
    class1_prob = class1_prob / total_prob
    class2_prob = class2_prob / total_prob
    class3_prob = class3_prob / total_prob

    # Select a class based on the probabilities
    selected_class = random.choices([1, 2, 3], weights=[class1_prob, class2_prob, class3_prob], k=1)[0]
    return selected_class

if __name__ == '__main__':
    
    # Example usage
    total_steps = 10
    selected_class = [curriculum_scheduler(step, total_steps) for step in range(total_steps)]