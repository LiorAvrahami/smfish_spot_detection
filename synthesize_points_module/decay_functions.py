import numpy as np


def exponential_decay(arr):
    return np.exp(-arr)
# Generate Gaussian random values with the same size as the given list
def exponential_decay_random(arr):
    return np.exp(-arr) + np.random.normal(0, 0.1, size = arr.shape)
# Generate Gaussian random values with the same size as the given list