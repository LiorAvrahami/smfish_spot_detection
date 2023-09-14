import numpy as np


def exponential_decay(arr):
    return np.exp(-arr)


def exponential_decay_random(arr):
    # Generate Gaussian random values with the same size as the given list
    return np.exp(-arr + np.random.normal(0, 0.1, size=arr.shape))
