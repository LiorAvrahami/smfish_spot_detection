import numpy as np
import numpy.typing as npt


def reduce_channels(channels_intensity: npt.NDArray) -> tuple[float, int]:
    """ reduce n channels to one float. this function returns a measure between 1 and 0. 
    the measure measures how singular the array is. an array is said to be singular if only
    one entry in the array is 1, and the others zero. 

    Returns:
        _type_: tuple (probability of dot in image, index of channel (starting at 1))
    """
    base_term = np.sum(channels_intensity**2)
    product = 1
    for i in range(len(channels_intensity)):
        product *= base_term - 2 * channels_intensity[i] + 1
    return np.exp(-product), int(np.argmax(channels_intensity) + 1)
