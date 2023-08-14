from typing import Any
import numpy as np
import numpy.typing as npt
from Global_Constants import *


def crop_image(large_image: npt.NDArray[np.float64], top_left: tuple[int, int], relevant_channels=None, relevant_z_min=None):
    if relevant_channels is None:
        relevant_channels = DEFAULT_CHANNELS
    if relevant_z_min is None:
        relevant_z_min = DEFAULT_ZMIN
    return large_image[top_left[0]:top_left[0] + CROPPED_IMAGE_SIZE[0], top_left[1] + CROPPED_IMAGE_SIZE[1],
                       relevant_z_min:relevant_z_min + CROPPED_IMAGE_SIZE[2], relevant_channels]


def normalize_image(image: npt.NDArray[np.float64]):
    normalized_image = np.copy(image)
    num_channels = image.shape[-1]
    # normalize each channel separately
    for ch_idx in range(num_channels):
        sub_image = image[..., ch_idx]
        low_quantile, high_quantile = np.quantile(sub_image, NORMALIZATION_QUANTILES)
        # we want the normalize low to be -1 and high to be 1.
        normalized_image[..., ch_idx] = sub_image * 2 / (high_quantile - low_quantile) - \
            (high_quantile + low_quantile) / (high_quantile - low_quantile)

    return normalized_image
