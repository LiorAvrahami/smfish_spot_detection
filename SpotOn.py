import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch

from regions_of_interest_module.regions_of_interest import find_points_of_interest
from image_manipulation.crop_image_for_classifier import crop, is_valid_size


def locate_spots(image: npt.NDArray):
    result_detected_points = []
    # clean image with noise reduction net
    # image = clean image

    points_of_interest = list(zip(*np.where(find_points_of_interest(image))))  # type: ignore
    points_of_interest: list[tuple[int, int, int]]  # points_of_interest = list of (x,y,z) points
    for point in points_of_interest:
        im_temp = crop(image, point[0], point[1])
        if not is_valid_size(im_temp):
            continue
        result_channel = np.random.randint  # TODO feed im_temp to classifier_network
        if result_channel != -1:
            result_detected_points.append((point[0], point[1], point[2], result_channel))

    return result_detected_points
