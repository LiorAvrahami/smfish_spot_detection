import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch

from regions_of_interest_module.regions_of_interest import find_points_of_interest


def locate_spots(image: npt.NDArray):
    # clean image with noise reduction net
    # image = clean image

    points_of_interest = list(zip(*np.where(find_points_of_interest(image))))
    points_of_interest: list[tuple[int, int, int]]  # points_of_interest = list of (x,y,z) points
    for point in points_of_interest:
        im_temp = image[]

def get_points_of_interest():


def
