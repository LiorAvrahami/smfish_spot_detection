import os
from typing import Optional
from .Image2numpy import convert_tiff_to_numpy
import numpy.typing as npt
import numpy as np
import csv
from Global_Constants import *
from image_manipulation.standardize_image import normalize_image


def load_tagged_image(folder_path: str, force_normalization_quantiles: Optional[tuple[float, float]] = None) -> tuple[npt.NDArray, npt.NDArray]:
    """load some tagged image and the tagged points, and normalize relative to the original large image 

    Args:
        folder_path (str): the path to the folder containing the image and the points .csv file

    Returns:
        tuple[npt.NDArray,npt.NDArray]: a tuple the elements of which are:
                                        - np array that contains the image
                                        - np array with shape (num_points, 4), and contains the coordinates of the tagged centers of all the points
                                                the points dimensions are in the order: x,y,z,channel
    """
    image_file_path = os.path.join(folder_path, "image.tif")
    original_big_image_file_path = os.path.join(folder_path, "OG_Big_Image.tif")
    points_file_path = os.path.join(folder_path, "Results.csv")
    image_array = convert_tiff_to_numpy(image_file_path)
    original_big_image_array = convert_tiff_to_numpy(original_big_image_file_path)
    # normalize image
    image_array = normalize_image(image_array,
                                  force_normalization_quantiles=force_normalization_quantiles,
                                  original_big_image_array=original_big_image_array)
    points_array = load_tagged_points_only(points_file_path)
    return image_array, points_array


def load_tagged_points_only(points_file_path):
    points_array = []
    with open(points_file_path) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        titles = next(csv_read)
        X_idx = titles.index("X")
        Y_idx = titles.index("Y")
        Z_idx = titles.index("Slice")
        Ch_idx = titles.index("Ch")
        for a in csv_read:
            new_array = [
                float(a[X_idx]) / CONVERSION_FACTOR_UM_TO_PIXELS,
                float(a[Y_idx]) / CONVERSION_FACTOR_UM_TO_PIXELS,
                int(a[Z_idx]) - 1,
                int(a[Ch_idx]) - 1
            ]
            points_array.append(new_array)
    return np.array(points_array)
