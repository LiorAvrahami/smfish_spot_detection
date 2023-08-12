import os
from .Image2numpy import convert_tiff_to_numpy
import numpy.typing as npt
import numpy as np
import csv


def load_tagged_image(folder_path: str) -> tuple[npt.NDArray, npt.NDArray]:
    """load some tagged image and the tagged points

    Args:
        folder_path (str): the path to the folder containing the image and the points .csv file

    Returns:
        tuple[npt.NDArray,npt.NDArray]: a tuple the elements of which are:
                                        - np array that contains the image
                                        - np array with shape (num_points, 4), and contains the coordinates of the tagged centers of all the points
                                                the points dimensions are in the order: x,y,z,channel
    """
    image_file_name = os.path.join(folder_path, "image.tif")
    points_file_name = os.path.join(folder_path, "Results.csv")
    image_array = convert_tiff_to_numpy(image_file_name)
    points_array = []
    with open(points_file_name) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        titles = next(csv_read)
        X_idx = titles.index("X")
        Y_idx = titles.index("Y")
        Z_idx = titles.index("Slice")
        Ch_idx = titles.index("Ch")
        for a in csv_read:
            new_array = [
                a[X_idx],
                a[Y_idx],
                a[Z_idx],
                a[Ch_idx]
            ]
            points_array.append(new_array)
    points_array = np.array(points_array)
    return image_array, points_array
