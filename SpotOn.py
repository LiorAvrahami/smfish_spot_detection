import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch

from regions_of_interest_module.regions_of_interest import find_points_of_interest
from image_manipulation.crop_image_for_classifier import crop, is_valid_size
from image_loader.Image2numpy import convert_image_file_to_numpy
from image_manipulation.standardize_image import normalize_image


def locate_spots(image: npt.NDArray) -> list[tuple[int, int, int, int]]:
    result_detected_points = []
    # clean image with noise reduction net
    # image = clean image

    points_of_interest = list(zip(*np.where(find_points_of_interest(image))))  # type: ignore
    points_of_interest: list[tuple[int, int, int]]  # points_of_interest = list of (x,y,z) points
    for point in points_of_interest:
        im_temp = crop(image, point[0], point[1])
        if not is_valid_size(im_temp):
            continue
        has_point = bool(np.random.randint(0, 2))  # TODO feed im_temp to classifier_network
        if not has_point:
            continue
        for result_channel in [0, 1, 2, 3]:
            result_detected_points.append((point[0], point[1], point[2], result_channel))

    return result_detected_points


def export_spots_to_csv_file(points, file_name):
    


if __name__ == "__main__":
    image = convert_image_file_to_numpy("images\\tagged_images_validation\\img2\\image.tif")
    image = normalize_image(image)
    detected_points = locate_spots(image)
    export_spots_to_roi_file(detected_points, "C:\\Liors_Stuff\\ML_Project\\smfish_spot_detection\\temp")
