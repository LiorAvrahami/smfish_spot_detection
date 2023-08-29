import csv
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch

from regions_of_interest_module.regions_of_interest import find_points_of_interest
from image_manipulation.crop_image_for_classifier import crop, is_valid_size
from image_loader.Image2numpy import convert_image_file_to_numpy
from image_manipulation.standardize_image import normalize_image

# now comes the neural network
from noise_reduction.model_denoise import DenoiseNet

image_cleaner_network = DenoiseNet()
image_cleaner_network.load_state_dict(torch.load('denoiseNet.pt'))


def clean_image(image: npt.NDArray) -> npt.NDArray:
    image = torch.tensor(image)  # type: ignore
    image = image.permute(3, 2, 1, 0)  # get ch,z,x,y # type: ignore
    image_real = image.unsqueeze(1)  # type: ignore
    pred = image_cleaner_network(image_real.float()).detach()
    pred = pred[:, 0, :, :, :]
    pred = pred.permute(3, 2, 1, 0)
    return pred.numpy()


def classify_roi(small_image: npt.NDArray):
    has_point = True  # bool(np.random.choice([False, True], p=[0.99, 0.01]))  # TODO feed im_temp to classifier_network
    result_channel = 0
    return has_point, result_channel


def locate_spots(image: npt.NDArray, b_use_denoising=True) -> list[tuple[int, int, int, int]]:
    result_detected_points = []

    # clean image with noise reduction net
    if b_use_denoising:
        image = clean_image(image)

    points_of_interest = list(zip(*np.where(find_points_of_interest(image))))  # type: ignore
    points_of_interest: list[tuple[int, int, int]]  # points_of_interest = list of (x,y,z) points
    for point in points_of_interest:
        small_image = crop(image, point[0], point[1])
        if not is_valid_size(small_image):
            continue
        has_point, result_channel = classify_roi(small_image)
        if has_point:
            result_detected_points.append((point[1], point[0], point[2], result_channel))

    return result_detected_points


def export_spots_to_csv_file(points, file_name):
    with open(file_name, "w+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator="\n")
        for point in points:
            csv_writer.writerow(point)


if __name__ == "__main__":
    image = convert_image_file_to_numpy("images\\tagged_images_validation\\img2\\image.tif")
    image = normalize_image(image)
    detected_points = locate_spots(image)
    export_spots_to_csv_file(detected_points, f"Result_{np.random.randint(1000)}.csv")
