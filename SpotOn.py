import csv
import itertools
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
from neural_network.model_classifier import spots_classifier_net as ClassifierNet

image_cleaner_network = DenoiseNet()
image_cleaner_network.load_state_dict(torch.load('denoiseNet.pt', map_location=torch.device('cpu')))

classifier_network = ClassifierNet()
classifier_network.load_state_dict(torch.load("classifyNet.pt", map_location=torch.device('cpu')))

NUM_OF_ROI_IN_BATCH = 300


def clean_image(image: npt.NDArray) -> npt.NDArray:
    image = torch.tensor(image)  # type: ignore
    image = image.permute(3, 2, 1, 0)  # get ch,z,x,y # type: ignore
    image_real = image.unsqueeze(1)  # type: ignore
    pred = image_cleaner_network(image_real.float()).detach()
    pred = pred[:, 0, :, :, :]
    pred = pred.permute(3, 2, 1, 0)
    return pred.numpy()


def classify_roi_list_Moc(small_image: npt.NDArray):
    has_point = np.full(len(small_image), True)
    result_channel = np.zeros(has_point.shape)
    return has_point, result_channel


def classify_roi_list(small_images_list: npt.NDArray, cutoff=0.5):
    images = torch.tensor(small_images_list)  # type: ignore
    images = images.permute(0, 4, 3, 2, 1)  # get ch,z,x,y # type: ignore
    pred = classifier_network(images.float()).detach()
    pred = torch.sigmoid(pred)
    has_point = pred.numpy() > cutoff
    result_channel = np.zeros(has_point.shape)
    return has_point, result_channel


def locate_spots(image: npt.NDArray, false_neg_to_false_pos_ratio, b_use_denoising=True, b_use_classifier=True) -> list[tuple[int, int, int, int]]:
    image = image[..., :3]
    result_detected_points = []

    # clean image with noise reduction net
    if b_use_denoising:
        print("cleaning image")
        image = clean_image(image)
        print("done with cleaning image")

    print("extracting points of interest")
    points_of_interest = list(zip(*np.where(find_points_of_interest(image))))  # type: ignore
    points_of_interest: list[tuple[int, int, int]]  # points_of_interest = list of (x,y,z) points
    print(f"done extracting points of interest, found {len(points_of_interest)}")

    print(f"classifying point of interest in batches of {NUM_OF_ROI_IN_BATCH}")
    for batch_index in itertools.count():
        print(f"working on batch number {batch_index}")
        # extract roi's for batch
        batch_points_of_interest = \
            points_of_interest[
                batch_index * NUM_OF_ROI_IN_BATCH:
                (batch_index + 1) * NUM_OF_ROI_IN_BATCH
            ]
        if len(batch_points_of_interest) == 0:
            break

        # prepare a batch of small images
        small_images_arr = []
        for point in batch_points_of_interest:
            small_image = crop(image, point[0], point[1])
            if not is_valid_size(small_image):
                continue
            small_images_arr.append(small_image)
        small_images_arr = np.array(small_images_arr)

        if b_use_classifier:
            has_point_list, result_channels_list = classify_roi_list(small_images_arr)
        else:
            has_point_list, result_channels_list = classify_roi_list_Moc(small_images_arr)

        for point_index in range(len(has_point_list)):
            if has_point_list[point_index]:
                point = batch_points_of_interest[point_index]
                result_detected_points.append((point[1], point[0], point[2], result_channels_list[point_index]))

    return result_detected_points


def export_spots_to_csv_file(points, file_name):
    with open(file_name, "w+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator="\n")
        for point in points:
            csv_writer.writerow(point)


if __name__ == "__main__":
    image = convert_image_file_to_numpy("images\\tagged_images_validation\\img6\\image.tif")
    image = normalize_image(image)
    detected_points = locate_spots(image, 1, b_use_classifier=False)
    export_spots_to_csv_file(detected_points, f"Result_{np.random.randint(1000)}.csv")
