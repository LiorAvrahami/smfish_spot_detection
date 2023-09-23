import csv
import itertools
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch

from regions_of_interest_module.get_regions_of_interest_from_net import get_regions_of_interest_generator_from_net
from image_loader.Image2numpy import convert_image_file_to_numpy
from image_manipulation.standardize_image import normalize_image

# now comes the neural network
from spot_classifier_NN.classifier_model import spots_classifier_net as ClassifierNet
classifier_networks = {}  # classifier networks, key is number of input channels

from noise_reduction.model_denoise import DenoiseNet
denoise_net = DenoiseNet()
denoise_net.load_state_dict(torch.load('denoiseNet.pt', map_location=torch.device('cpu')))

NUM_OF_ROI_IN_BATCH = 300


def classify_roi_list_Moc(roi_batch_images: npt.NDArray, roi_batch_coordinates):
    has_point = np.full(len(roi_batch_images), True)
    result_channel = np.zeros(has_point.shape)
    return has_point, result_channel


def load_classifier_net(num_channels) -> ClassifierNet:
    """load classifier network on the fly. this allows us to not load all the classifier networks every time SpotOn is run.

    Args:
        num_channels (int): the number of channels of the image to be classified

    Returns:
        ClassifierNet: the correct classifier net object
    """
    global classifier_networks
    if num_channels == 1:
        key = 1
    elif num_channels == 2:
        key = 2
    elif num_channels >= 3:
        key = 3
    else:
        raise NotImplementedError
    if key in classifier_networks:
        print(f"loading classifier for {key} channels")
        classifier_networks[key] = ClassifierNet(key)
        classifier_networks[key].load_state_dict(torch.load(
            f"classifyNet_{key}_ch.pt", map_location=torch.device('cpu')))
    return classifier_networks[key]


def classify_roi_list(roi_batch_images: npt.NDArray, roi_batch_coordinates, cutoff=0.5):
    num_channels = roi_batch_images.shape[-1]
    current_classifier_network = load_classifier_net(num_channels)
    images = torch.tensor(roi_batch_images)  # type: ignore
    images = images.permute(0, 4, 3, 2, 1)  # get ch,z,x,y # type: ignore
    pred = current_classifier_network(images.float()).detach()
    pred = torch.sigmoid(pred)
    has_point = pred.numpy() > cutoff
    result_channel = np.zeros(has_point.shape)
    return has_point, result_channel


def locate_spots(image: npt.NDArray, false_neg_to_false_pos_ratio, b_use_denoising=True, b_use_classifier=True) -> list[tuple[int, int, int, int]]:
    result_detected_points = []

    num_channels = image.shape[-1]

    print("extracting points of interest")
    get_next_roi_batch = get_regions_of_interest_generator_from_net(
        image, denoise_net, NUM_OF_ROI_IN_BATCH, num_channels_out=num_channels)

    print(f"classifying point of interest in batches of {NUM_OF_ROI_IN_BATCH}")
    while True:
        roi_batch_images, roi_batch_small_coordinates, _ = get_next_roi_batch()
        if len(roi_batch_images) == 0:
            break

        if b_use_classifier:
            has_point_list, result_channels_list = classify_roi_list(roi_batch_images, roi_batch_small_coordinates)
        else:
            has_point_list, result_channels_list = classify_roi_list_Moc(roi_batch_images, roi_batch_small_coordinates)

        for point_index in range(len(has_point_list)):
            if has_point_list[point_index]:
                point = roi_batch_small_coordinates[point_index]
                result_detected_points.append((point[1], point[0], point[2], result_channels_list[point_index]))

    return result_detected_points


def export_spots_to_csv_file(points, file_name):
    print(f"saving to {file_name}")
    with open(file_name, "w+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator="\n")
        for point in points:
            csv_writer.writerow(point)


if __name__ == "__main__":
    image = convert_image_file_to_numpy("images\\tagged_images_validation\\img6\\image.tif")
    image = normalize_image(image)
    detected_points = locate_spots(image, 1, b_use_classifier=True)
    export_spots_to_csv_file(detected_points, f"Result_{np.random.randint(1000)}.csv")
