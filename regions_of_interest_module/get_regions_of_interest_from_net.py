import numpy as np
import numpy.typing as npt

from noise_reduction.use_denoise_net import clean_image
from regions_of_interest_module.regions_of_interest import find_points_of_interest
from image_manipulation.crop_image_for_classifier import crop, is_valid_size
import scipy.ndimage


def printv(*args, verbosity, **kwargs):
    if verbosity:
        return print(*args, **kwargs)


def get_regions_of_interest_generator_from_net(image, denoise_net, batch_size, verbosity=True, b_use_denoising_net=True):
    """
    creates a generator that creates and returns batches of ROI images on demand.
    Example Call:
        get_next_roi_batch = get_regions_of_interest_generator_from_net(image, denoise_net, batch_size)
        roi_batch_images, roi_batch_coordinates = get_next_roi_batch()

    Args:
        image (_type_): _description_
        net (_type_): _description_
        batch_size (_type_): _description_
        verbosity (bool): if true then prints diagnostics

    Returns: a function. the returned function takes no arguments and returns a tuple of small images and the coordinates of these images in 4d:
    """

    # clean image with noise reduction net
    if b_use_denoising_net:
        printv("cleaning image", verbosity=verbosity)
        denoised_image = clean_image(image, denoise_net=denoise_net)
        printv("done with cleaning image", verbosity=verbosity)
    else:
        denoised_image = -1 * scipy.ndimage.gaussian_laplace(image, sigma=1, mode="mirror").astype(float)

    roi_coordinates = list(zip(*np.where(find_points_of_interest(denoised_image))))  # type: ignore
    roi_coordinates: list[tuple[int, int, int, int]]  # points_of_interest = list of (x,y,z) points
    printv(f"done extracting points of interest, found {len(roi_coordinates)}", verbosity=verbosity)

    batch_index = 0

    def batch_generator_function():
        nonlocal batch_index
        print(f"working on batch number {batch_index}")
        roi_batch_coordinates = roi_coordinates[batch_index * batch_size:(batch_index + 1) * batch_size]

        # prepare a batch of small images
        roi_batch_images = []
        for point in roi_batch_coordinates:
            small_image = crop(image, point[0], point[1])
            if not is_valid_size(small_image):
                continue
            roi_batch_images.append(small_image)
        roi_batch_images = np.array(roi_batch_images)

        batch_index += 1
        return roi_batch_images, roi_batch_coordinates

    return batch_generator_function
