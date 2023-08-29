import skimage
import numpy as np
import scipy.ndimage

B_USE_GAUSSIAN_LAPLACE = False


def find_points_of_interest(image, num_poi_per_channel):
    points_of_interest = np.full(image.shape[:-1], False)
    for ch_index in [0]:  # range(image.shape[3]):
        im = image[:, :, :, ch_index]
        if B_USE_GAUSSIAN_LAPLACE:
            im = scipy.ndimage.gaussian_laplace(im, sigma=1, mode="mirror") * (-1)

        is_local_maximum = im == scipy.ndimage.maximum_filter(im, size=3, mode="reflect")
        local_maximum_values = im[is_local_maximum]
        cutoff_val = np.sort(local_maximum_values)[-num_poi_per_channel]

        indexes_points = im > cutoff_val

        points_of_interest[indexes_points * is_local_maximum] = True
    return points_of_interest
