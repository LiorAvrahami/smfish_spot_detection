import skimage
import numpy as np
import scipy.ndimage


def find_points_of_interest(image):
    points_of_interest = np.full(image.shape, False)
    for ch_index in range(image.shape[3]):
        im = image[:, :, :, ch_index]

        is_local_maximum = im == scipy.ndimage.maximum_filter(im, size=3, mode="reflect")
        # find the cutoff value, this value is the value of the num_poi_per_channel'th brightest local maximum after cleaning.
        cutoff_val = np.mean(im) + np.std(im) * 3

        indexes_points = im > cutoff_val

        points_of_interest[indexes_points * is_local_maximum, ch_index] = True
    return points_of_interest
