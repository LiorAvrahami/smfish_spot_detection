import skimage
import numpy as np
import scipy.ndimage

def find_points_of_interest(image):
    points_of_interest = np.full(image.shape[:-1],False)
    for ch_index in range(image.shape[3]):
        im = image[:, :, :, ch_index]
        im_after_gl = scipy.ndimage.gaussian_laplace(im,sigma=1,mode="mirror")*(-1)
        cutoff_val = find_cutoff_value(im_after_gl)
        points_of_interest[im_after_gl > cutoff_val] = True
    return points_of_interest


def find_cutoff_value(image):
    # return skimage.filters.threshold_otsu(image)
    return np.percentile(image,95)
