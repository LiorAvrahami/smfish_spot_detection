import os
from tkinter.tix import Tree
from typing import Optional
import numpy as np
import numpy.typing as npt
from image_manipulation.standardize_image import crop_image, normalize_image
from Global_Constants import *
from image_loader.Image2numpy import convert_image_file_to_numpy
from Global_Constants import FILE_EXTENTIONS_TO_IGNORE


class SmallBackgroundGenerator:
    big_images: list[npt.NDArray]
    # minimum number of channels in the background images.
    min_num_channels: int

    @staticmethod
    def make_default(min_num_channels):
        bg_dir_path = "images\\backgrounds"
        big_images_paths = [os.path.join(bg_dir_path, a) for a in os.listdir(
            bg_dir_path) if os.path.splitext(a)[-1] not in FILE_EXTENTIONS_TO_IGNORE]
        big_images = [convert_image_file_to_numpy(path) for path in big_images_paths]
        return SmallBackgroundGenerator(big_images=big_images,
                                        min_num_channels=min_num_channels)

    def __init__(self, big_images: list[npt.NDArray], min_num_channels: int) -> None:
        """Args:
            big_images (list[npt.NDArray]): the list  of large images to be used as backgrounds. 
                                            NOTE the given images don't need to be normalized
        """
        self.min_num_channels = min_num_channels
        self.big_images = []
        for image in big_images:
            self.big_images.append(normalize_image(image))

    def __next__(self):
        return self.get_next()

    def get_next(self):
        # find an image with enough channels
        while True:
            index = np.random.randint(0, len(self.big_images))
            big_image = self.big_images[index]
            if big_image.shape[-1] >= self.min_num_channels:
                break

        return make_small_background(big_image, self.min_num_channels)


def make_small_background(large_background_image, min_num_channels):
    large_width = large_background_image.shape[0]
    large_height = large_background_image.shape[1]
    top_left = (int(np.random.uniform(0, large_width - CROPPED_IMAGE_SIZE[0])),
                int(np.random.uniform(0, large_height - CROPPED_IMAGE_SIZE[1])))
    cropped = crop_image(large_image=large_background_image, top_left=top_left)
    transformed = preform_random_transformation(cropped)
    return transformed


def preform_random_transformation(image):
    # if equals 1 then doesn't flip, if equals -1 then flips the relevant axis
    b_flip_x = 1 if np.random.randint(0, 2) == 0 else -1
    b_flip_y = 1 if np.random.randint(0, 2) == 0 else -1

    b_rotate = True if np.random.randint(0, 2) == 0 else False

    if b_rotate:
        return np.rot90(image[::b_flip_x, ::b_flip_y, :, :])
    else:
        return image[::b_flip_x, ::b_flip_y, :, :]
