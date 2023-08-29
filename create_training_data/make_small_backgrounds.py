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
    relevant_channels_array: Optional[list[tuple[int]]] = None
    relevant_z_min_array: Optional[list[int]] = None

    @staticmethod
    def make_default():
        bg_dir_path = "images\\backgrounds"
        big_images_paths = [os.path.join(bg_dir_path, a) for a in os.listdir(bg_dir_path) if os.path.splitext(a)[-1] not in FILE_EXTENTIONS_TO_IGNORE]
        big_images = [convert_image_file_to_numpy(path) for path in big_images_paths]
        relevant_channels_array = [(0, 1, 2) for i in range(len(big_images))]
        relevant_z_min_array = [0 for i in range(len(big_images))]
        return SmallBackgroundGenerator(big_images=big_images,
                                        relevant_channels_array=relevant_channels_array,  # type: ignore
                                        relevant_z_min_array=relevant_z_min_array)

    def __init__(self, big_images: list[npt.NDArray], relevant_channels_array: Optional[list[tuple[int]]] = None, relevant_z_min_array: Optional[list[int]] = None) -> None:
        """Args:
            big_images (list[npt.NDArray]): the list  of large images to be used as backgrounds. 
                                            NOTE the given images don't need to be normalized
        """
        self.relevant_channels_array = relevant_channels_array
        self.relevant_z_min_array = relevant_z_min_array
        self.big_images = []
        for image in big_images:
            self.big_images.append(normalize_image(image))

    def __next__(self):
        return self.get_next()

    def get_next(self):
        index = np.random.randint(0, len(self.big_images))
        big_image = self.big_images[index]

        relevant_channels = self.relevant_channels_array[index] if self.relevant_channels_array is not None else None
        relevant_z_min = self.relevant_z_min_array[index] if self.relevant_z_min_array is not None else None

        return make_small_background(big_image, relevant_channels=relevant_channels, relevant_z_min=relevant_z_min)


def make_small_background(large_background_image, relevant_channels=None, relevant_z_min=None):
    large_width = large_background_image.shape[0]
    large_height = large_background_image.shape[1]
    top_left = (int(np.random.uniform(0, large_width - CROPPED_IMAGE_SIZE[0])),
                int(np.random.uniform(0, large_height - CROPPED_IMAGE_SIZE[1])))
    cropped = crop_image(large_image=large_background_image, top_left=top_left,
                         relevant_channels=relevant_channels, relevant_z_min=relevant_z_min)
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
