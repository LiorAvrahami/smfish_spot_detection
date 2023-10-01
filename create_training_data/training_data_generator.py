import os
import numpy as np
from regions_of_interest_module.does_roi_have_point import does_roi_have_spot
from regions_of_interest_module.get_regions_of_interest_from_net import get_default_denoise_net, get_regions_of_interest_generator_from_net
from synthesize_points_module.synthesize_points import add_points_to_image_multichannel
from synthesize_points_module.points_parameters_generator import PointsParametersGenerator
from create_training_data.make_small_backgrounds import SmallBackgroundGenerator
from image_loader.load_tagged_image import load_tagged_image
from image_manipulation.crop_image_for_classifier import crop, is_valid_size


class NoiseReductionTrainingDataGenerator():
    batch_size: int
    backgrounds_generator: SmallBackgroundGenerator
    points_parameters_generator: PointsParametersGenerator

    @staticmethod
    def make_default_training_data_generator(batch_size):
        points_parameters_generator = PointsParametersGenerator.make_default()
        backgrounds_generator = SmallBackgroundGenerator.make_default(min_num_channels=1)
        return NoiseReductionTrainingDataGenerator(batch_size, points_parameters_generator, backgrounds_generator)

    def __init__(self, batch_size, points_parameters_generator: PointsParametersGenerator, backgrounds_generator: SmallBackgroundGenerator):
        self.batch_size = batch_size
        self.points_parameters_generator = points_parameters_generator
        self.backgrounds_generator = backgrounds_generator

    def __next__(self):
        return self.get_next_batch()

    def get_next_batch(self):
        """returns a batch of images of the size that was detailed in the initialization

        Returns:
            list of tuples, each tuple consists of a pair of images,
            the first image is a synthesized image with some background and the second 
            is a the same image but without the background.
        """
        images = []
        for i in range(self.batch_size):
            image = self.backgrounds_generator.get_next()
            empty_image = np.zeros(image.shape)
            add_points_to_image_multichannel(empty_image, self.points_parameters_generator)
            image += empty_image
            images.append((image, empty_image))
        return images


class ClassifierTrainingDataGenerator():
    batch_size: int
    backgrounds_generator: SmallBackgroundGenerator
    points_parameters_generator: PointsParametersGenerator
    num_channels: int

    @staticmethod
    def make_default_training_data_generator(batch_size, num_channels):
        points_parameters_generator = PointsParametersGenerator.make_default()
        backgrounds_generator = SmallBackgroundGenerator.make_default(min_num_channels=num_channels)
        return ClassifierTrainingDataGenerator(batch_size, points_parameters_generator, backgrounds_generator, num_channels=num_channels)

    @staticmethod
    def make_training_data_generator_with_empty_backgrounds(batch_size, num_channels):
        points_parameters_generator = PointsParametersGenerator.make_default()
        backgrounds_generator = SmallBackgroundGenerator.make_empty_background_generator(min_num_channels=num_channels)
        return ClassifierTrainingDataGenerator(batch_size, points_parameters_generator, backgrounds_generator, num_channels=num_channels)

    def __init__(self, batch_size, points_parameters_generator: PointsParametersGenerator, backgrounds_generator: SmallBackgroundGenerator, num_channels: int):
        self.batch_size = batch_size
        self.points_parameters_generator = points_parameters_generator
        self.backgrounds_generator = backgrounds_generator
        self.num_channels = num_channels

    def __next__(self):
        return self.get_next_batch()

    def get_image_and_points(self, min_num_channels):
        # get random image with enough channels
        image = None
        while image is None:
            image = self.backgrounds_generator.get_next()
            if image.shape[-1] < min_num_channels:
                image = None

        dots_locations = add_points_to_image_multichannel(image, self.points_parameters_generator)
        return image, dots_locations, None

    def get_next_batch_priv(self):
        """returns a batch of images of the size that was detailed in the initialization

        Returns:
            list of tuples, each tuple consists of a pair of (image,tag),
            image is a synthesized image with some background 
            and the tag is a vector of size 4, with the elements:
            (does any channel have a point?, does the first channel have a point?, does the second channel have a point?, does the third channel have a point?)
        """
        images = []  # all the small images around roi's
        small_coordinates = []  # the coordinates of the roi's relative to the small images
        big_coordinates = []  # the coordinates of the roi's relative to the big images
        is_image_of_dot = []  # the tag of whether the image has a dot near it's center or not.
        indexes_of_used_images = []
        num_has_spot = 0
        num_has_no_spot = 0

        def append(img, sml_coords, big_coords, tag, image_index):
            nonlocal num_has_spot, num_has_no_spot
            assert img.shape[-1] == self.num_channels
            if not is_valid_size(img):
                return False
            images.append(img)
            small_coordinates.append(sml_coords)
            big_coordinates.append(big_coords)
            is_image_of_dot.append(tag)
            indexes_of_used_images.append(image_index)

            if tag == True:
                num_has_spot += 1
            if tag == False:
                num_has_no_spot += 1

            return True

        while len(images) < self.batch_size:
            # get next large image
            try:
                large_image, dots_locations, image_index = self.get_image_and_points(self.num_channels)
            except StopIteration:
                break
            # break up large image into small ROI images
            denoise_net = get_default_denoise_net()
            get_roi = get_regions_of_interest_generator_from_net(
                large_image, denoise_net, batch_size=1, num_channels_out=self.num_channels, b_use_denoising_net=True, verbosity=False)
            while True:  # loop over all roi's in the given image
                roi_image_arr, roi_sml_coords_arr, roi_big_coords_arr = get_roi()
                if len(roi_sml_coords_arr) != 1:
                    break
                roi_sml_coords = roi_sml_coords_arr[0]  # batch size is 1
                roi_image = roi_image_arr[0]            # batch size is 1
                roi_big_coords = roi_big_coords_arr[0]  # batch size is 1
                # determine if roi has a point in it
                has_point = does_roi_have_spot(roi_big_coords, dots_locations)
                # skip if has enough of the current tag (this is in order to combat class imbalance)
                if has_point and num_has_spot >= self.batch_size / 2:
                    continue
                if (not has_point) and num_has_no_spot >= self.batch_size / 2:
                    continue
                # crop tag and append
                append(roi_image, roi_sml_coords, roi_big_coords, has_point, image_index)
        images = images[:self.batch_size]
        is_image_of_dot = is_image_of_dot[:self.batch_size]
        return images, small_coordinates, big_coordinates, is_image_of_dot, indexes_of_used_images

    def get_next_batch(self):
        vals = self.get_next_batch_priv()
        return vals[:4]


class ClassifierCheckerDataGenerator(ClassifierTrainingDataGenerator):
    image_folder_names: list[str]
    images_tags: list[tuple]
    image_index = 0

    @staticmethod
    def make_default_training_data_generator(batch_size, num_channels):
        # this function is only for initiating traning data generators.
        raise Exception()

    def __init__(self, folder_path: str, num_channels) -> None:
        super().__init__(100000, None, None, num_channels)
        self.image_folder_names = sorted(os.listdir(folder_path))
        self.images_tags = []
        for name in self.image_folder_names:
            image, points_array = load_tagged_image(os.path.join(folder_path, name))
            self.images_tags.append((image.transpose(1, 0, 2, 3), points_array))

    def get_image_and_points(self, min_num_channels):
        # get random image with enough channels
        image, tag, used_image_index = None, None, None
        while image is None:
            if self.image_index >= len(self.images_tags):
                self.image_index = 0
                raise StopIteration()
            image, tag = self.images_tags[self.image_index]
            used_image_index = self.image_index
            self.image_index += 1
            if image.shape[-1] < min_num_channels:
                image = None
        return image, tag, used_image_index

    def get_next_batch(self):
        images, small_coordinates, big_coordinates, is_image_of_dot, indexes_of_used_images = self.get_next_batch_priv()

        # filter out some images of not dots in order for the validation data to be balanced
        indexes_to_keep = [i for i in range(len(is_image_of_dot)) if is_image_of_dot[i] == True]
        indexes_of_not_dots = [i for i in np.random.permutation(len(is_image_of_dot)) if is_image_of_dot[i] == False]
        indexes_to_keep += indexes_of_not_dots[:len(indexes_to_keep)]

        new_images = [images[i] for i in indexes_to_keep]
        new_small_coordinates = [small_coordinates[i] for i in indexes_to_keep]
        new_big_coordinates = [big_coordinates[i] for i in indexes_to_keep]
        new_is_image_of_dot = [is_image_of_dot[i] for i in indexes_to_keep]
        new_indexes_of_used_images = [indexes_of_used_images[i] for i in indexes_to_keep]

        names_of_used_images = [self.image_folder_names[i] for i in new_indexes_of_used_images]
        return new_images, new_small_coordinates, new_big_coordinates, new_is_image_of_dot, names_of_used_images


class ClassifierValidationDataGenerator(ClassifierCheckerDataGenerator):
    def __init__(self, num_channels) -> None:
        super().__init__(os.path.join("images", "tagged_images_validation"), num_channels)


class ClassifierTestDataGenerator(ClassifierCheckerDataGenerator):
    def __init__(self, num_channels) -> None:
        super().__init__(os.path.join("images", "tagged_images_test"), num_channels)


class TaggedImagesGenerator():
    batch_size: int
    backgrounds_generator: SmallBackgroundGenerator
    points_parameters_generator: PointsParametersGenerator

    @staticmethod
    def make_default_training_data_generator(batch_size, num_channels):
        points_parameters_generator = PointsParametersGenerator.make_default()
        backgrounds_generator = SmallBackgroundGenerator.make_default(num_channels)
        return TaggedImagesGenerator(batch_size, points_parameters_generator, backgrounds_generator)

    def __init__(self, batch_size, points_parameters_generator: PointsParametersGenerator, backgrounds_generator: SmallBackgroundGenerator):
        self.batch_size = batch_size
        self.points_parameters_generator = points_parameters_generator
        self.backgrounds_generator = backgrounds_generator

    def __next__(self):
        return self.get_next_batch()

    def get_next_batch(self):
        """returns a batch of images of the size that was detailed in the initialization

        Returns:
            list of tuples, each tuple consists of a pair of images,
            the first image is a synthesized image with some background and the second 
            is a the same image but without the background.
        """
        image_and_tags = []
        for i in range(self.batch_size):
            image = self.backgrounds_generator.get_next()
            empty_image = np.zeros(image.shape)
            tags = add_points_to_image_multichannel(empty_image, self.points_parameters_generator)
            # swap x and y becasue that is the convention for tagged images (we want to stick to the convention)
            tags = tags[:, [1, 0, 2, 3]]
            image += empty_image
            image_and_tags.append((image, tags))
        return image_and_tags
