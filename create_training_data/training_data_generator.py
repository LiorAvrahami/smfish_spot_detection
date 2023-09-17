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

    @staticmethod
    def make_default_training_data_generator(batch_size, min_num_channels):
        points_parameters_generator = PointsParametersGenerator.make_default()
        backgrounds_generator = SmallBackgroundGenerator.make_default(min_num_channels=min_num_channels)
        return ClassifierTrainingDataGenerator(batch_size, points_parameters_generator, backgrounds_generator)

    def __init__(self, batch_size, points_parameters_generator: PointsParametersGenerator, backgrounds_generator: SmallBackgroundGenerator):
        self.batch_size = batch_size
        self.points_parameters_generator = points_parameters_generator
        self.backgrounds_generator = backgrounds_generator

    def __next__(self):
        return self.get_next_batch()

    def get_image_and_points(self):
        image = self.backgrounds_generator.get_next()
        dots_locations = add_points_to_image_multichannel(image, self.points_parameters_generator)
        return image, dots_locations

    def get_next_batch(self):
        """returns a batch of images of the size that was detailed in the initialization

        Returns:
            list of tuples, each tuple consists of a pair of (image,tag),
            image is a synthesized image with some background 
            and the tag is a vector of size 4, with the elements:
            (does any channel have a point?, does the first channel have a point?, does the second channel have a point?, does the third channel have a point?)
        """
        images = []  # all the small images around roi's
        small_coordinates = []  # the coordinates of the roi's relative to the small images
        is_image_of_dot = []  # the tag of whether the image has a dot near it's center or not.
        num_has_spot = 0
        num_has_no_spot = 0

        def append(img, coords, tag):
            nonlocal num_has_spot, num_has_no_spot
            if not is_valid_size(img):
                return False

            images.append(img)
            small_coordinates.append(coords)
            is_image_of_dot.append(tag)

            if tag == True:
                num_has_spot += 1
            if tag == False:
                num_has_no_spot += 1

            return True

        while len(images) < self.batch_size:
            # get next large image
            try:
                large_image, dots_locations = self.get_image_and_points()
            except StopIteration:
                break
            # break up large image into small ROI images
            denoise_net = get_default_denoise_net()
            get_roi = get_regions_of_interest_generator_from_net(large_image, denoise_net, 1, False)
            while True:  # loop over all roi's in the given image
                roi_coords_arr = get_roi()[1]
                if len(roi_coords_arr) != 1:
                    break
                roi_coords = roi_coords_arr[0]  # batch size is 1
                # determine if roi has a point in it
                has_point = does_roi_have_spot(roi_coords, dots_locations)
                # skip if has enough of the current tag (this is in order to combat class imbalance)
                if has_point and num_has_spot > self.batch_size / 2:
                    continue
                if (not has_point) and num_has_no_spot > self.batch_size / 2:
                    continue
                # crop tag and append
                small_im_coords, small_image = crop(large_image, *roi_coords)
                append(small_image, small_im_coords, has_point)
        images = images[:self.batch_size]
        is_image_of_dot = is_image_of_dot[:self.batch_size]
        return images, small_coordinates, is_image_of_dot


class ClassifierCheckerDataGenerator(ClassifierTrainingDataGenerator):
    images_tags: list[tuple]
    image_index = 0

    def __init__(self, folder_path: str) -> None:
        super().__init__(10000, None, None)
        image_folder_names = os.listdir(folder_path)
        self.images_tags = []
        for name in image_folder_names:
            image, points_array = load_tagged_image(os.path.join(folder_path, name))
            self.images_tags.append((image.transpose(1, 0, 2, 3), points_array))

    def get_image_and_points(self):
        if self.image_index >= len(self.images_tags):
            self.image_index = 0
            raise StopIteration()
        v = self.images_tags[self.image_index]
        self.image_index += 1
        return v


class ClassifierValidationDataGenerator(ClassifierCheckerDataGenerator):
    def __init__(self) -> None:
        super().__init__(os.path.join("images", "tagged_images_validation"))


class ClassifierTestDataGenerator(ClassifierCheckerDataGenerator):
    def __init__(self) -> None:
        super().__init__(os.path.join("images", "tagged_images_test"))


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
