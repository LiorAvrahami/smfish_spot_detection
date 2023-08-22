import numpy as np
from synthesize_points_module.synthesize_points import add_points_to_image_multichannel
from synthesize_points_module.points_parameters_generator import PointsParametersGenerator
from create_training_data.make_small_backgrounds import SmallBackgroundGenerator


class NoiseReductionTrainingDataGenerator():
    batch_size: int
    backgrounds_generator: SmallBackgroundGenerator
    points_parameters_generator: PointsParametersGenerator

    @staticmethod
    def make_default_training_data_generator(batch_size):
        points_parameters_generator = PointsParametersGenerator.make_default()
        backgrounds_generator = SmallBackgroundGenerator.make_default()
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
            the first image is a synthesized image with some background and the second is a the same image but without the background.
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
    def make_default_training_data_generator(batch_size):
        points_parameters_generator = PointsParametersGenerator.make_default()
        backgrounds_generator = SmallBackgroundGenerator.make_default()
        return ClassifierTrainingDataGenerator(batch_size, points_parameters_generator, backgrounds_generator)

    def __init__(self, batch_size, points_parameters_generator: PointsParametersGenerator, backgrounds_generator: SmallBackgroundGenerator):
        self.batch_size = batch_size
        self.points_parameters_generator = points_parameters_generator
        self.backgrounds_generator = backgrounds_generator

    def __next__(self):
        return self.get_next_batch()

    def get_next_batch(self):
        images = []
        dots_locations_arr = []
        for i in range(self.batch_size):
            image = self.backgrounds_generator.get_next()
            dots_locations = add_points_to_image_multichannel(image, self.points_parameters_generator)
            images.append(image)
            dots_locations_arr.append(dots_locations)
        return images, dots_locations_arr
