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
        """returns a batch of images of the size that was detailed in the initialization

        Returns:
            list of tuples, each tuple consists of a pair of images,
            the first image is a synthesized image with some background and the second is a the same image but without the background.
        """
        image_size = 5
        images = []
        is_image_of_dot = []
        num_dots = 0
        num_not_dots = 0

        def append(img, tag):
            if img.shape[:2] != (image_size * 2 + 1, image_size * 2 + 1):
                return False
            images.append(img)
            is_image_of_dot.append(int(tag))
            return True

        while len(images) < self.batch_size:
            image = self.backgrounds_generator.get_next()
            dots_locations = add_points_to_image_multichannel(image, self.points_parameters_generator)
            for dot_location in dots_locations:
                b_success = append(
                    image[int(dot_location[0]) - image_size:int(dot_location[0]) + image_size + 1,
                          int(dot_location[1]) - image_size:int(dot_location[1]) + image_size + 1,
                          :,
                          :],
                    dot_location[-1]+1)
                num_dots += 1 if b_success else 0
            while num_not_dots < num_dots:
                x = np.random.uniform(0, image.shape[0])
                y = np.random.uniform(0, image.shape[1])
                b_success = append(image[int(x) - image_size:int(x) + image_size + 1,
                                         int(y) - image_size:int(y) + image_size + 1,
                                         :,
                                         :],
                                   -1)
                num_not_dots += 1 if b_success else 0
        images = images[:self.batch_size]
        is_image_of_dot = is_image_of_dot[:self.batch_size]
        return images, is_image_of_dot
