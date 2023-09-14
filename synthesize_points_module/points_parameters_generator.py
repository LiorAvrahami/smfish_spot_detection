from abc import abstractmethod
from synthesize_points_module.decay_functions import *


class PointsParametersGenerator():
    @staticmethod
    def make_default():
        # TODO
        return MokPointsParametersGenerator()

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_points_parameters(self) -> dict:
        pass


class MokPointsParametersGenerator(PointsParametersGenerator):
    def get_points_parameters(self) -> dict:
        min_intensity = np.random.randint(200, 800)
        max_intensity = min_intensity + np.random.randint(300, 600)
        return {
            "points_density": 0.0001,
            "radius_1_min_max": (2.3, 1.9),
            "radius_2_min_max": (2, 1.7),
            "radius_3_min_max": (1.8, 1.5),
            "intensity_min_max": (min_intensity, max_intensity),
            "decay_function": exponential_decay_random,
            "z_metric_factor": 1.4,
        }
