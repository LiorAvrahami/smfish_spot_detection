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
        return {
            "points_density": 0.0001,
            "radius_1_min_max": (2.3, 1.9),
            "radius_2_min_max": (2, 1.7),
            "radius_3_min_max": (1.8, 1.5),
            "intensity_min_max": (1000, 400),
            "decay_function": exponential_decay,
            "z_metric_factor": 2.4,
        }
