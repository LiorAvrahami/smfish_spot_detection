import numpy as np
import numpy.typing as npt

from Global_Constants import MAX_DISTANCE_FROM_ROI_CENTER


def does_roi_have_spot(roi_coordinate, spots_locations):
    displacments = spots_locations - roi_coordinate  # subtract roi_coordinate from the coordinates of each spot
    close_indexes: npt.NDArray  # bool array, is true where the displacement is small enough and thus the spot is in the roi
    close_indexes = displacments[:, -1] == 0  # for some spot to be close, it has to be on the same channel
    displacments = displacments[:, :-1]  # no more need for channel in displacement
    distance = np.sum((displacments / MAX_DISTANCE_FROM_ROI_CENTER)**2, axis=-1)**0.5
    close_indexes = distance < 1
    return np.any(close_indexes)
