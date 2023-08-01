import itertools
import numpy as np
import numpy.typing as npt


def add_points_to_image(image, num_points, radius_1_min_max, radius_2_min_max, radius_3_min_max,
                        intensity_min_max, decay_function, z_metric_factor) -> npt.NDArray[np.float32]:
    """_summary_

    Args:
        image (_type_): _description_
        other: LOOK UP "make_points_shapes" FOR DOCUMENTATION OF OTHER ARGUMENTS

    Returns:
        npt.NDArray[np.float32]: the locations of the added points. shape is (num_points,3)
    """
    point_images, point_sub_pixel_displacement = make_points_shapes(num_points, radius_1_min_max, radius_2_min_max, radius_3_min_max,
                                                                    intensity_min_max, decay_function, z_metric_factor)
    centers_x = np.random.uniform(0, image.shape[0], num_points).astype(np.int32)
    centers_y = np.random.uniform(0, image.shape[1], num_points).astype(np.int32)
    centers_z = np.random.uniform(0, image.shape[2], num_points).astype(np.int32)
    centers_vectors = np.stack([centers_x, centers_y, centers_z], axis=-1)
    for point_index in range(point_images.shape[0]):
        plant_point_in_image(point_images[point_index], image, centers_x[point_index],
                             centers_y[point_index], centers_z[point_index])
    return point_sub_pixel_displacement + centers_vectors


def make_points_shapes(num_points, radius_1_min_max, radius_2_min_max, radius_3_min_max,
                       intensity_min_max, decay_function, z_metric_factor) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """returns small images of synthesized signal points

    Args:
        num_points (_type_): _description_
        radius_1_min_max (_type_): _description_
        radius_2_min_max (_type_): _description_
        radius_3_min_max (_type_): _description_
        intensity_min_max (_type_): _description_
        decay_function (_type_): _description_
        z_metric_factor (_type_): _description_

    Returns:
        npt.NDArray[np.float32]: 
            1) the 3d image of each point, shape of np array is (num_points,image_size,image_size,image_size)
            2) the location of each point relative to the top left corner of the image, shape is (num_points,3)
    """
    angles_array = np.random.uniform(0, 2 * np.pi, (3, num_points))
    cos = np.cos(angles_array)
    sin = np.sin(angles_array)

    radius1_array = np.random.uniform(*radius_1_min_max, size=num_points)
    radius2_array = np.random.uniform(*radius_2_min_max, size=num_points)
    radius3_array = np.random.uniform(*radius_3_min_max, size=num_points)

    sub_pixel_displacement_array = np.random.uniform(0, 1, (num_points, 3)).astype(np.float32)
    intensity_array = np.random.uniform(*intensity_min_max, size=num_points)

    non_rotated_metric_matrixes = np.zeros((num_points, 3, 3))
    non_rotated_metric_matrixes[:, 0, 0] = 1 / (radius1_array)**2
    non_rotated_metric_matrixes[:, 1, 1] = 1 / (radius2_array)**2
    non_rotated_metric_matrixes[:, 2, 2] = 1 / (radius3_array)**2

    rotation_matrix = np.array(
        [[cos[0] * cos[1], cos[0] * sin[1] * sin[2] - sin[0] * cos[2], cos[0] * sin[1] * cos[2] + sin[0] * sin[2]],
         [sin[0] * cos[1], sin[0] * sin[1] * sin[2] + cos[0] * cos[2], sin[0] * sin[1] * cos[2] - cos[0] * sin[2]],
         [-sin[1], cos[1] * sin[2], cos[1] * cos[2]]]
    ).transpose([2, 0, 1])

    rotation_matrix_inverse = rotation_matrix.transpose([0, 2, 1])
    rotated_metric_matrixes = np.matmul(
        np.matmul(rotation_matrix_inverse, non_rotated_metric_matrixes), rotation_matrix)

    # the point's image size is equal to the largest radius times 4
    image_size = int(max(radius_1_min_max[1], radius_2_min_max[1], radius_3_min_max[1])) * 4
    point_images = np.full((num_points, image_size, image_size, image_size), np.nan, np.float32)

    image_coordinates_1d = np.arange(image_size, dtype=np.float32)
    image_coordinates_1d -= np.mean(image_coordinates_1d)
    image_coordinates_vectors_base = np.array(np.meshgrid(image_coordinates_1d, image_coordinates_1d,
                                                          image_coordinates_1d, indexing='ij'))
    image_coordinates_vectors_base = image_coordinates_vectors_base.transpose(1, 2, 3, 0)

    for point_index in range(num_points):
        point_metric = rotated_metric_matrixes[point_index]
        # subtract the same random sub-pixel 3-vector from all vectors in image_coordinates_vectors_base.
        image_coordinate_vectors = (
            image_coordinates_vectors_base -
            sub_pixel_displacement_array[point_index, np.newaxis, np.newaxis, np.newaxis, :]
        )

        # apply the metric bilinear form to get the metric distance from the center of the point's center.
        image_coordinates_distances_array = np.matmul(image_coordinate_vectors[..., np.newaxis, :],
                                                      np.matmul(point_metric, image_coordinate_vectors[..., np.newaxis]))

        point_images[point_index] = intensity_array[point_index] * \
            decay_function(image_coordinates_distances_array[..., 0, 0])

    return point_images, sub_pixel_displacement_array


def plant_point_in_image(point_image: npt.NDArray[np.float32], target_image, center_x: int, center_y: int, center_z: int):
    point_sz_x, point_sz_y, point_sz_z = point_image.shape
    image_sz_x, image_sz_y, image_sz_z = target_image.shape

    target_image[max(center_x - point_sz_x // 2, 0):center_x + point_sz_x // 2,
                 max(center_y - point_sz_y // 2, 0): center_y + point_sz_y // 2,
                 max(center_z - point_sz_z // 2, 0): center_z + point_sz_z // 2] += \
        point_image[max(point_sz_x // 2 - center_x, 0): point_sz_x // 2 + image_sz_x - center_x,
                    max(point_sz_y // 2 - center_y, 0): point_sz_y // 2 + image_sz_y - center_y,
                    max(point_sz_z // 2 - center_z, 0): point_sz_z // 2 + image_sz_z - center_z]


def exponential_decay(arr):
    return np.exp(-arr)
