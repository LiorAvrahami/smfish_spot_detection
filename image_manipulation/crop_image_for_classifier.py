image_size = 5
image_size_z = 2
small_image_shape = (image_size * 2 + 1, image_size * 2 + 1, image_size_z * 2 + 1)


def crop(image, x, y, z):
    return image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1, z - image_size_z:z + image_size_z + 1, :]


def is_valid_size(cropped_image):
    return cropped_image.shape[:3] == small_image_shape[:3]
