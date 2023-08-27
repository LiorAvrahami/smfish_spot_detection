image_size = 5
image_shape = (image_size * 2 + 1, image_size * 2 + 1)


def crop(image, x, y):
    image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1, :, :]


def is_valid_size(cropped_image):
    return cropped_image.shape[:2] == (image_size * 2 + 1, image_size * 2 + 1)
