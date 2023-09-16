image_size = 5
image_size_z = 2
image_size_ch = 1
small_image_shape = (image_size * 2 + 1, image_size * 2 + 1, image_size_z * 2 + 1)


def crop(image, x, y, z, ch):
    """_summary_

    Args:
        image (ndarray): the image to crop
        x (int): the x around which to crop
        y (int): the y around which to crop
        z (int): the z around which to crop
        ch (int): the channel around which to crop

    Returns:
        tuple: tuple consists of two parts:
            - relative coordinates (x,y,z,ch). these are the given coordinates relative to the top-left corner of the cropped image. 
                for example if the cropping goes well, and cropped image is perfectly centered around the given coordinates, then the returned 
                (x,y,z,ch) will be (x,y,z,ch) = (image_size,image_size,image_size_z,image_size_ch)
            - the cropped image
    """
    if image.shape[-1] <= 2:
        # image has less then 3 channels, return all channels
        return (image_size, image_size, image_size_z, ch), \
            image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
                  z - image_size_z:z + image_size_z + 1, :]
    elif ch == image.shape[-1]:
        # channel is last channel, return the last 3 channels (this code assumes image_size_ch = 1)
        return (image_size, image_size, image_size_z, 2), \
            image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
                  z - image_size_z:z + image_size_z + 1, -3:]
    elif ch == 0:
        # channel is first channel, return the first 3 channels (this code assumes image_size_ch = 1)
        return (image_size, image_size, image_size_z, 0), \
            image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
                  z - image_size_z:z + image_size_z + 1, :3]
    else:
        # ch is an interior channel.
        return (image_size, image_size, image_size_z, image_size_ch), \
            image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
                  z - image_size_z:z + image_size_z + 1, ch - image_size_ch:ch + image_size_ch + 1]


def is_valid_size(cropped_image):
    return cropped_image.shape[:3] == small_image_shape[:3]
