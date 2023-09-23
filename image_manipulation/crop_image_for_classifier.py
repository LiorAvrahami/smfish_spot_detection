image_size = 5
image_size_z = 2
image_size_ch = 1
small_image_shape = (image_size * 2 + 1, image_size * 2 + 1, image_size_z * 2 + 1)


def crop(num_channels_out, image, x, y, z, ch):
    """_summary_

    Args:
        num_channels (int): the number of channels in the cropped image
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
    # NOTE the implementation of this function assumes that the number of channels is 1, 2 or 3.
    num_channels_in = image.shape[-1]
    # input image should have more channels than the number required from the cropped image
    assert num_channels_in >= num_channels_out

    # if num_channels_in == num_channels_out:
    #     # image doesn't need to be cropped in the channel axis.
    #     return (image_size, image_size, image_size_z, ch), \
    #         image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
    #               z - image_size_z:z + image_size_z + 1, :]

    # if num_channels_out == 1:
    #     return (image_size, image_size, image_size_z, 0), \
    #         image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
    #               z - image_size_z:z + image_size_z + 1, ch:ch + 1]

    if ch == image.shape[-1] - 1:
        # channel is last channel, return the last 3 channels (this code assumes image_size_ch = 1)
        ret = (image_size, image_size, image_size_z, num_channels_out - 1), \
            image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
                  z - image_size_z:z + image_size_z + 1, -num_channels_out:]
    elif ch == 0:
        # channel is first channel, return the first 3 channels (this code assumes image_size_ch = 1)
        ret = (image_size, image_size, image_size_z, 0), \
            image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
                  z - image_size_z:z + image_size_z + 1, :num_channels_out]
    else:
        out_ch_coordinate = (num_channels_out + 1) // 2 - 1
        ch_left_bound = ch - out_ch_coordinate
        ch_right_bound = ch + out_ch_coordinate + (1 if num_channels_out % 2 == 0 else 0)
        # ch is an interior channel.
        ret = (image_size, image_size, image_size_z, out_ch_coordinate), \
            image[x - image_size:x + image_size + 1, y - image_size:y + image_size + 1,
                  z - image_size_z:z + image_size_z + 1, ch_left_bound:ch_right_bound + 1]
    assert ret[1].shape[-1] == num_channels_out
    return ret


def is_valid_size(cropped_image):
    return cropped_image.shape[:3] == small_image_shape[:3]
