import numpy as np
from image_loader.Image2numpy import convert_tiff_to_numpy
from image_loader.load_tagged_image import load_tagged_image
import matplotlib.pyplot as plt
plt.ion()


if True:
    image, points = load_tagged_image("images\\tagged_images\\img3")
    plt.figure()
    plt.imshow(np.max(image[:, :, :, 0], axis=0))
    print(image.shape)
    # plt.plot(points[:,0],points[:,1],"o")
    plt.show()
