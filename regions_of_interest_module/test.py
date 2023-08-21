import os,sys
try:
    import this_is_root
except:
    os.chdir(os.path.pardir) # change workdir to be root dir
    sys.path.insert(0, os.path.realpath("."))

import regions_of_interest_module.regions_of_interest as regions_of_interest
from image_loader.load_tagged_image import load_tagged_image
from image_loader.Image2numpy import convert_image_file_to_numpy
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

if True:
    image = convert_image_file_to_numpy("images\\tagged_images\img3\OG_Big_Image.nd2")
    points_of_interest = regions_of_interest.find_points_of_interest(image)
    mask = np.ma.masked_where(points_of_interest ,np.ones(points_of_interest.shape))
    plt.imshow(np.max(image[...,0],axis=-1),cmap="gray")
    plt.figure()
    plt.imshow(np.max(image[...,0],axis=-1),cmap="gray")
    plt.imshow(np.max(points_of_interest,axis=-1),cmap="magma",alpha=0.2)

plt.show()
