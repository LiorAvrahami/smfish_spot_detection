import tifffile
import numpy as np

def save_numpy_to_image_file(image,name):
    reordered_image = image.astype(np.uint16).transpose(2,3,1,0)
    tifffile.imwrite(name,reordered_image,imagej=True)