import numpy as np
import scipy.ndimage
from image_loader.Image2numpy import convert_image_file_to_numpy
from image_loader.load_tagged_image import load_tagged_image
import matplotlib.pyplot as plt


if False:
    image, points = load_tagged_image("images\\tagged_images\\img2")
    plt.figure()
    plt.imshow(np.max(image[:, :, :, 0], axis=-1))
    print(image.shape)
    plt.plot(points[:,0],points[:,1],"xr")

# plot graph of intensity vs z
if False:
    image, points = load_tagged_image("images\\tagged_images\\img2")
    points_to_draw = np.arange(10)
    # draw image
    plt.figure()
    plt.imshow(np.max(image[:, :, :, 0], axis=-1))
    print(image.shape)
    plt.plot(points[points_to_draw,0],points[points_to_draw,1],"xr")
    plt.figure()
    # draw z profile graphs
    for i in points_to_draw:
        y,x,z,ch = np.round(points[i]).astype(int)
        plt.plot(image[x,y,:,ch])
        plt.plot([z],[image[x,y,z,ch]],"xr")

# laplacian of gaussian
if True:
    image, points = load_tagged_image("images\\tagged_images\\img2")
    ch_index = 0
    plt.figure()
    plt.imshow(np.max(image[:, :, :, ch_index], axis=-1))
    plt.plot(points[:,0],points[:,1],"xr")
    plt.figure()
    im = image[:, :, :, ch_index]
    im_filtered = scipy.ndimage.gaussian_laplace(im,sigma=2,mode="mirror")*(-1)
    plt.imshow(np.max(im_filtered,axis=-1),cmap="gray")
    plt.plot(points[:,0],points[:,1],"xr")
    plt.title("sigma2")
    plt.figure()
    im = image[:, :, :, ch_index]
    im_filtered = scipy.ndimage.gaussian_laplace(im,sigma=1,mode="mirror")*(-1)
    plt.imshow(np.max(im_filtered,axis=-1),cmap="gray")
    plt.plot(points[:,0],points[:,1],"xr")
    plt.figure()
    im = image[:, :, :, ch_index]
    im_filtered = scipy.ndimage.gaussian_laplace(im,sigma=1,mode="mirror")*(-1)
    plt.imshow(np.max(im_filtered,axis=-1),cmap="gray")
    plt.plot(points[:,0],points[:,1],"xr")
    # plt.figure()
    cutoff = np.percentile(im_filtered,95)
    print(cutoff)
    # chosen = np.ones(im_filtered.shape)
    chosen = np.ma.masked_where(im_filtered<cutoff ,np.ones(im_filtered.shape))
    # selectted_points = np.ma.masked_where
    plt.imshow(np.max(chosen,axis=-1),cmap="inferno",alpha = 0.5)
    plt.colorbar()


plt.show()