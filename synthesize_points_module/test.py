from tkinter.tix import Tree
import numpy as np
from sympy import false
import synthesize_points_module.synthesize_points as synthesize_points
import matplotlib.pyplot as plt

# test single point
if True:
    _,(ax1,ax2) = plt.subplots(1,2)
    r = 3
    image1 = synthesize_points.make_points_shapes(5, [r, r], [r, r], [r, r], [1, 1], synthesize_points.exponential_decay, 1)[0]
    image2 = synthesize_points.make_points_shapes(5, [r, r], [r, r], [r, r], [1, 1], synthesize_points.exponential_decay, 2.4)[0]
    sh = image1[0].shape
    x,y,z = sh[0]//2,sh[1]//2,sh[2]//2
    def pl(im,a):
        assert(im.shape == sh)
        ax1.plot(im[x,y],a)
        ax2.plot(im[x,:,z],a)

    for im in image1:
        pl(im,"r")
    for im in image2:
        pl(im,"g")

# test synthesize_points
if False:
    r=3.5
    images, locations = synthesize_points.make_points_shapes(
        20, [r, r], [r, r], [r, r], [1, 1], synthesize_points.exponential_decay, 1)
    for i in range(20):
        im_size = images[i].shape
        plt.figure()
        plt.imshow(np.max(images[i],axis=-1),cmap="gray",vmin = 0.35)
        plt.plot(locations[i][1] + im_size[1]//2,locations[i][0] + im_size[0]//2,".r")

# test synthesize_points and add them to image
b_draw_tagged_locations = True
b_add_noise = False
if False:
    # np.random.seed(3)
    image = np.zeros((100, 100, 4))
    # note that I create the random noise anyway for the random seed to be consistent.
    noise = np.random.normal(0, 0.25, image.shape)
    if b_add_noise:
        image += noise
    points_locations = synthesize_points.add_points_to_image(
        image, 0.001, [1.5, 2], [1, 1.5], [1, 1], [1, 3], synthesize_points.exponential_decay, 1)
    _, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    for z_stack in range(image.shape[2]):
        # add title
        axes[z_stack].set_title(f"z={z_stack}")

        # draw image
        axes[z_stack].imshow(image[:, :, z_stack], cmap="gray", vmin=np.min(image), vmax=np.max(image))
        if b_draw_tagged_locations:
            # draw points
            indexes = np.minimum(np.round(points_locations[:, 2]), image.shape[2] - 1) == z_stack
            axes[z_stack].plot(points_locations[indexes, 1], points_locations[indexes, 0], ".r", alpha=0.5)
            axes[z_stack].set_xlim(0, image.shape[0])
            axes[z_stack].set_ylim(0, image.shape[1])

plt.show()
