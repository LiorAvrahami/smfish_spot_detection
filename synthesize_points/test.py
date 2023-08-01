from tkinter.tix import Tree
import numpy as np
import synthesize_points
import matplotlib.pyplot as plt

# test synthesize_points
if False:
    images, locations = synthesize_points.make_points_shapes(
        20, [3, 3], [4, 4], [1, 1], [1, 1], synthesize_points.exponential_decay, 1)
    im_size = images.shape[1]
    for i in range(20):
        plt.imshow(images[i, im_size // 2])
        plt.show()

# test synthesize_points and add them to image
b_draw_tagged_locations = True
b_add_noise = True
if True:
    # np.random.seed(3)
    image = np.zeros((100, 100, 4))
    # note that I create the random noise anyway for the random seed to be consistent.
    noise = np.random.normal(0, 0.25, image.shape)
    if b_add_noise:
        image += noise
    points_locations = synthesize_points.add_points_to_image(
        image, 20, [1.5, 2], [1, 1.5], [1, 1], [1, 3], synthesize_points.exponential_decay, 1)
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
            axes[z_stack].plot(points_locations[indexes, 1], points_locations[indexes, 0], "rx", alpha=0.5)
            axes[z_stack].set_xlim(0, image.shape[0])
            axes[z_stack].set_ylim(0, image.shape[1])
    plt.show()
