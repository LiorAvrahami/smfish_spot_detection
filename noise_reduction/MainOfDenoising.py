# <editor-fold desc="imports">
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as func
from tqdm.notebook import tqdm
from PIL import Image
import glob

import os
import sys

os.chdir(os.path.pardir)  # change workdir to be root dir
sys.path.insert(0, os.path.realpath("."))

# </editor-fold>

cleaning_train = {}
cleaning_valid = {}
# train_valid_split = 0.85
#################
from create_training_data.training_data_generator import NoiseReductionTrainingDataGenerator

PicGenerator = NoiseReductionTrainingDataGenerator.make_default_training_data_generator(16)
# need to decide the number of batches. each batch is a list, in which each element is two images - the first one w bg
PicsBatch = PicGenerator.get_next_batch()

# # plot the image
# image_array_4D = PicsBatch[0][0]  # this is a 4D image
# z_index = 2
# Channel_index = 1
# image_array_slice = image_array_4D[:, :, z_index, Channel_index]
#
# # Display the image
# plt.imshow(image_array_slice, cmap='gray')  # Use 'cmap' parameter to specify colormap (e.g., 'gray')
# plt.axis('off')  # Turn off axes
# plt.show()

# now comes the neural network
from model_denoise import DenoiseNet

net = DenoiseNet()

from custom_dataset import CustomDataset

train_ds = CustomDataset(PicsBatch)

# # plotting an example
# x, y = train_ds[10]
# x_plot = x.numpy()
# y_plot = y.numpy()
# z_index = 3
# Channel_index = 2
# # x = x[:, :, :, Channel_index]
# # y = y[:, :, :, Channel_index] now every channel is it's own image
# x_plot = np.max(x_plot, axis=-1)
# y_plot = np.max(y_plot, axis=-1)
# fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=150)
#
# ax[0].imshow(x_plot, cmap='Greys_r')
# ax[0].set_title('Target: Clean image of only the spots')
# ax[1].imshow(y_plot, cmap='Greys_r')
# ax[1].set_title('Input: Noised original image')
#
# # Adjust spacing between subplots for better visualization
# plt.tight_layout()
#
# plt.show()

from denoise_training import train_valid_loop

train_loss, valid_loss = train_valid_loop(net, 4, Nepochs=5, learning_rate=0.001)

# Load the best model from training

net.load_state_dict(torch.load('saved_model.pt', map_location=torch.device('cpu')))

# Plot the training and validation losses

fig, axes = plt.subplots()
axes.plot(range(len(train_loss)), train_loss, label='train loss')
axes.plot(range(len(valid_loss)), valid_loss, label='valid loss')
axes.set_yscale('log')
axes.legend()
plt.show()
