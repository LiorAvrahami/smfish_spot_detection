import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

from image_manipulation.crop_image_for_classifier import small_image_shape


class spots_classifier_net(nn.Module):

    def __init__(self, num_input_channels):
        super().__init__()
        intermediate_num_channels = 7 * num_input_channels

        image_size_after_conv_layers = (small_image_shape[0] - 6) * \
            (small_image_shape[1] - 6) * (small_image_shape[2] - 4)
        linear_input_size = intermediate_num_channels * image_size_after_conv_layers + 1

        intermediate_linear_size = linear_input_size // 3
        output_channels = 1

        self.intermediate_num_channels = intermediate_num_channels
        self.image_size_after_conv_layers = image_size_after_conv_layers
        self.linear_input_size = linear_input_size
        self.intermediate_linear_size = intermediate_linear_size
        self.output_channels = output_channels

        self.cnn_layers = nn.Sequential(
            nn.Conv3d(in_channels=num_input_channels, out_channels=intermediate_num_channels, kernel_size=4, padding="same"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=3, padding="same"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.LeakyReLU(),


            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=3, padding="same"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.LeakyReLU(),

            # start decreasing image size with padding="valid" in order to increase information density before linear layers
            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=3, padding="valid"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=(2, 3, 3), padding="valid"),  # kernel_size: (kz,kx,ky)
            nn.BatchNorm3d(intermediate_num_channels),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=(2, 3, 3), padding="valid"),  # kernel_size: (kz,kx,ky)
            nn.BatchNorm3d(intermediate_num_channels),
            nn.LeakyReLU(),

            # flatten out the image in order for it to enter the linear layers
            nn.Flatten(),
        )

        # finally we pass through several linear layers in order to reach a binary classification
        self.linear_layers = nn.Sequential(
            nn.Linear(linear_input_size, intermediate_linear_size),
            nn.LeakyReLU(),
            nn.Linear(intermediate_linear_size, intermediate_linear_size),
            nn.LeakyReLU(),
            nn.Linear(intermediate_linear_size, 100),
            nn.LeakyReLU(),
            nn.Linear(100, output_channels),
            nn.Flatten(0))

    def forward(self, image, small_image_coordinates):
        channel_coordinates = small_image_coordinates[:, -1]  # channel is the last coordinate
        channel_coordinates = channel_coordinates.unsqueeze(1)
        after_cnn = self.cnn_layers(image)
        after_cnn = torch.cat((after_cnn, channel_coordinates), dim=1)  # concatenate the labels to the feature vector
        rslt = self.linear_layers(after_cnn)
        return rslt
