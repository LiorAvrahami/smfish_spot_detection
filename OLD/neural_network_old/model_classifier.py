import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

from image_manipulation.crop_image_for_classifier import small_image_shape


class spots_classifier_net(nn.Module):

    def __init__(self, num_input_channels):
        super().__init__()
        intermediate_num_channels = 5 * num_input_channels
        output_channels = 1

        image_size_after_conv_layers = (small_image_shape[0] - 6) * \
            (small_image_shape[1] - 6) * (small_image_shape[2] - 6)
        arbitrator_layer_size = intermediate_num_channels * image_size_after_conv_layers

        intermediate_linear_size = arbitrator_layer_size // 3

        self.cnn_layers = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=intermediate_num_channels, kernel_size=4, padding="same"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.ReLU(),

            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=3, padding="same"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.ReLU(),


            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=3, padding="same"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.ReLU(),

            # start decreasing image size with padding="valid" in order to increase information density before linear layers
            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=3, padding="valid"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.ReLU(),

            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=3, padding="valid"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.ReLU(),

            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=3, padding="valid"),
            nn.BatchNorm3d(intermediate_num_channels),
            nn.ReLU(),

            # flatten out the image in order for it to enter the linear layers
            nn.Flatten(),
        )

        # arbitrator layer, this layer is the conditional CNN. here we use a different layer, with different weights depending on
        # the value of the roi channel.
        self.arbitrator_layers = nn.ModuleList(
            [nn.Linear(arbitrator_layer_size, arbitrator_layer_size) for i in range(num_input_channels)]
        )

        # finally we pass through several linear layers in order to reach a binary classification
        self.linear_layers = nn.Sequential(
            nn.Linear(arbitrator_layer_size, intermediate_linear_size),
            nn.ReLU(),
            nn.Linear(arbitrator_layer_size, intermediate_linear_size),
            nn.ReLU(),
            nn.Linear(intermediate_linear_size, 100),
            nn.ReLU(),
            nn.Linear(100, output_channels),
            nn.Flatten(0))

    def forward(self, image, small_image_coordinates):
        channel = small_image_coordinates[-1]  # channel is the last coordinate
        after_cnn = self.cnn_layers(image)
        after_arbitrator = self.arbitrator_layers[channel](after_cnn)
        rslt = self.linear_layers(after_arbitrator)
        return rslt
