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
        arbitrator_layer_size = intermediate_num_channels * image_size_after_conv_layers

        intermediate_linear_size = arbitrator_layer_size // 3
        output_channels = 1

        self.intermediate_num_channels = intermediate_num_channels
        self.image_size_after_conv_layers = image_size_after_conv_layers
        self.arbitrator_layer_size = arbitrator_layer_size
        self.intermediate_linear_size = intermediate_linear_size
        self.output_channels = output_channels

        self.cnn_layers = nn.Sequential(
            nn.Conv3d(in_channels=num_input_channels, out_channels=intermediate_num_channels, kernel_size=4, padding="same"),
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
                      out_channels=intermediate_num_channels, kernel_size=(2, 3, 3), padding="valid"),  # kernel_size: (kz,kx,ky)
            nn.BatchNorm3d(intermediate_num_channels),
            nn.ReLU(),

            nn.Conv3d(in_channels=intermediate_num_channels,
                      out_channels=intermediate_num_channels, kernel_size=(2, 3, 3), padding="valid"),  # kernel_size: (kz,kx,ky)
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
            nn.Linear(intermediate_linear_size, intermediate_linear_size),
            nn.ReLU(),
            nn.Linear(intermediate_linear_size, 100),
            nn.ReLU(),
            nn.Linear(100, output_channels),
            nn.Flatten(0))

    def forward(self, image, small_image_coordinates):
        channel_coordinates = small_image_coordinates[:, -1]  # channel is the last coordinate
        after_cnn = self.cnn_layers(image)

        # # apply a different arbitrator layer for input's in the batch with different channel_coordinate.
        # # we loop over all the possible values of channel_coordinates, for each one we take all the images
        # # in the batch with the current channel_coordinates, and apply the current channel_coordinate's
        # # arbitrator_layer on them.
        # after_arbitrator = torch.full_like(after_cnn, torch.nan)
        # for current_channel_coordinate in range(torch.max(channel_coordinates) + 1):
        #     # the index of the images in the batch that have the correct channel
        #     indexes_with_current_channel = channel_coordinates == current_channel_coordinate
        #     after_arbitrator[indexes_with_current_channel, :] = \
        #         self.arbitrator_layers[current_channel_coordinate](after_cnn[indexes_with_current_channel, :])

        # # make sure all of after_arbitrator was filled out.
        # assert not torch.any(torch.isnan(after_arbitrator))

        rslt = self.linear_layers(after_cnn)
        return rslt
