import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict


class DenoiseNet(nn.Module):

    # we put only 1 channel in the network at a time
    def __init__(self):
        super().__init__()
        num_of_features = 8  # for now

        self.model = nn.Sequential(OrderedDict([
            ('conv1-1', nn.Conv3d(in_channels=1, out_channels=num_of_features, kernel_size=3, padding="same")),
            ('batch_norm_1', nn.BatchNorm3d(num_of_features)),
            ('relu1', nn.ReLU()),

            ('conv2-1', nn.Conv3d(in_channels=num_of_features, out_channels=num_of_features, kernel_size=3, padding="same")),
            ('batch_norm_2', nn.BatchNorm3d(num_of_features)),
            ('relu2', nn.ReLU()),

            ('conv3-1', nn.Conv3d(in_channels=num_of_features, out_channels=num_of_features, kernel_size=3, padding="same")),
            ('batch_norm_3', nn.BatchNorm3d(num_of_features)),
            ('relu3', nn.ReLU()),

            ('conv4-1', nn.Conv3d(in_channels=num_of_features, out_channels=num_of_features, kernel_size=3, padding="same")),
            ('batch_norm_4', nn.BatchNorm3d(num_of_features)),
            ('relu4', nn.ReLU()),

            ('conv5-1',
             nn.Conv3d(in_channels=num_of_features, out_channels=num_of_features, kernel_size=3, padding="same")),
            ('batch_norm_5', nn.BatchNorm3d(num_of_features)),
            ('relu5', nn.ReLU()),

            ('conv_out', nn.Conv3d(in_channels=num_of_features, out_channels=1, kernel_size=3, padding="same")),
        ]))

    def forward(self, x):
        y = self.model(x)
        return y
