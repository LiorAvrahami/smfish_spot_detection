import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict


class spots_classifier_net(nn.Module):

    def __init__(self):
        super().__init__()
        channel_num = 10
        output_channels = 1

        self.model = nn.Sequential(OrderedDict([
            ('conv1-1', nn.Conv3d(in_channels=3, out_channels=channel_num, kernel_size=3, padding="same")),
            ('batch_norm_1', nn.BatchNorm3d(channel_num)),
            ('relu1', nn.ReLU()),

            ('conv2-1',
             nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('batch_norm_2', nn.BatchNorm3d(channel_num)),
            ('relu2', nn.ReLU()),

            ('conv3-1',
             nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('batch_norm_3', nn.BatchNorm3d(channel_num)),
            ('relu3', nn.ReLU()),

            ('conv4-1',
             nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('batch_norm_4', nn.BatchNorm3d(channel_num)),
            ('relu4', nn.ReLU()),

            ('conv5-1',
             nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('batch_norm_5', nn.BatchNorm3d(channel_num)),
            ('relu5', nn.ReLU()),

            ('conv_out', nn.Conv3d(in_channels=channel_num, out_channels=1, kernel_size=3, padding="same")),

            ('flatten', nn.Flatten()),
            ('fcn1', nn.Linear(1250, 500)),
            ('relu6', nn.ReLU()),
            ('fcn2', nn.Linear(500, 100)),
            ('relu7', nn.ReLU()),
            ('fcn3', nn.Linear(100, output_channels)), ('flatten2', nn.Flatten(0))]))


    def forward(self, x):
        rslt = self.model(x)
        return rslt
