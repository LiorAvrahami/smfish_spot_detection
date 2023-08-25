
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

class spots_classifier_net(nn.Module):

    def __init__(self):
        super().__init__()
        channel_num = 10
        ### Add your network layers here...
        ### You should use nn.Conv3d(), nn.BatchNorm2d(), and nn.ReLU()
        ### They can be added as seperate layers and cascaded in the forward
        ### or you can combine them using the nn.Sequential() class and an OrderedDict (very clean!)

        self.model = nn.Sequential(OrderedDict([
            ('conv1-1', nn.Conv3d(in_channels=3, out_channels=channel_num, kernel_size=3, padding="same")),
            ('relu1-1', nn.ReLU()),
            ('conv1-2', nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('relu1-2', nn.ReLU()),
            ('drop1',nn.Dropout(p=0.15)),

            ('conv2-1', nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('relu2-1', nn.ReLU()),
            ('conv2-2', nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('relu2-2', nn.ReLU()),
            ('drop2',nn.Dropout(p=0.15)),

            ('max_pool1', nn.MaxPool3d(2,2)),
            ('batch_norm_1', nn.BatchNorm3d(channel_num)),

            ('conv3-1', nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('relu3-1', nn.ReLU()),
            ('conv3-2', nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, padding="same")),
            ('relu3-2', nn.ReLU()),
            ('drop3',nn.Dropout(p=0.15)),

            ('sig_out', nn.Sigmoid()),
        ]))
        
        
    def forward(self):
        
        ### Now pass the input image x through the network layers
        ### Then add the result to the input image (to offset the noise

        return self.model()