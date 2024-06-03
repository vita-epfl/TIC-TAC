import logging

import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter


class AuxNet_HG(nn.Module):
    def __init__(self, arch):
        """
        Auxiliary network which predicts flattened matrix using intermediate outputs of the Hourglass
        """
        super(AuxNet_HG, self).__init__()

        self.fc_arch = arch['fc']

        # Derived from Hourglass
        self.conv_arch_spatial = arch['spatial_dim']
        self.conv_arch_channels = arch['channels']

        # List that houses the network
        self.pytorch_layers = []

        self.pytorch_layers.append(
            ConvolutionFeatureExtractor(channels=self.conv_arch_channels))

        # Initializing for input-output chaining across layers
        input_nodes_fc_network = arch['channels'][-1]

        in_feat = input_nodes_fc_network
        for out_feat in self.fc_arch:
            self.pytorch_layers.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            self.pytorch_layers.append(nn.ReLU())
            in_feat = out_feat

        self.pytorch_layers = self.pytorch_layers[:-1]  # Removing the ReLU after the output layer
        self.pytorch_layers = nn.ModuleList(self.pytorch_layers)


    def forward(self, x):
        """

        :param x:
        :return:
        """

        # Conv feature extractor
        # Restoring heatmaps
        with torch.no_grad():
            conv_x = []
            border = 0

            for size in self.conv_arch_spatial:
                conv_x.append(x[:, :, border: border + (size**2)].reshape(x.shape[0], x.shape[1], size, size))
                border += (size**2)

        x = self.pytorch_layers[0](conv_x)
        # [1:] skips the ConvFeatExtract layer
        for layer in self.pytorch_layers[1:]:
            x = layer(x)

        return x


class ConvolutionFeatureExtractor(nn.Module):
    def __init__(self, channels):
        super(ConvolutionFeatureExtractor, self).__init__()

        self.hg_conv_feat_extract = []
        self.depth = len(channels)

        # Down from 64 to 4
        for i in range(self.depth-1):    # 32 --> 16, 16 --> 8, 8 --> 4
            self.hg_conv_feat_extract.append(
                torch.nn.Conv2d(
                    in_channels=channels[i], out_channels=channels[i+1],
                    kernel_size=(2, 2), stride=2, padding=0))
            
            self.hg_conv_feat_extract.append(torch.nn.ReLU())

        self.hg_conv_feat_extract = nn.ModuleList(self.hg_conv_feat_extract)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        x_ = x[0]
        for i in range(self.depth - 1):
            x_ = self.hg_conv_feat_extract[2 * i](x_)
            x_ = self.hg_conv_feat_extract[(2 * i) + 1](x_)
            x_ = x[i+1] + x_

        return x_.squeeze()


class AuxNet_ViTPose(nn.Module):
    def __init__(self, arch):
        """
        Auxiliary network which predicts flattened matrix using intermediate outputs of the Hourglass
        """
        super(AuxNet_ViTPose, self).__init__()

        self.fc_arch = arch['fc']

        # List that houses the network
        self.pytorch_layers = []

        # Initializing for input-output chaining across layers
        input_nodes_fc_network = 64

        in_feat = input_nodes_fc_network
        for out_feat in self.fc_arch:
            self.pytorch_layers.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            self.pytorch_layers.append(nn.ReLU())
            in_feat = out_feat

        self.pytorch_layers = self.pytorch_layers[:-1]  # Removing the ReLU after the output layer
        self.pytorch_layers = nn.Sequential(*self.pytorch_layers)


    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = x.squeeze()
        x = self.pytorch_layers(x)
        return x