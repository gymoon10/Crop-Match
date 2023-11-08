"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet_crop as network

class ERFNet_Semantic_Crop(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        print('Creating Branched ERFNet_Semantic_Crop with {} classes'.format(num_classes))

        self.encoder = network.Encoder()
        self.decoder = network.Decoder()

    def forward(self, input_):

        feat_enc = self.encoder(input_)  # (N, 128, 64, 64)
        output, feat_dec = self.decoder.forward(feat_enc)  # (N, 3, 512, 512) / (N, 16, 256, 256)

        return output, feat_dec