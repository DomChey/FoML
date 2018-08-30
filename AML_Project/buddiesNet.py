'''
Neural Network to determine DNN Buddies

@author: Dominique Cheray
'''

import torch.nn as nn
import torch.nn.functional as F

class BuddiesNet(nn.Module):
    """Implementation of the neural net for the DNN Buddies"""
    def __init__(self):
        """Build the nertwork as described in the paper
        https://arxiv.org/pdf/1711.08762.pdf"""
        super(BuddiesNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(336, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 2))


    def forward(self, x):
        """Forward pass through BuddiesNet"""
        out = self.layers(x)
        return out
