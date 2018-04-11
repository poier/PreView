"""
Copyright 2018 ICG, Graz University of Technology

This file is part of PreView.

PreView is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PreView is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PreView.  If not, see <http://www.gnu.org/licenses/>.
"""

import torch.nn as nn

from enum import IntEnum


#%% General definitions
class ConditioningType(IntEnum):
    OFF = 0                 # no conditioning
    INPUT = 1               # condition on input
#    POSE_EMBEDDING = 2      # condition on learned embedding => for now we don't have the "GT"/real embedding for samples
    POSE = 3                # condition on pose (joint positions)
    INPUT_AND_POSE = 4      # condition on input and pose


#%% Network definition
class DiscriminatorNetDcGan(nn.Module):
    """
    Discriminator of DC-GAN architecture [1].
    
    [1] Radford et al., ICLR 2016, https://arxiv.org/pdf/1511.06434.pdf
    """
    
    def __init__(self, do_use_gpu=True, num_features=64, num_input_channels=1):
        """
        
        Arguments:
            do_use_gpu (boolean, optional): compute on GPU (=default) or CPU?
            num_features (int, optional): number of feature channels in the 
                first/lowest layer (highest resolution), it is increased 
                inversely proportional with downscaling at each layer; 
                default: 64
            num_input_channels (int, optional): number of channels in the input 
                image (more channels can also be used to 
                create a conditional discriminator)
                default: 1
        """
        super(DiscriminatorNetDcGan, self).__init__()
        
        self.do_use_gpu = do_use_gpu
        nf = num_features

        # Encoder
        self.conv1 = nn.Conv2d(num_input_channels, nf, (4, 4), stride=2, padding=1, 
                               bias=False)
        self.conv2 = nn.Conv2d(nf, (nf * 2), (4, 4), stride=2, padding=1, 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(nf * 2)
        self.conv3 = nn.Conv2d((nf * 2), (nf * 4), (4, 4), stride=2, padding=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(nf * 4)
        self.conv4 = nn.Conv2d((nf * 4), (nf * 8), (4, 4), stride=2, padding=1, 
                               bias=False)
        self.bn4 = nn.BatchNorm2d(nf * 8)
        self.conv5 = nn.Conv2d((nf * 8), 1, (4, 4), stride=1, 
                               padding=0, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.bn2(self.conv2(x)))
        x = self.leakyrelu(self.bn3(self.conv3(x)))
        x = self.leakyrelu(self.bn4(self.conv4(x)))
#        x = F.sigmoid(self.conv5(x))
        x = self.conv5(x)   # for e.g., least-squares GAN/loss we don't want a sigmoid
        return x
        