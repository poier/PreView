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

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredViewAndHaPeNetDcGan(nn.Module):
    """
    Encoder-Decoder net, based on DC-GAN architecture [1].
    
    [1] Radford et al., ICLR 2016, https://arxiv.org/pdf/1511.06434.pdf
    """
    
    def __init__(self, do_use_gpu=True, num_prior_dim=30, num_cond_dim=3,
                 num_output_dim=(3*24), num_features=64):
        """
        
        Arguments:
            do_use_gpu (boolean, optional): compute on GPU (=default) or CPU?
            num_prior_dim (int, optional): number of dimensions in 
                "bottleneck"/code layer; default: 30
            num_cond_dim (int, optional): number of dimensions of the vector 
                on which the decoder is conditioned (additional to the "prior")
                default: 3
            num_output_dim (int, optional): number of output dimensions; 
                e.g., 3D coordinates of joint positions, model parameters, ...;
                default: 72
            num_features (int, optional): number of feature channels in the 
                first/lowest layer (highest resolution), it is increased 
                inversely proportional with downscaling at each layer; 
                default: 64
        """
        super(PredViewAndHaPeNetDcGan, self).__init__()
        
        self.do_use_gpu = do_use_gpu
        nf = num_features

        # Encoder
        self.conv1 = nn.Conv2d(1,   nf,      (4, 4), stride=2, padding=1, 
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
        self.conv5 = nn.Conv2d((nf * 8), num_prior_dim, (4, 4), stride=1, 
                               padding=0, bias=False)

        # Decoder
        self.convt1 = nn.ConvTranspose2d((num_prior_dim + num_cond_dim), 
#        self.convt1 = nn.ConvTranspose2d(num_prior_dim, 
                                         (nf * 8), (4, 4), 
                                         stride=1, padding=0, bias=False)
        self.bn1_d = nn.BatchNorm2d(nf * 8)
        self.convt2 = nn.ConvTranspose2d((nf * 8), (nf * 4), (4, 4), 
                                         stride=2, padding=1, bias=False)
        self.bn2_d = nn.BatchNorm2d(nf * 4)
        self.convt3 = nn.ConvTranspose2d((nf * 4), (nf * 2), (4, 4), 
                                         stride=2, padding=1, bias=False)
        self.bn3_d = nn.BatchNorm2d(nf * 2)
        self.convt4 = nn.ConvTranspose2d((nf * 2), nf, (4, 4), 
                                         stride=2, padding=1, bias=False)
        self.bn4_d = nn.BatchNorm2d(nf)
        self.convt5 = nn.ConvTranspose2d(nf, 1, (4, 4), 
                                         stride=2, padding=1, bias=False)
                                         
        # Additional layer
        self.conv6 = nn.Conv2d(num_prior_dim, num_output_dim, (1, 1), stride=1, 
                               padding=0, bias=True)

        
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = F.tanh(self.conv5(x))
        return x
        

    def decode(self, z, com):
        # Add two singleton dimensions; BxD => BxDx1x1
        com = com.unsqueeze(2).unsqueeze(2)
        z = torch.cat((z, com), 1)
        y = self.leakyrelu(self.bn1_d(self.convt1(z)))
        y = self.leakyrelu(self.bn2_d(self.convt2(y)))
        y = self.leakyrelu(self.bn3_d(self.convt3(y)))
        y = self.leakyrelu(self.bn4_d(self.convt4(y)))
        y = F.tanh(self.convt5(y))
        return y
        
        
    def regress(self, z):
#        y = F.tanh(self.conv6(z))
        y = self.conv6(z)
        return y
        

    def forward(self, x, com=None):
        """
        Arguments:
            com (Tensor, optional): conditional for generator/decoder; 
                if None no image is generated
                default: None
        """
        z = self.encode(x)
        o = None
        if not com is None:
            o = self.decode(z, com)
        y = self.regress(z)
        return y, o, z
        
        