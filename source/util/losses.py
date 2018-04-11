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
from torch.autograd import Variable

import numpy as np
from scipy import ndimage
from enum import IntEnum


#%% General definitions
class LossType(IntEnum):
    L1 = 0
    L2 = 1
    HUBER = 2


#%%
def joint_pos_distance_loss(estimate, target, size_average=True):
    """
    Joint distances (L2-norm per joint, i.e., not squared)
    
    Arguments:
        estimate (torch.autograd.Variable) estimated positions, BxDx1x1,
            where D is the number of dimensions, i.e., N*3, 
            where N is the number of joint positions
        target (torch.autograd.Variable) target positions, BxNx3, 
            where N is the number of joint positions
    """
    divisor = (target.size()[0] * target.size()[1]) if size_average else 1.0
    
    est = estimate.view(target.size())
    return torch.sum(torch.sqrt(torch.pow((est - target), 2).sum(dim=2, keepdim=False))) / divisor


def joint_pos_huber_loss(estimate, target, size_average=True, 
                         ignore_zero_label_samples=False, delta=1):
    """
    Compute the sum of Huber loss applied separately to the error (distance 
    to ground truth) of each joint position
    
    Arguments:
        estimate (torch.autograd.Variable) estimated positions, BxDx1x1,
            where D is the number of dimensions, i.e., N*3, 
            where N is the number of joint positions
        target (torch.autograd.Variable) target positions, BxNx3, 
            where N is the number of joint positions
        size_average (bool, optional) default: True
        ignore_zero_label_samples (bool, optional) if True the loss for samples 
            with the label (vector) being zero is ignored; default: False
        delta (float, optional) distance where the loss switches from quadratic 
            to linear
    """

    est = estimate.view(target.size())
    dist_sq = torch.pow((est - target), 2).sum(dim=2, keepdim=False)
    dist = torch.sqrt(dist_sq)
    switch = (dist < delta).float()
    huber = switch * 0.5 * dist_sq + (1 - switch) * delta * (dist - 0.5 * delta)
    if ignore_zero_label_samples:
        loss = huber.sum(dim=1, keepdim=False)
        # Ignore loss for "zero labels" (=> target with all values zero)
        do_use = (torch.abs(target).sum(dim=2, keepdim=False).sum(dim=1, keepdim=False) > 0).float()
        loss = do_use * loss
        # Divide only through valid samples (i.e., non zero label); Note, divisor might become zero (loss should also be zero then)
        divisor = (torch.sum(do_use) * target.size()[1]) if size_average else Variable(torch.ones(1))
        loss = torch.sum(loss) if (divisor.data < 1.0).all() else torch.sum(loss) / divisor
    else:
        divisor = (target.size()[0] * target.size()[1]) if size_average else 1.0
        loss = torch.sum(huber) / divisor
        
    return loss


def huber_loss(estimate, target, size_average=True, 
                         do_ignore_background=False, 
                         delta=1,
                         do_erode_background=False):
    """
    Compute the Huber loss.
    Note, in principle, this is the same as nn.SmoothL1Loss but with 
    e.g., specifiable delta (delta is fixed to 1 in the implementation of 
    nn.SmoothL1Loss). That is:
    
                              { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < delta
        loss(x, y) = 1/n \sum {
                              { delta * (|x_i - y_i| - 0.5 * delta),  otherwise
    
    `x` and `y` arbitrary shapes with a total of `n` elements each
    the sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal variable
    `size_average` to `False`
    
    Arguments:
        estimate (torch.autograd.Variable) 
        target (torch.autograd.Variable) 
        size_average (bool, optional) default: True
        do_ignore_background (bool, optional) if True the loss for samples 
            with the target value being one is ignored; default: False
        delta (float, optional) distance where the loss switches from quadratic 
            to linear
    """
    
    est = estimate.view(target.size())
    dist_abs = torch.abs(est - target)
    dist_sq = torch.pow(dist_abs, 2)
    switch = (dist_abs < delta).float()
    huber = switch * 0.5 * dist_sq + (1 - switch) * delta * (dist_abs - 0.5 * delta)
    if do_ignore_background:
        loss = huber
        # Ignore loss for background (=> target 1)
        do_use = (target < 1.0).float()
        
        if do_erode_background:
            # We just dilate the foreground
            do_use_gpu = do_use.is_cuda
            ksize = 7
            struct_el = np.ones((1,1,ksize,ksize)) == 1
            do_use = Variable(torch.from_numpy(
                ndimage.binary_dilation(
                    do_use.data.cpu().numpy(), struct_el).astype(np.float32)))
            if do_use_gpu:
                do_use = do_use.cuda()
            
        loss = do_use * loss
        # Divide only through valid samples (i.e., non zero label); 
        # Note, divisor might become zero (loss should also be zero then)
        divisor = torch.sum(do_use) if size_average else Variable(torch.ones(1))
        loss = torch.sum(loss) if (divisor.data < 1.0).all() else torch.sum(loss) / divisor
    else:
        # Compute dimensionality (get rid of loop?)
        d = 1
        for i in target.size():
            d *= i
        divisor = d if size_average else 1.0
        loss = torch.sum(huber) / divisor
        
    return loss
    